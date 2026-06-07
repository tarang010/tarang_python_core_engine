"""
Tarang 2.2.0 — file7_captions.py
=====================================
STATELESS: Pure compute engine. No local file reads or writes.
Accepts text string + duration, returns captions as list of dicts.
Express stores captions in MongoDB Document.captions.

Changes in v2.2.0:
  FIX 11 — CRITICAL: TTS_CHUNK_SIZE corrected from 700 → 950 to exactly match
            file2_tts.py CHUNK_SIZE = 950. This was the root cause of the
            audio/captions desync (28-min audio with captions ending at 4 min).
            The chunk-boundary simulation uses this value to count how many
            INTER_CHUNK_SILENCE gaps exist; undercount compounded catastrophically
            over long documents.
  FIX 12 — INTER_CHUNK_SILENCE corrected from 0.95s → 0.22s to match file2_tts
            _PAUSE["between_chunks"] = 220ms. The old value was 4× too large,
            causing the total_gap_time to vastly exceed real silence in the audio,
            which compressed speech_time to near zero and collapsed all timestamps
            to the first few minutes.
  FIX 13 — Heading/section silences now modelled in gap accounting. file2_tts
            emits 900ms before + 600ms after each heading chunk. These are now
            included in total_gap_time so heading-heavy documents don't drift.

Changes in v2.1.0:
  FIX 1  — Syllable counter: abbreviation guard, silent-e edge cases, and
            punctuation pauses handled as separate time offsets (not fake
            syllables added to the count, which mixed units).
  FIX 2  — Chunk boundary simulation rewritten to match file2_tts sentence-
            aware chunking, not raw character offset chunking.
  FIX 3  — Merged-sentence duration: short sentences merged for display now
            carry the sum of their individual durations, not a recomputed share.
  FIX 4  — Abbreviation-aware sentence splitter: "e.g.", "i.e.", "Dr.", "vs.",
            "etc." and similar no longer split mid-sentence.
  FIX 5  — Number → spoken-form syllable expansion: years (1985 → 5 syl),
            money ($42 → 4 syl), ordinals (3rd → 2 syl), plain integers
            and decimals converted to approximate spoken syllable count.
  FIX 6  — Gap weight mapping uses sentence-text lookup, not raw character
            offsets, so weights land on the correct sentence after merging/
            splitting.
  FIX 7  — Clause splitter: minimum clause length (MIN_CLAUSE_WORDS = 5)
            prevents 1-2 word fragments; maximum recursion guard splits
            clauses that are still too long at conjunctions.
  FIX 8  — Logging: added avg_caption_duration, min_duration, max_duration
            to the caption OK log line so outliers are immediately visible.
  FIX 9  — Minimum caption display time: MIN_CAPTION_DURATION_SEC = 1.2s
            floor applied after proportional calculation; surplus distributed
            to adjacent captions.
  FIX 10 — speech_time lower bound: removed the arbitrary 0.85 floor and
            replaced with exact gap subtraction, clamped so speech_time
            is never less than 60% of duration_sec (handles edge cases with
            many chunks gracefully without silently compressing timing).
"""

import re
import math
import logging

logger = logging.getLogger("file7_captions")

# ── Constants ─────────────────────────────────────────────────────────────────
# FIX 11: Must exactly match file2_tts.py CHUNK_SIZE = 950
# Previously 700 — caused chunk count undercount → wrong total_gap_time → full desync
TTS_CHUNK_SIZE          = 950    # must match file2_tts CHUNK_SIZE exactly

# FIX 12: Must match file2_tts._PAUSE["between_chunks"] = 220ms = 0.22s
# Previously 0.95s — was 4× too large, compressed all captions into first ~4 min
INTER_CHUNK_SILENCE     = 0.22   # seconds — matches file2_tts between_chunks pause

# FIX 13: Heading silences from file2_tts._PAUSE: before=900ms, after=600ms
# These fire for every heading chunk and must be included in gap accounting
HEADING_PAUSE_BEFORE    = 0.90   # seconds — file2_tts _PAUSE["before_heading"]
HEADING_PAUSE_AFTER     = 0.60   # seconds — file2_tts _PAUSE["after_heading"]
SECTION_PAUSE           = 0.50   # seconds — file2_tts _PAUSE["between_sections"] (transitions)

MAX_SENTENCE_WORDS      = 20     # split longer sentences at clause boundaries
MIN_SENTENCE_WORDS      = 4      # merge shorter sentences with the next
MIN_CLAUSE_WORDS        = 5      # FIX 7: prevent tiny clause fragments
MIN_CAPTION_DURATION_SEC = 1.2   # FIX 9: no caption shorter than this

# Acronym/abbreviation syllable table (shared with file2_tts)
ACRONYM_SYLLABLES = {
    "SQL":3,"API":3,"HTML":4,"CSS":3,"HTTP":4,"HTTPS":5,"URL":3,
    "CPU":3,"GPU":3,"RAM":3,"ROM":3,"SDK":3,"IDE":3,"CLI":3,"GUI":3,
    "JSON":4,"XML":3,"YAML":4,"REST":4,"OOP":3,"AWS":3,"GCP":3,
    "NLP":3,"AI":2,"ML":2,"TTS":3,"OCR":3,"PDF":3,"DNA":3,"RNA":3,
    "ATP":3,"IN":2,"NCC":3,"NDA":3,"IMA":3,"OTA":3,"CDS":3,"EW":2,
}

# FIX 4: abbreviations that must NOT trigger a sentence split
_ABBREV_PATTERN = re.compile(
    r'\b(?:e\.g|i\.e|etc|vs|Dr|Mr|Mrs|Ms|Prof|Sr|Jr|St|Ave|Blvd|Dept|approx|'
    r'est|min|max|approx|no|vol|ch|pp|fig|eq|sec|Jan|Feb|Mar|Apr|Jun|Jul|Aug|'
    r'Sep|Oct|Nov|Dec)\.',
    re.IGNORECASE
)

# Sentence / clause splitters
_SENTENCE_END    = re.compile(r'(?<=[.!?])\s+')
_CLAUSE_BOUNDARY = re.compile(
    r'(?<=,)\s+|(?<=;)\s+|(?<=:)\s+'
    r'|(?=\s+(?:and|but|or|because|however|therefore|although|while|when|if|since)\s)'
)


# ══════════════════════════════════════════════════════════════════════════════
# FIX 1 + FIX 5 — Syllable counting
# ══════════════════════════════════════════════════════════════════════════════

def _int_to_syllables(n: int) -> int:
    """
    FIX 5: Approximate syllable count for an integer as spoken aloud.
    Uses the fact that English number words have predictable syllable patterns.
    """
    if n < 0:
        return 2 + _int_to_syllables(abs(n))  # "negative X"
    if n == 0:
        return 1  # "zero"

    # Syllable counts for 0-19
    _ones = [0,1,1,1,1,1,2,2,1,1,2,3,2,2,2,2,2,3,2,2]
    # Syllable counts for tens (20,30,...90)
    _tens = [0,0,2,2,2,2,2,3,2,2]

    def _below_1000(x):
        if x == 0:    return 0
        if x < 20:    return _ones[x]
        if x < 100:   return _tens[x // 10] + _ones[x % 10]
        # hundreds: e.g. "three hundred and forty-two" = 1+2+1+syl(42)
        hundreds  = 1 + 2  # "X hundred"
        remainder = x % 100
        if remainder:
            hundreds += 1 + _below_1000(remainder)  # "and ..."
        return hundreds

    total = 0
    if n >= 1_000_000_000:
        total += _below_1000(n // 1_000_000_000) + 3  # "billion"
        n %= 1_000_000_000
    if n >= 1_000_000:
        total += _below_1000(n // 1_000_000) + 3       # "million"
        n %= 1_000_000
    if n >= 1000:
        total += _below_1000(n // 1000) + 2            # "thousand"
        n %= 1000
    total += _below_1000(n)
    return max(1, total)


def _token_syllables(token: str) -> int:
    """
    FIX 1 + FIX 5: Syllable count for a single token.
    Handles: acronyms, numbers (years, money, ordinals, plain ints, decimals),
    and regular words with improved vowel-run counting.
    """
    raw   = token.strip(".,!?;:\"'()-—[]")
    upper = raw.upper()

    # Acronym table lookup
    if upper in ACRONYM_SYLLABLES:
        return ACRONYM_SYLLABLES[upper]

    # FIX 5: currency — "$42" → "forty-two dollars"
    money = re.match(r'^[$€£¥](\d+)(?:\.(\d+))?$', raw)
    if money:
        main  = _int_to_syllables(int(money.group(1)))
        cents = _int_to_syllables(int(money.group(2))) + 1 if money.group(2) else 0
        return main + 2 + cents  # "X dollars [and Y cents]"

    # FIX 5: ordinals — "3rd", "21st", "42nd"
    ordinal = re.match(r'^(\d+)(?:st|nd|rd|th)$', raw, re.IGNORECASE)
    if ordinal:
        return _int_to_syllables(int(ordinal.group(1)))  # ordinal suffix is silent

    # FIX 5: year pattern — 4-digit number 1000-2099 → spoken as two pairs
    # "1985" → "nineteen eighty-five" ≈ 5 syllables
    year = re.match(r'^(1\d{3}|20\d{2})$', raw)
    if year:
        y     = int(raw)
        first = _int_to_syllables(y // 100)   # "nineteen"
        last  = _int_to_syllables(y % 100)    # "eighty-five"
        return first + last

    # FIX 5: plain integer
    if re.match(r'^\d+$', raw):
        return _int_to_syllables(int(raw))

    # FIX 5: decimal — "3.14" → "three point one four"
    decimal = re.match(r'^(\d+)\.(\d+)$', raw)
    if decimal:
        left  = _int_to_syllables(int(decimal.group(1)))
        right = sum(_int_to_syllables(int(d)) for d in decimal.group(2))  # digit by digit
        return left + 1 + right  # "point"

    # Regular word — improved vowel-run count
    word = raw.lower()
    if not word:
        return 1

    # FIX 1: handle common silent-e edge cases before generic rule
    word = re.sub(r'([^aeiou])e$', r'\1', word)   # drop silent terminal e
    word = re.sub(r'([^aeiou])ed$', r'\1', word)  # dropped -ed (walked, talked)

    vowels = "aeiouy"
    count  = 0
    prev_v = False
    for ch in word:
        is_v = ch in vowels
        if is_v and not prev_v:
            count += 1
        prev_v = is_v

    return max(1, count)


def count_syllables_sentence(sentence: str) -> int:
    """
    FIX 1: Returns integer syllable count only — no fake pause syllables.
    Punctuation pauses are handled as time offsets in _pause_offset_sec().
    """
    return max(1, sum(_token_syllables(w) for w in sentence.split()))


def _pause_offset_sec(sentence: str) -> float:
    """
    FIX 1: Separate pause time in seconds for punctuation within a sentence.
    This used to be added as fake syllables (mixing units); now it's a clean
    additive time offset applied after the proportional calculation.
    Comma ≈ 0.15s, semicolon ≈ 0.25s, colon ≈ 0.20s.
    """
    return (
        sentence.count(",") * 0.15
        + sentence.count(";") * 0.25
        + sentence.count(":") * 0.20
    )


# ══════════════════════════════════════════════════════════════════════════════
# FIX 4 — Abbreviation-aware sentence splitter
# ══════════════════════════════════════════════════════════════════════════════

def _protect_abbreviations(text: str) -> str:
    """
    FIX 4: Replace periods in known abbreviations with a placeholder so the
    sentence splitter does not treat them as sentence endings.
    """
    return _ABBREV_PATTERN.sub(lambda m: m.group().replace(".", "‹dot›"), text)


def _restore_abbreviations(text: str) -> str:
    return text.replace("‹dot›", ".")


def split_into_sentences(text: str) -> list:
    """
    FIX 4 + FIX 7: Abbreviation-aware splitting + minimum clause length guard.
    """
    text = re.sub(r'\s+', ' ', text.strip())
    protected = _protect_abbreviations(text)

    raw_parts = _SENTENCE_END.split(protected)
    sentences = []
    buffer    = ""

    for part in raw_parts:
        part = _restore_abbreviations(part.strip())
        if not part:
            continue
        if len(part.split()) < MIN_SENTENCE_WORDS and buffer:
            buffer = buffer.rstrip() + " " + part
            continue
        if buffer:
            sentences.append(buffer.strip())
        buffer = part

    if buffer.strip():
        sentences.append(_restore_abbreviations(buffer.strip()))

    # FIX 7: split long sentences with minimum clause length guard
    final = []
    for sentence in sentences:
        words = sentence.split()
        if len(words) <= MAX_SENTENCE_WORDS:
            final.append(sentence)
            continue

        clauses     = _CLAUSE_BOUNDARY.split(sentence)
        current     = ""
        for clause in clauses:
            clause = clause.strip()
            if not clause:
                continue
            combined = (current + " " + clause).strip() if current else clause
            # FIX 7: only split if both sides meet minimum clause length
            if (len(combined.split()) > MAX_SENTENCE_WORDS
                    and len(current.split()) >= MIN_CLAUSE_WORDS
                    and len(clause.split())  >= MIN_CLAUSE_WORDS):
                final.append(current.strip())
                current = clause
            else:
                current = combined

        if current.strip():
            # FIX 7: if remaining clause is still too long, hard-split at midpoint
            words_left = current.split()
            if len(words_left) > MAX_SENTENCE_WORDS * 1.5:
                mid = len(words_left) // 2
                final.append(" ".join(words_left[:mid]))
                final.append(" ".join(words_left[mid:]))
            else:
                final.append(current.strip())

    # Merge stragglers
    merged = []
    for s in final:
        if len(s.split()) < MIN_SENTENCE_WORDS and merged:
            merged[-1] = merged[-1].rstrip() + " " + s
        else:
            merged.append(s)

    return [s for s in merged if s.strip()]


# ══════════════════════════════════════════════════════════════════════════════
# FIX 2 + FIX 13 — Chunk boundary simulation matching file2_tts sentence-aware
#                  chunking, now also accounting for heading/section pauses
# ══════════════════════════════════════════════════════════════════════════════

# Heading-intro phrases emitted by file2_tts clean_file1_output()
_HEADING_PREFIXES = ("next:", "now let us look at", "the following section")
_TRANSITION_PREFIXES_CAP = (
    "let us begin", "moving on to", "with this in mind",
    "this concludes", "having understood", "let us proceed",
    "we now explore", "let us now", "an important concept",
    "the next key idea", "another essential point",
    "it is important to understand", "let us consider", "now we turn",
)


def _sentence_gap_sec(sentence: str) -> float:
    """
    FIX 13: Return the extra silence (seconds) that file2_tts inserts around
    this sentence based on its classification as heading / transition / body.
    Body sentences get INTER_CHUNK_SILENCE only at actual chunk boundaries
    (handled separately in estimate_chunk_boundaries). This function returns
    the PER-SENTENCE overhead beyond that baseline.
    """
    low = sentence.lower().strip()
    if any(low.startswith(p) for p in _HEADING_PREFIXES):
        # Heading chunk: 900ms before + 600ms after (file2_tts always emits both)
        return HEADING_PAUSE_BEFORE + HEADING_PAUSE_AFTER
    if any(low.startswith(p) for p in _TRANSITION_PREFIXES_CAP):
        # Transition: between_sections pause (500ms) in place of normal between_chunks
        return SECTION_PAUSE - INTER_CHUNK_SILENCE   # incremental delta
    return 0.0


def estimate_chunk_sentence_indices(sentences: list,
                                     chunk_size: int = TTS_CHUNK_SIZE) -> list:
    """
    FIX 2 + FIX 11: Simulate file2_tts sentence-aware chunking.
    file2_tts accumulates sentences into chunks and starts a new chunk when
    adding the next sentence would exceed chunk_size characters (now 950,
    matching CHUNK_SIZE in file2_tts).
    Returns list of sentence indices where a new chunk BEGINS (index > 0 only).
    Each such boundary contributes INTER_CHUNK_SILENCE to total_gap_time.
    """
    boundary_indices = []
    current_len      = 0

    for i, sentence in enumerate(sentences):
        s_len = len(sentence) + 1  # +1 for space separator
        if current_len + s_len > chunk_size and current_len > 0:
            boundary_indices.append(i)   # new chunk starts at sentence i
            current_len = s_len
        else:
            current_len += s_len

    return boundary_indices


# ══════════════════════════════════════════════════════════════════════════════
# FIX 9 — Minimum caption duration enforcement
# ══════════════════════════════════════════════════════════════════════════════

def _enforce_min_duration(durations: list,
                           floor: float = MIN_CAPTION_DURATION_SEC) -> list:
    """
    FIX 9: Raise any duration below `floor` to `floor`, taking the surplus
    from the longest adjacent caption to keep total duration constant.
    """
    durations = list(durations)
    for i, d in enumerate(durations):
        if d < floor:
            deficit = floor - d
            durations[i] = floor
            # Take surplus from the longest neighbour
            neighbours = [(abs(i - j), j) for j in range(len(durations))
                          if j != i and durations[j] > floor + deficit]
            if neighbours:
                neighbours.sort()
                _, j = neighbours[0]
                durations[j] -= deficit
    return durations


# ══════════════════════════════════════════════════════════════════════════════
# Main caption generator
# ══════════════════════════════════════════════════════════════════════════════

def generate_captions(text: str, duration_sec: float) -> dict:
    """
    STATELESS: Returns sentence-by-sentence captions with duration-accurate timing.

    Algorithm — v2.1.0:
      1.  Split text into display sentences (abbreviation-aware, FIX 4)
      2.  Count syllables per sentence (number expansion, FIX 5)
      3.  Compute per-sentence pause offset in seconds (FIX 1)
      4.  Detect chunk boundaries by sentence index (FIX 2)
      5.  Compute speech_time = duration - actual gap total (FIX 10)
      6.  Assign proportional duration from syllables + pause offsets
      7.  Add gap silence to sentences that start a new chunk (FIX 6)
      8.  Enforce minimum caption display time (FIX 9)
      9.  Build cumulative timestamps

    Args:
        text:         Full extracted text (after file1/file2 processing)
        duration_sec: Actual audio duration from file3_modulator

    Returns:
        { status, captions: [{start, end, text}], total_segments,
          duration_sec, method }
    """
    if not text or not text.strip():
        return {"status": "error", "error": "No text provided."}
    if not duration_sec or duration_sec <= 0:
        return {"status": "error", "error": "Invalid audio duration."}

    try:
        # ── Step 1: Split into display sentences ─────────────────────────────
        sentences = split_into_sentences(text)
        if not sentences:
            return {"status": "error", "error": "No sentences found in text."}

        n = len(sentences)
        logger.info(
            f"Caption generation | sentences={n} | duration={duration_sec}s"
        )

        # ── Step 2: Syllable counts (FIX 5 number expansion) ─────────────────
        syllable_counts = [count_syllables_sentence(s) for s in sentences]
        total_syllables = sum(syllable_counts)
        if total_syllables == 0:
            return {"status": "error", "error": "Could not count syllables."}

        # ── Step 3: Per-sentence punctuation pause offsets (FIX 1) ───────────
        pause_offsets = [_pause_offset_sec(s) for s in sentences]
        total_pause_offset = sum(pause_offsets)

        # ── Step 4: Chunk boundary sentence indices (FIX 2 + FIX 11) ────────────
        chunk_starts     = estimate_chunk_sentence_indices(sentences)
        num_chunks       = len(chunk_starts)
        # FIX 12: INTER_CHUNK_SILENCE now 0.22s (was 0.95s — 4× too large)
        total_chunk_gap  = num_chunks * INTER_CHUNK_SILENCE

        # FIX 13: Add per-sentence heading/transition overhead
        per_sentence_gap = sum(_sentence_gap_sec(s) for s in sentences)
        total_gap_time   = total_chunk_gap + per_sentence_gap

        # FIX 6: gap weights keyed by sentence index (not character offset)
        gap_weights = [0.0] * n
        for idx in chunk_starts:
            # The silence precedes sentence idx — attach it to the preceding sentence
            target = max(0, idx - 1)
            gap_weights[target] += INTER_CHUNK_SILENCE
        # Also distribute heading overhead to the heading sentence itself
        for i, sentence in enumerate(sentences):
            extra = _sentence_gap_sec(sentence)
            if extra > 0:
                gap_weights[i] += extra

        # ── Step 5: speech_time (FIX 10) ─────────────────────────────────────
        # Exact subtraction; floor at 60% to handle edge cases gracefully.
        speech_time = duration_sec - total_gap_time - total_pause_offset
        speech_time = max(speech_time, duration_sec * 0.60)

        # ── Step 6: Per-sentence proportional duration ────────────────────────
        raw_durations = []
        for i in range(n):
            syl_share  = (syllable_counts[i] / total_syllables) * speech_time
            raw_durations.append(syl_share + pause_offsets[i] + gap_weights[i])

        # ── Step 7 already integrated in gap_weights above ───────────────────

        # ── Step 8: Enforce minimum caption duration (FIX 9) ─────────────────
        durations = _enforce_min_duration(raw_durations)

        # Rescale so total exactly equals duration_sec
        total_assigned = sum(durations)
        if total_assigned > 0:
            scale = duration_sec / total_assigned
            durations = [d * scale for d in durations]

        # ── Step 9: Build cumulative timestamps ───────────────────────────────
        captions   = []
        start_time = 0.0

        for i, (sentence, dur) in enumerate(zip(sentences, durations)):
            end_time = min(round(start_time + dur, 3), duration_sec)
            captions.append({
                "start": round(start_time, 3),
                "end":   end_time,
                "text":  sentence,
            })
            start_time = end_time
            if start_time >= duration_sec:
                break

        # Ensure last caption reaches audio end
        if captions:
            captions[-1]["end"] = round(duration_sec, 3)

        # ── FIX 8: Logging with min/max/avg caption duration ─────────────────
        cap_durations = [c["end"] - c["start"] for c in captions]
        avg_dur = round(sum(cap_durations) / len(cap_durations), 2) if cap_durations else 0
        min_dur = round(min(cap_durations), 2) if cap_durations else 0
        max_dur = round(max(cap_durations), 2) if cap_durations else 0
        sps     = round(n / duration_sec, 3)

        logger.info(
            f"Captions OK | segments={len(captions)} | duration={duration_sec}s | "
            f"chunks={num_chunks} | chunk_gap={round(total_chunk_gap,1)}s | "
            f"heading_gap={round(per_sentence_gap,1)}s | total_gap={round(total_gap_time,1)}s | "
            f"sps={sps} | avg_cap={avg_dur}s | min_cap={min_dur}s | "
            f"max_cap={max_dur}s | method=syllable_proportional"
        )

        return {
            "status":          "success",
            "captions":        captions,
            "total_segments":  len(captions),
            "duration_sec":    round(duration_sec, 2),
            "sentences_count": n,
            "method":          "syllable_proportional_sentence",
        }

    except Exception as e:
        logger.error(f"Caption generation failed: {e}", exc_info=True)
        return {"status": "error", "error": str(e)}


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from pathlib import Path

    if len(sys.argv) < 3:
        print("Usage: python file7_captions.py <extracted_txt> <duration_sec>")
        sys.exit(1)

    text_content = Path(sys.argv[1]).read_text(encoding="utf-8")
    duration     = float(sys.argv[2])
    result       = generate_captions(text_content, duration)

    if result["status"] == "success":
        caps = result["captions"]
        durs = [c["end"] - c["start"] for c in caps]
        print(f"\n✓ Generated {result['total_segments']} sentence-level captions")
        print(f"  Method   : {result['method']}")
        print(f"  Duration : {result['duration_sec']}s")
        print(f"  Avg cap  : {round(sum(durs)/len(durs), 2)}s")
        print(f"  Min cap  : {round(min(durs), 2)}s")
        print(f"  Max cap  : {round(max(durs), 2)}s")
        print(f"\nFirst 10 captions:")
        for cap in caps[:10]:
            secs = cap["end"] - cap["start"]
            print(f"  [{cap['start']:6.2f}s – {cap['end']:6.2f}s | {secs:.2f}s]  {cap['text']}")
        if len(caps) > 10:
            print(f"  … and {len(caps) - 10} more")
    else:
        print(f"✗ {result['error']}")
        sys.exit(1)
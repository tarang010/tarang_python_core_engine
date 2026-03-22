"""
Tarang 1.0.0.1 — file7_captions.py
STATELESS: Pure compute engine. No local file reads or writes.
Accepts text string + duration, returns captions as list of dicts.
Express stores captions in MongoDB Document.captions.

Algorithm: Chunk-Gap-Corrected Weighted Proportional Timing
Fixes late-drift caused by pyttsx3 inter-chunk silence gaps.
"""

import re
import logging

logger = logging.getLogger("file7_captions")

# ── Must match file2_tts.py constants exactly ─────────────────────────────────
TTS_CHUNK_SIZE      = 800    # CHUNK_SIZE in file2_tts.py
TTS_RATE_WPM        = 180    # PYTTSX3_RATE in file2_tts.py
INTER_CHUNK_SILENCE = 0.95   # seconds pyttsx3 inserts between chunks (measured)

SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+')

FUNCTION_WORDS = {
    "a","an","the","and","or","but","of","to","in","on","at","by","for",
    "with","as","is","it","its","be","was","are","were","has","had","have",
    "do","does","did","not","no","so","if","up","out","my","we","he","she",
    "they","i","you","me","us","him","her","his","our","their","that","this",
    "these","those","from","into","than","then","when","where","which","who",
    "will","would","can","could","may","might","shall","should","about",
    "after","before",
}

PUNCT_PAUSES = {
    ",":0.35, ";":0.45, ":":0.45, "-":0.25, "—":0.35,
    "(":0.15, ")":0.15, ".":0.65, "!":0.65, "?":0.65,
}

ACRONYM_SYLLABLES = {
    "SQL":3,"API":3,"HTML":4,"CSS":3,"HTTP":4,"HTTPS":5,"URL":3,
    "CPU":3,"GPU":3,"RAM":3,"ROM":3,"SDK":3,"IDE":3,"CLI":3,"GUI":3,
    "JSON":4,"XML":3,"YAML":4,"REST":4,"OOP":3,"AWS":3,"GCP":3,
    "NLP":3,"AI":2,"ML":2,"TTS":3,"OCR":3,"PDF":3,"DNA":3,"RNA":3,
    "ATP":3,"IN":2,"NCC":3,"NDA":3,"IMA":3,"OTA":3,"CDS":3,"EW":2,
    "RF":2,"IF":2,"AM":2,"FM":2,"PM":2,"PWM":3,"PPM":3,"PCM":3,
    "ASK":3,"FSK":3,"PSK":3,"QAM":3,"IoT":3,"SPI":3,"USB":3,"IC":2,
    "PA":2,"PLL":3,"VCO":3,"LNA":3,
}

CHUNK_WORDS = 10   # caption display chunk size


# ── TTS chunk boundary detection ──────────────────────────────────────────────

def estimate_tts_chunks(text: str, chunk_size: int = TTS_CHUNK_SIZE) -> set:
    """
    Simulate file2_tts.py's chunk_text() to find word indices
    where pyttsx3 starts a new synthesis chunk (= silence gap inserted).
    Returns set of word indices where a new TTS chunk begins.
    """
    sentences  = re.split(r'(?<=[.!?])\s+', text.strip())
    boundaries = set()
    word_idx   = 0
    current    = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        sentence_words = len(sentence.split())

        if len(sentence) > chunk_size:
            if current:
                boundaries.add(word_idx)
                current = ""
            sub_parts = re.split(r'(?<=[,;])\s+', sentence)
            sub_chunk = ""
            for part in sub_parts:
                if len(sub_chunk) + len(part) + 1 <= chunk_size:
                    sub_chunk += (" " if sub_chunk else "") + part
                else:
                    if sub_chunk:
                        boundaries.add(word_idx)
                        word_idx += len(sub_chunk.split())
                    sub_chunk = part
            if sub_chunk:
                word_idx += len(sub_chunk.split())
            continue

        if len(current) + len(sentence) + 1 <= chunk_size:
            current  += (" " if current else "") + sentence
            word_idx += sentence_words
        else:
            if current:
                boundaries.add(word_idx)
            current   = sentence
            word_idx += sentence_words

    return boundaries


# ── Word weight helpers ───────────────────────────────────────────────────────

def count_syllables(word: str) -> int:
    word = word.lower().strip(".,!?;:\"'()-—")
    if not word:
        return 1
    vowels, count, prev_vowel = "aeiouy", 0, False
    for ch in word:
        is_v = ch in vowels
        if is_v and not prev_vowel:
            count += 1
        prev_vowel = is_v
    if word.endswith("e") and count > 1:
        count -= 1
    return max(1, count)


def is_numeric_token(word: str) -> bool:
    cleaned = re.sub(r'[.,\-/]', '', word)
    if cleaned.isdigit():
        return True
    if re.search(r'\d', cleaned) and re.search(r'[A-Za-z]', cleaned):
        return True
    return False


def word_weight(word: str) -> float:
    clean = word.strip(".,!?;:\"'()-—")
    upper = clean.upper()
    if upper in ACRONYM_SYLLABLES:
        return float(ACRONYM_SYLLABLES[upper])
    if not clean:
        return 0.2
    if is_numeric_token(word):
        char_count = len(re.sub(r'[.,\-/]', '', clean))
        return max(1.0, char_count * 0.6)
    lower = clean.lower()
    syl   = count_syllables(clean)
    if lower in FUNCTION_WORDS:
        return max(0.4, syl * 0.6)
    return float(syl) + (0.2 if syl > 3 else 0.0)


def punct_pause_after(word: str) -> float:
    if not word:
        return 0.0
    return PUNCT_PAUSES.get(word[-1], 0.0)


def compute_weights(words: list) -> list:
    return [word_weight(w) + punct_pause_after(w) for w in words]


# ── Caption chunk splitter ────────────────────────────────────────────────────

def split_into_caption_chunks(text: str) -> list:
    """
    Split text into display-sized caption segments.
    Respects sentence boundaries, merges short fragments,
    breaks long sentences into CHUNK_WORDS-sized pieces.
    """
    text = re.sub(r'\s+', ' ', text.strip())
    raw  = SENTENCE_SPLIT.split(text)

    sentences = []
    buffer    = ""
    for part in raw:
        part = part.strip()
        if not part:
            continue
        if len(part.split()) < 4 and buffer:
            buffer += " " + part
        else:
            if buffer:
                sentences.append(buffer.strip())
            buffer = part
    if buffer.strip():
        sentences.append(buffer.strip())

    final = []
    for sentence in sentences:
        words = sentence.split()
        if len(words) <= CHUNK_WORDS:
            final.append(sentence)
        else:
            for i in range(0, len(words), CHUNK_WORDS):
                chunk = " ".join(words[i:i + CHUNK_WORDS])
                if chunk:
                    final.append(chunk)
    return final


# ── Main caption generator ────────────────────────────────────────────────────

def generate_captions(text: str, duration_sec: float) -> dict:
    """
    STATELESS: Pure compute — accepts text string + duration, returns caption list.
    No file reads or writes. Express stores result in MongoDB Document.captions.

    Algorithm — Chunk-Gap-Corrected Weighted Proportional Timing:
      1. Compute per-word weights (syllables, acronyms, numerics, pauses)
      2. Detect TTS chunk boundaries (where pyttsx3 inserts silence gaps)
      3. Insert gap weights at those boundaries
      4. Normalise total weight to duration_sec
      5. Map each caption display chunk to its start/end time

    Args:
        text:         Full extracted text (before acronym expansion)
        duration_sec: Real audio duration from modulate step

    Returns:
        {
            status, captions, total_segments,
            duration_sec, wps, method
        }
    """
    if not text or not text.strip():
        return {"status": "error", "error": "No text provided."}
    if not duration_sec or duration_sec <= 0:
        return {"status": "error", "error": "Invalid audio duration."}

    try:
        all_words   = text.split()
        total_words = len(all_words)
        all_weights = compute_weights(all_words)

        # Detect TTS chunk boundaries
        chunk_boundaries = estimate_tts_chunks(text)
        num_gaps         = len(chunk_boundaries)

        # Compute speech weight (excluding gaps)
        speech_weight = sum(all_weights)
        if speech_weight == 0:
            return {"status": "error", "error": "Text has no content."}

        # Allocate time: speech_time = real duration minus accumulated gap silence
        total_gap_time = num_gaps * INTER_CHUNK_SILENCE
        speech_time    = max(duration_sec - total_gap_time, duration_sec * 0.85)

        # Convert gap time to weight units
        weight_per_sec = speech_weight / speech_time
        gap_weight     = INTER_CHUNK_SILENCE * weight_per_sec

        # Build cumulative weight array with gap weights injected at boundaries
        cumulative = [0.0]
        running    = 0.0
        for i, w in enumerate(all_weights):
            if i in chunk_boundaries:
                running += gap_weight   # silence gap before this word
            running += w
            cumulative.append(running)

        total_weight = running

        # Map caption chunks to timestamps
        caption_chunks = split_into_caption_chunks(text)
        captions       = []
        word_cursor    = 0

        for chunk in caption_chunks:
            n          = len(chunk.split())
            end_cursor = min(word_cursor + n, total_words)
            start_time = (cumulative[word_cursor] / total_weight) * duration_sec
            end_time   = (cumulative[end_cursor]  / total_weight) * duration_sec
            captions.append({
                "start": round(start_time, 3),
                "end":   round(end_time, 3),
                "text":  chunk,
            })
            word_cursor = end_cursor
            if word_cursor >= total_words:
                break

        actual_wps = round(total_words / duration_sec, 2)

        logger.info(
            f"Captions generated | segments={len(captions)} | duration={duration_sec}s | "
            f"gaps={num_gaps} | gap_time={round(total_gap_time,1)}s | "
            f"wps={actual_wps} | method=chunk_gap_corrected"
        )

        return {
            "status":         "success",
            "captions":       captions,       # list of {start, end, text} dicts
            "total_segments": len(captions),
            "duration_sec":   round(duration_sec, 2),
            "wps":            actual_wps,
            "method":         "chunk_gap_corrected",
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
        print(f"\n✓ Generated {result['total_segments']} segments")
        print(f"  Method : {result['method']}")
        print(f"  WPS    : {result['wps']}")
        print(f"\nFirst 10 captions:")
        for cap in result["captions"][:10]:
            print(f"  [{cap['start']:6.2f}s – {cap['end']:6.2f}s]  {cap['text']}")
        if len(result["captions"]) > 10:
            print(f"  … and {len(result['captions']) - 10} more")
    else:
        print(f"✗ {result['error']}")
        sys.exit(1)

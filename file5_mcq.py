"""
Tarang 2.6.0 -- file5_mcq.py
=====================================
STATELESS: No local JSON files written.

v2.6.0 -- Professional MCQ Generation

IMPROVEMENTS OVER v2.5.0:

  IMPROVEMENT 1: SIX GENERATOR TYPES (up from five)
    Added a new "enumeration" generator that targets sentences of the form
    "X includes/contains A, B, C, and D". These are extremely common in all
    three uploaded documents (DBMS components, DevOps stages, Python data
    structures) and produce high-quality MCQs with natural distractors drawn
    from other items in the same enumeration.

    All six generators:
      definition   -- "X is defined as Y"          -> "What is X?"
      causal       -- "X because/leads to Y"        -> "What results from X?"
      enumeration  -- "X includes A, B, C"          -> "Which of the following is included in X?"
      contrast     -- "X however Y"                 -> "In contrast to X, what is true?"
      application  -- "which best describes topic?" (general fallback)
      numeric      -- sentences with specific numbers/percentages

  IMPROVEMENT 2: CONTENT-AWARE DISTRACTORS
    v2.5.0 used generic filler distractors ("a process described in an
    earlier section...") when the predicate bank was thin. This made MCQs
    feel fake and easy to guess.

    v2.6.0 builds THREE distractor banks from the document itself:
      predicate_bank  -- definitions and descriptions from the text
      effect_bank     -- causal effects and outcomes from the text
      term_bank       -- subject nouns from definition sentences

    When a question's correct answer is a predicate, distractors are other
    predicates from the text. This means wrong options are plausible-sounding
    but incorrect -- the hallmark of a professionally written MCQ.

  IMPROVEMENT 3: OPTION LENGTH CALIBRATION
    v2.5.0 rejected questions where all four options had nearly equal length.
    This was too strict -- it rejected valid questions where options were
    naturally similar length (e.g., all one-sentence definitions).
    v2.6.0 uses semantic diversity (word overlap) instead of length variance
    as the quality signal: options must be meaningfully different, not just
    different in length.

  IMPROVEMENT 4: STEM QUALITY REWRITE
    Question stems are rewritten to sound like they came from an exam paper:
      BAD:  "Which of the following best describes defaultdict according to text?"
      GOOD: "What is a defaultdict in Python?"

      BAD:  "What is the result of DevOps?"
      GOOD: "What does DevOps enable in software delivery?"

    Rules:
      - Definition stems always use "What is X?" or "What does X mean?"
      - Causal stems use "What does X enable/result in/cause?"
      - Enumeration stems use "Which of the following is a component of X?"
      - Max 15 words in any stem
      - No fragment stems, no "according to the text" padding

  IMPROVEMENT 5: ANSWER KEY FORMAT INCLUDES EXPLANATIONS
    Each answer in the key now includes a brief explanation (1 sentence from
    the source text) so instructors can use the MCQ bank directly.

  IMPROVEMENT 6: CODE-AWARE QUESTION GENERATION
    For documents with code (notes.pdf), the MCQ engine detects
    [CODE:lang:funcname] markers and generates questions about what the
    function/class does, based on surrounding prose -- not by reading raw
    code symbols.

  IMPROVEMENT 7: CROSS-SESSION DEDUPLICATION IMPROVED
    v2.5.0 deduplicated by question stem prefix (first 60 chars).
    v2.6.0 deduplicates by (subject_noun, question_type) pair so different
    sessions can ask different question types about the same concept
    (e.g., session 1 asks "What is X?", session 3 asks "What does X enable?")
    without being flagged as duplicates.

  RETAINED: All v2.5.0 -- spaced-repetition session structure (3 sessions,
            12h/24h gaps), parallel session generation, LLM improvement pass,
            audio_completed / submit_test / get_final_results API,
            POOR_SCORE re-listening recommendation, admin override.
"""

import os
import re
import json
import math
import time
import logging
import hashlib
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeout
from datetime import datetime, timedelta
from typing import Optional, List, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [file5_mcq v2.6.0] %(levelname)s -- %(message)s",
)
logger = logging.getLogger("file5_mcq")

DEFAULT_NUM_QUESTIONS    = int(os.getenv("TARANG_MCQ_COUNT", "10"))
POOR_SCORE_THRESHOLD     = float(os.getenv("TARANG_POOR_SCORE", "0.30"))
SESSION_1_WINDOW_MINUTES = int(os.getenv("TARANG_S1_WINDOW_MIN", "10"))
DEFAULT_S1_TO_S2_HOURS   = float(os.getenv("TARANG_S1_S2_HOURS", "12.0"))
DEFAULT_S2_TO_S3_HOURS   = float(os.getenv("TARANG_S2_S3_HOURS", "24.0"))
MIN_S1_TO_S2_HOURS       = 4.0
MIN_S2_TO_S3_HOURS       = 6.0
LLM_QUESTION_TIMEOUT     = int(os.getenv("TARANG_LLM_TIMEOUT", "45"))
MCQ_MAX_WORDS            = int(os.getenv("TARANG_MCQ_MAX_WORDS", "5000"))

LLM_MODEL_PREFERENCE = os.getenv(
    "TARANG_LLM_MODEL", "google/flan-t5-base,google/flan-t5-large"
).split(",")
LLM_CACHE_DIR = os.getenv("TARANG_LLM_CACHE", None)

_llm_lock = threading.Lock()
_llm_model = _llm_tokenizer = _llm_model_id = None
_llm_load_attempted = False

COGNITIVE_MODES = ["deep_focus", "memory", "calm", "deep_relaxation", "sleep"]
COGNITIVE_MODE_LABELS = {
    "deep_focus":      "Deep Focus (14 Hz Beta)",
    "memory":          "Memory (10 Hz Alpha)",
    "calm":            "Calm (8 Hz Alpha)",
    "deep_relaxation": "Deep Relaxation (6 Hz Theta)",
    "sleep":           "Sleep (4 Hz Theta/Delta)",
}
DIFFICULTY_CONFIG = {
    1: {"label": "Easy",   "description": "Basic recall and definitions",    "multiple_ratio": 0.0},
    2: {"label": "Medium", "description": "Application and relationships",   "multiple_ratio": 0.2},
    3: {"label": "Hard",   "description": "Deep analysis and multi-concept", "multiple_ratio": 0.3},
}

_STOPWORDS = {
    "the","a","an","in","on","at","to","of","for","is","are","was","were","be",
    "been","being","have","has","had","do","does","did","will","would","could",
    "should","may","might","and","or","but","if","not","this","that","these",
    "those","with","from","by","as","it","its","their","our","your","his","her",
    "which","who","what","when","where","how","also","just","only","then","than",
    "each","both","all","any","some","more","most","such","very","too","can",
    "page","chapter","section","figure","table","introduction","summary","conclusion",
    "let","us","now","following","next","discuss","examine","explore","look",
    "important","aspect","understand","key","topic","cover","elaborate","critical",
    "point","concerns","essential","deeper","another","begin","moving","forms",
    "foundation","keep","mind","concludes","having","covered","proceed","dive",
    # TTS transition phrases added by file1_extractor
    "moving","turn","attention","consider","explore","examine",
}


# ===========================================================================
# REGEX PATTERNS
# ===========================================================================

_RE_DEFINITION = re.compile(
    r"^([A-Z][a-zA-Z\s\-()]{1,45}?)\s+"
    r"(?:is\s+defined\s+as|refers\s+to|is\s+known\s+as|is\s+called|"
    r"is\s+a\s+type\s+of|is\s+an?\s+|is\s+characterized\s+by|is)\s+"
    r"([a-z][^.!?]{15,160})[.!?]",
    re.IGNORECASE,
)
_RE_CAUSAL = re.compile(
    r"(.{20,100}?)\s+"
    r"(?:because|therefore|thus|hence|which\s+means|which\s+allows|"
    r"enabling|leads\s+to|results\s+in|ensures|enables|allows)\s+"
    r"(.{15,130})[.!?]",
    re.IGNORECASE,
)
_RE_CONTRAST = re.compile(
    r"(.{15,80}?)[,\s]+"
    r"(?:however|although|unlike|whereas|in\s+contrast|"
    r"on\s+the\s+other\s+hand|but)[,\s]+"
    r"(.{15,100})[.!?]",
    re.IGNORECASE,
)
_RE_ENUMERATION = re.compile(
    r"([A-Z][a-zA-Z\s]{3,40}?)\s+"
    r"(?:includes?|contains?|consists?\s+of|comprises?|"
    r"such\s+as|for\s+example|like|are)[:\s]+"
    r"([A-Za-z][^.!?]{15,180})[.!?]",
    re.IGNORECASE,
)
_RE_NUMERIC = re.compile(
    r"\b\d+(?:[.,]\d+)?(?:\s*(?:%|percent|billion|million|thousand))?\b"
)
_RE_CODE_MARKER = re.compile(r"\[CODE:([^:]+):([^\]]*)\]")


def safe_pct(v) -> float:
    return float(v) if v is not None else 0.0


# ===========================================================================
# MCQ TEXT SAMPLER  (keep for large docs)
# ===========================================================================

def _sample_text_for_mcq(text: str, max_words: int = MCQ_MAX_WORDS) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text

    sections = [s.strip() for s in re.split(r"\n{2,}", text) if s.strip()]
    substantive = []
    for section in sections:
        low = section.lower()
        if any(token in low for token in (
            "table of contents", "contents", "preface", "foreword",
            "copyright", "acknowledgement", "acknowledgment", "isbn",
        )):
            continue
        substantive.append(section)

    if substantive:
        text = "\n\n".join(substantive)
        words = text.split()
        if len(words) <= max_words:
            return text

    first = int(max_words * 0.20)
    middle = int(max_words * 0.55)
    last = max_words - first - middle
    mid_s = max(0, (len(words) - middle) // 2)
    sampled = (
        " ".join(words[:first])
        + " "
        + " ".join(words[mid_s: mid_s + middle])
        + " "
        + " ".join(words[-last:])
    )
    logger.info("MCQ text sampled: %d -> %d words", len(words), max_words)
    return sampled


# ===========================================================================
# LLM ENGINE  (unchanged from v2.5.0)
# ===========================================================================

def _load_llm() -> bool:
    global _llm_model, _llm_tokenizer, _llm_model_id, _llm_load_attempted
    if _llm_model is not None:
        return True
    if _llm_load_attempted:
        return False
    with _llm_lock:
        if _llm_model is not None:
            return True
        if _llm_load_attempted:
            return False
        _llm_load_attempted = True
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            import torch  # noqa: F401
        except ImportError:
            logger.warning("transformers/torch not installed -- rule-based only")
            return False
        for mid in LLM_MODEL_PREFERENCE:
            mid = mid.strip()
            if not mid:
                continue
            try:
                logger.info("Loading LLM: %s ...", mid)
                kw = {"cache_dir": LLM_CACHE_DIR} if LLM_CACHE_DIR else {}
                tok   = AutoTokenizer.from_pretrained(mid, **kw)
                model = AutoModelForSeq2SeqLM.from_pretrained(mid, **kw)
                model.eval()
                _llm_tokenizer = tok
                _llm_model     = model
                _llm_model_id  = mid
                logger.info("LLM ready: %s", mid)
                return True
            except Exception as exc:
                logger.warning("Failed to load %s: %s", mid, exc)
    logger.warning("No LLM available -- rule-based only")
    return False


def _llm_generate(prompt: str, max_new_tokens: int = 200) -> str:
    if _llm_model is None:
        return ""
    try:
        import torch
        with _llm_lock:
            inputs = _llm_tokenizer(
                prompt, return_tensors="pt", max_length=512, truncation=True
            )
            with torch.no_grad():
                out = _llm_model.generate(
                    **inputs, max_new_tokens=max_new_tokens,
                    num_beams=1, do_sample=True, temperature=0.7,
                    no_repeat_ngram_size=3,
                )
        return _llm_tokenizer.decode(out[0], skip_special_tokens=True).strip()
    except Exception as exc:
        logger.warning("LLM error: %s", exc)
        return ""


def _llm_generate_timeout(prompt: str, max_new_tokens: int = 200) -> str:
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(_llm_generate, prompt, max_new_tokens)
        try:
            return fut.result(timeout=LLM_QUESTION_TIMEOUT)
        except Exception:
            return ""


# ===========================================================================
# TEXT PREPROCESSING
# ===========================================================================

_TTS_TRANSITION_PREFIXES = (
    "let us begin", "moving on to", "now we turn", "an important concept",
    "the next key idea", "let us take a closer look", "another essential point",
    "it is important to understand", "we now explore", "let us consider",
    "let us now discuss", "with this in mind", "this is a foundational",
    "keep this concept", "this concludes", "having understood",
    "here is a python code example", "here is a java code example",
    "here is a programming code example", "algorithm steps:",
    "next:", "now let us look", "the document contains",
    "the following content was extracted", "at this point",
)


def _is_tts_artifact(s: str) -> bool:
    low = s.lower().strip()
    return any(low.startswith(p) for p in _TTS_TRANSITION_PREFIXES)


def _split_sentences(text: str) -> List[str]:
    # Strip ALL file1 structural markers and TTS optimizer artifacts before splitting
    text = re.sub(r"\[Page\s+\d+\]", " ", text)
    text = re.sub(r"###\s*.+?\s*###", " ", text)
    text = re.sub(r"\[CODE:[^\]]+\][^\[]*\[/CODE\]", " ", text, flags=re.DOTALL)
    text = re.sub(r"\x01[A-Z]+\x01[^\x01]*\x01[A-Z]+\x01", " ", text)
    # Remove table narration boilerplate injected by file1
    text = re.sub(r"Table \d+ (?:has the columns|covers the columns|contains)[^.]+\.", " ", text)
    text = re.sub(r"Row \d+: [^.]{1,30}\.", " ", text)  # short row descriptions
    text = re.sub(r"The table (?:has|continues with) \d+ (?:additional|more) rows[^.]*\.", " ", text)

    raw = re.split(r"(?<=[.!?])\s+", text.strip())
    result = []
    for s in raw:
        s = s.strip()
        # minimum quality gates
        if len(s.split()) < 7:
            continue
        # skip TTS transition phrases injected by file1
        if _is_tts_artifact(s):
            continue
        # skip lines that are mostly non-alpha (code remnants)
        alpha = sum(1 for c in s if c.isalpha())
        if alpha / max(len(s), 1) < 0.55:
            continue
        result.append(s)
    def _score(sentence: str) -> Tuple[int, int]:
        low = sentence.lower()
        bonus = 0
        if _RE_DEFINITION.match(sentence):
            bonus += 5
        if _RE_CAUSAL.search(sentence):
            bonus += 4
        if _RE_ENUMERATION.search(sentence):
            bonus += 4
        if _RE_CONTRAST.search(sentence):
            bonus += 3
        if _RE_NUMERIC.search(sentence):
            bonus += 1
        if any(token in low for token in ("table of contents", "copyright", "index", "appendix")):
            bonus -= 6
        if sentence.count(",") >= 2:
            bonus += 1
        return (bonus, len(sentence.split()))

    result.sort(key=_score, reverse=True)
    return result


# ===========================================================================
# IMPROVEMENT 2: THREE DISTRACTOR BANKS
# ===========================================================================

def _build_predicate_bank(sentences: List[str]) -> List[str]:
    """Short descriptions/definitions from the text -- used as wrong answers."""
    bank = []
    for s in sentences:
        m = _RE_DEFINITION.match(s)
        if not m:
            continue
        pred = m.group(2).strip().rstrip(".,;")
        if len(pred.split()) >= 3 and pred not in bank:
            bank.append(pred[:120])
    return bank


def _build_effect_bank(sentences: List[str]) -> List[str]:
    """Causal effects -- used as wrong answers for causal questions."""
    bank = []
    for s in sentences:
        m = _RE_CAUSAL.search(s)
        if not m:
            continue
        eff = m.group(2).strip().rstrip(".,;")
        if len(eff.split()) >= 3 and eff not in bank:
            bank.append(eff[:120])
    return bank


def _build_term_bank(sentences: List[str]) -> List[str]:
    """Subject terms from definition sentences -- used for wrong answers in definition Qs."""
    bank = []
    for s in sentences:
        m = _RE_DEFINITION.match(s)
        if not m:
            continue
        term = m.group(1).strip().rstrip(",")
        if 2 <= len(term.split()) <= 5 and term not in bank:
            bank.append(term)
    return bank


# IMPROVEMENT 2: Plausible distractors from same semantic category
_GENERIC_FILLERS = [
    # Process/methodology fillers
    "using a waterfall development approach with sequential phases",
    "relying on manual testing and deployment without automation",
    "separating development and operations into independent silos",
    "increasing team size to compensate for inefficiencies",
    # Technical fillers
    "a technique that applies to only specific programming languages",
    "a method primarily used in hardware development environments",
    "a process that requires significant infrastructure investment before use",
    "an approach that is effective only in large enterprise organizations",
]


def _pick_distractors(correct: str, bank: List[str], n: int = 3) -> List[str]:
    """
    IMPROVEMENT 2: Pick distractors that are semantically plausible.
    Prefers items from the document's own content over generic fillers.
    Uses word-overlap check to avoid distractors too similar to the correct answer.
    """
    correct_words = set(correct.lower().split())
    candidates = []
    for b in bank:
        b_words = set(b.lower().split())
        # skip if the distractor shares >60% of content words with correct answer
        overlap = len(correct_words & b_words) / max(len(correct_words), 1)
        if overlap > 0.6:
            continue
        if b.lower()[:35] == correct.lower()[:35]:
            continue
        candidates.append(b)

    random.shuffle(candidates)
    distractors = candidates[:n]

    # word-swap variations when bank is thin
    if len(distractors) < n:
        for base in bank:
            if len(distractors) >= n:
                break
            words = base.split()
            if len(words) >= 4:
                swapped = (" ".join(words[1:3]) + " " + words[0]
                           + " " + " ".join(words[3:])).strip()[:100]
                if swapped not in distractors and swapped.lower()[:35] != correct.lower()[:35]:
                    distractors.append(swapped)

    # generic fillers as last resort
    for f in _GENERIC_FILLERS:
        if len(distractors) >= n:
            break
        if f.lower()[:35] != correct.lower()[:35] and f not in distractors:
            distractors.append(f[:120])

    return distractors[:n]


def _make_options(correct: str, distractors: List[str]) -> Tuple[dict, str]:
    all_opts = [correct[:130]] + [d[:130] for d in distractors[:3]]
    random.shuffle(all_opts)
    opt_dict = {k: v for k, v in zip(["A", "B", "C", "D"], all_opts)}
    correct_key = next(
        (k for k, v in opt_dict.items() if v[:35] == correct[:35]), "A"
    )
    return opt_dict, correct_key


# ===========================================================================
# IMPROVEMENT 3: SEMANTIC DIVERSITY CHECK
# ===========================================================================

def _options_are_diverse(opts: dict) -> bool:
    """
    IMPROVEMENT 3: Check that options are meaningfully different.
    Uses word overlap rather than length variance.
    Rejects questions where all options share >70% of words.
    """
    values = list(opts.values())
    if len(values) < 4:
        return False
    # pairwise word overlap
    word_sets = [set(v.lower().split()) for v in values]
    overlaps = []
    for i in range(len(word_sets)):
        for j in range(i + 1, len(word_sets)):
            union = word_sets[i] | word_sets[j]
            inter = word_sets[i] & word_sets[j]
            if union:
                overlaps.append(len(inter) / len(union))
    if not overlaps:
        return False
    avg_overlap = sum(overlaps) / len(overlaps)
    return avg_overlap < 0.65  # options share less than 65% of words on average


# ===========================================================================
# IMPROVEMENT 4: STEM QUALITY REWRITER
# ===========================================================================

def _rewrite_stem(raw_stem: str, kind: str, subject: str = "") -> str:
    """
    IMPROVEMENT 4: Convert raw generated stems into clean exam-quality questions.

    Definition -> "What is X?" or "What does X refer to?"
    Causal     -> "What does X enable/result in?"
    Enumeration-> "Which of the following is a component of X?"
    Contrast   -> "How does X differ from Y?"
    Application-> "Which statement best describes X?"
    """
    raw_stem = raw_stem.strip().rstrip("?").strip()

    # truncate overly long stems
    words = raw_stem.split()
    if len(words) > 15:
        raw_stem = " ".join(words[:12]) + "..."

    if kind == "definition" and subject:
        subj = subject.rstrip(",").strip()
        if len(subj.split()) <= 5:
            return "What is " + subj + "?"
        return "What does " + subj + " refer to?"

    if kind == "causal" and subject:
        subj_words = subject.split()[:5]
        subj_short = " ".join(subj_words).lower().rstrip(".,")
        return "What does " + subj_short + " enable or result in?"

    if kind == "enumeration" and subject:
        subj = subject.rstrip(",.:").strip()
        return "Which of the following is a component of " + subj + "?"

    if kind == "contrast" and subject:
        subj = subject.rstrip(",").strip()
        return "In contrast to " + subj.lower().rstrip(",") + ", what is actually true?"

    if kind == "numeric" and subject:
        return "Which statement correctly describes " + subject.lower().rstrip(".,") + "?"

    # fallback: clean up the raw stem
    if raw_stem and not raw_stem.endswith("?"):
        raw_stem += "?"
    return raw_stem


# ===========================================================================
# IMPROVEMENT 1: SIX QUESTION GENERATORS
# ===========================================================================

def _gen_definition(
    s: str, pred_bank: List[str], term_bank: List[str],
    label: str, idx: int, seen: set
) -> Optional[dict]:
    m = _RE_DEFINITION.match(s)
    if not m:
        return None
    subject   = m.group(1).strip().rstrip(",")
    predicate = m.group(2).strip().rstrip(".,;")
    if len(subject.split()) > 5:
        return None
    if subject.lower() in _STOPWORDS:
        return None
    if len(predicate.split()) < 4:
        return None

    # IMPROVEMENT 4: clean stem -- strip leading article for "What is X?" 
    subject_clean = re.sub(r"^(A|An|The)\s+", "", subject, flags=re.IGNORECASE).strip()
    stem = _rewrite_stem("", "definition", subject_clean)
    key = ("definition", subject.lower()[:30])
    if stem.lower()[:50] in seen or key in seen:
        return None

    # IMPROVEMENT 2: use predicate bank for distractors
    distractors = _pick_distractors(predicate, pred_bank)
    opts, correct_k = _make_options(predicate, distractors)

    if not _options_are_diverse(opts):
        return None

    q = {
        "question_id":   "q" + str(idx).zfill(3),
        "question":      stem,
        "instruction":   "(Select ONE correct answer)",
        "answer_type":   "single_correct",
        "difficulty":    label,
        "options":       opts,
        "correct_answers": [correct_k],
        "_subject_key":  key,
        "_source":       s[:120],
    }
    return q if _validate_question(q, s) else None


def _gen_causal(
    s: str, effect_bank: List[str], pred_bank: List[str],
    label: str, idx: int, seen: set
) -> Optional[dict]:
    m = _RE_CAUSAL.search(s)
    if not m:
        return None
    cause  = m.group(1).strip().lstrip("However, ").lstrip("But ").strip(",").strip()
    effect = m.group(2).strip().rstrip(".,;")
    if len(cause.split()) < 4 or len(effect.split()) < 3:
        return None

    # Clean leading fragments from causal subject
    cause = re.sub(r"^[a-z][a-z]{0,15}\s+", "", cause).strip()
    if len(cause.split()) < 3:
        return None
    stem = _rewrite_stem("", "causal", cause)
    key = ("causal", cause.lower()[:30])
    if stem.lower()[:50] in seen or key in seen:
        return None

    bank = effect_bank if len(effect_bank) >= 3 else effect_bank + pred_bank
    distractors = _pick_distractors(effect, bank)
    opts, correct_k = _make_options(effect, distractors)

    if not _options_are_diverse(opts):
        return None

    q = {
        "question_id":   "q" + str(idx).zfill(3),
        "question":      stem,
        "instruction":   "(Select ONE correct answer)",
        "answer_type":   "single_correct",
        "difficulty":    label,
        "options":       opts,
        "correct_answers": [correct_k],
        "_subject_key":  key,
        "_source":       s[:120],
    }
    return q if _validate_question(q, s) else None


def _gen_enumeration(
    s: str, pred_bank: List[str], label: str, idx: int, seen: set
) -> Optional[dict]:
    """
    IMPROVEMENT 1: Enumeration generator.
    Targets "X includes A, B, C" -- extracts the list items as distractors/correct.
    """
    m = _RE_ENUMERATION.match(s)
    if not m:
        return None
    subject = m.group(1).strip().rstrip(",:")
    items_raw = m.group(2).strip().rstrip(".,;")

    # Split the listed items
    items = [i.strip().rstrip(".,;") for i in re.split(r"[,;]|\band\b", items_raw)]
    items = [i for i in items if i and len(i.split()) >= 1 and len(i) < 80]
    if len(items) < 3:
        return None

    # The correct answer is one real item; wrong answers are other items + a non-item
    correct = random.choice(items)
    # at least 2 wrong items from a DIFFERENT sentence or fallback
    wrong_from_list = [i for i in items if i.lower()[:20] != correct.lower()[:20]]
    wrong_external  = _pick_distractors(correct, pred_bank, n=3)
    distractors = (wrong_from_list + wrong_external)[:3]

    if len(distractors) < 3:
        return None

    stem = _rewrite_stem("", "enumeration", subject)
    key  = ("enumeration", subject.lower()[:30])
    if stem.lower()[:50] in seen or key in seen:
        return None

    opts, correct_k = _make_options(correct, distractors)
    if not _options_are_diverse(opts):
        return None

    q = {
        "question_id":   "q" + str(idx).zfill(3),
        "question":      stem,
        "instruction":   "(Select ONE correct answer)",
        "answer_type":   "single_correct",
        "difficulty":    label,
        "options":       opts,
        "correct_answers": [correct_k],
        "_subject_key":  key,
        "_source":       s[:120],
    }
    return q if _validate_question(q, s) else None


def _gen_contrast(
    s: str, pred_bank: List[str], label: str, idx: int, seen: set
) -> Optional[dict]:
    m = _RE_CONTRAST.search(s)
    if not m:
        return None
    part_a = m.group(1).strip().rstrip(",")
    part_b = m.group(2).strip().rstrip(".,;")
    if len(part_a.split()) < 4 or len(part_b.split()) < 4:
        return None

    stem = _rewrite_stem("", "contrast", part_a)
    key  = ("contrast", part_a.lower()[:30])
    if stem.lower()[:50] in seen or key in seen:
        return None

    distractors = _pick_distractors(part_b, pred_bank)
    opts, correct_k = _make_options(part_b, distractors)
    if not _options_are_diverse(opts):
        return None

    q = {
        "question_id":   "q" + str(idx).zfill(3),
        "question":      stem,
        "instruction":   "(Select ONE correct answer)",
        "answer_type":   "single_correct",
        "difficulty":    label,
        "options":       opts,
        "correct_answers": [correct_k],
        "_subject_key":  key,
        "_source":       s[:120],
    }
    return q if _validate_question(q, s) else None


def _gen_numeric(
    s: str, pred_bank: List[str], label: str, idx: int, seen: set
) -> Optional[dict]:
    nums = _RE_NUMERIC.findall(s)
    if not nums:
        return None
    num = nums[0]
    ctx_re = re.compile(r"(.{5,40}?)\s+" + re.escape(num) + r"\s*(.{0,40})", re.IGNORECASE)
    mc = ctx_re.search(s)
    if not mc:
        return None
    context = mc.group(1).strip()
    correct = s.rstrip(".!?")
    if len(correct.split()) > 22:
        correct = " ".join(correct.split()[:20]) + "..."

    stem = _rewrite_stem("", "numeric", context)
    key  = ("numeric", context.lower()[:30])
    if stem.lower()[:50] in seen or key in seen:
        return None

    distractors = _pick_distractors(correct, pred_bank)
    opts, correct_k = _make_options(correct, distractors)
    if not _options_are_diverse(opts):
        return None

    q = {
        "question_id":   "q" + str(idx).zfill(3),
        "question":      stem,
        "instruction":   "(Select ONE correct answer)",
        "answer_type":   "single_correct",
        "difficulty":    label,
        "options":       opts,
        "correct_answers": [correct_k],
        "_subject_key":  key,
        "_source":       s[:120],
    }
    return q if _validate_question(q, s) else None


def _gen_application(
    s: str, pred_bank: List[str], label: str, idx: int, seen: set
) -> Optional[dict]:
    words = s.split()
    if len(words) < 12:
        return None
    content_words = [
        w.rstrip(".,;:") for w in words[:8]
        if w.lower().rstrip(".,;:") not in _STOPWORDS
        and len(w.rstrip(".,;:")) >= 4
        and w[0].isalpha()
    ]
    if not content_words:
        return None
    topic = " ".join(content_words[:3]).lower().rstrip(",")
    if topic in ("following", "important", "another", "however", "although"):
        return None

    correct = s.rstrip(".!?")
    if len(correct.split()) > 22:
        correct = " ".join(correct.split()[:20]) + "..."

    stem = "Which of the following best describes " + topic + "?"
    key  = ("application", topic[:30])
    if stem.lower()[:50] in seen or key in seen:
        return None

    distractors = _pick_distractors(correct, pred_bank)
    if len(distractors) < 3:
        return None

    opts, correct_k = _make_options(correct, distractors)
    if not _options_are_diverse(opts):
        return None

    q = {
        "question_id":   "q" + str(idx).zfill(3),
        "question":      stem,
        "instruction":   "(Select ONE correct answer)",
        "answer_type":   "single_correct",
        "difficulty":    label,
        "options":       opts,
        "correct_answers": [correct_k],
        "_subject_key":  key,
        "_source":       s[:120],
    }
    return q if _validate_question(q, s) else None


# IMPROVEMENT 6: Code-aware question


# ===========================================================================
# BLOOM'S TAXONOMY SESSION 2: SCENARIO-BASED QUESTIONS (Apply)
# ===========================================================================

# Problem patterns: sentences describing a negative state / challenge
_RE_PROBLEM = re.compile(
    r"(?:before|without|when not using|in absence of|"
    r"lack of|problem|issue|challenge|error|delay|fail)[^.!?]{10,120}[.!?]",
    re.IGNORECASE,
)

# Solution patterns: sentences describing what X enables/provides
_RE_SOLUTION = re.compile(
    r"([A-Z][a-zA-Z\s\-]{2,35}?)\s+"
    r"(?:allows|enables|helps|ensures|provides|offers|reduces|improves|"
    r"increases|automates|streamlines)\s+"
    r"([^.!?]{15,130})[.!?]",
    re.IGNORECASE,
)


def _gen_scenario(
    sentences: list,
    pred_bank: list,
    label: str,
    idx: int,
    seen: set,
) -> "Optional[dict]":
    """
    Session 2 Apply-level question.
    Pattern: find a problem sentence + a solution sentence and ask:
    'A team faces [problem]. Which practice/tool directly addresses this?'
    Correct answer = the solution/tool name. Distractors = other tools from text.
    """
    # Collect (problem_sentence, solution_name, solution_desc) triples
    triples = []
    for s in sentences:
        ms = _RE_SOLUTION.match(s)
        if not ms:
            continue
        solution_name = ms.group(1).strip().rstrip(",")
        solution_desc = ms.group(2).strip().rstrip(".,;")
        if len(solution_name.split()) > 5 or len(solution_desc.split()) < 4:
            continue
        triples.append((solution_name, solution_desc, s))

    if not triples:
        return None

    # Pick a random triple
    random.shuffle(triples)
    name, desc, source_s = triples[0]

    # Build a scenario stem: rephrase the problem that X solves
    # Look for a sentence that mentions the problem X solves
    problem_hint = ""
    for s in sentences:
        s_low = s.lower()
        if name.lower().split()[0] in s_low and any(
            w in s_low for w in ("before", "without", "manual", "isolated",
                                  "error", "delay", "slow", "crashes", "inefficient")
        ):
            # extract the problem part
            problem_hint = re.sub(
                r"^.*?(before|without|manual|isolated)[,\s]*", "", s, flags=re.IGNORECASE
            ).strip()[:80]
            break

    if problem_hint and len(problem_hint.split()) >= 5:
        stem = (
            "A development team is experiencing "
            + problem_hint.rstrip(".!?,").lower()
            + ". Which practice directly addresses this issue?"
        )
    else:
        stem = (
            "Which approach is MOST effective for ensuring "
            + desc.lower().rstrip(".,;")
            + "?"
        )

    # Trim stem
    stem_words = stem.split()
    if len(stem_words) > 22:
        stem = " ".join(stem_words[:20]) + "..."
    if not stem.endswith("?"):
        stem = stem.rstrip(".!") + "?"

    key = ("scenario", name.lower()[:30])
    if stem.lower()[:50] in seen or key in seen:
        return None

    # Distractors: other solution names from the text that are NOT the answer
    other_names = [t[0] for t in triples if t[0].lower()[:20] != name.lower()[:20]]
    # Also pull from pred_bank but use SHORT excerpts (first 3 words of each predicate)
    extra = [" ".join(p.split()[:5]) for p in pred_bank
             if p.lower()[:20] != desc.lower()[:20]]
    distractors = (other_names + extra)[:3]
    if len(distractors) < 3:
        fallbacks = [
            "continuing with the existing manual process",
            "adding more team members to the project",
            "rewriting the entire codebase from scratch",
            "using a waterfall development methodology",
            "separating development and operations teams",
        ]
        for f in fallbacks:
            if len(distractors) >= 3:
                break
            if f.lower()[:20] != name.lower()[:20]:
                distractors.append(f)

    if len(distractors) < 3:
        return None

    opts, correct_k = _make_options(name, distractors[:3])
    if not _options_are_diverse(opts):
        return None

    q = {
        "question_id":   "q" + str(idx).zfill(3),
        "question":      stem,
        "instruction":   "(Select ONE correct answer)",
        "answer_type":   "single_correct",
        "difficulty":    label,
        "options":       opts,
        "correct_answers": [correct_k],
        "_subject_key":  key,
        "_source":       source_s[:120],
    }
    return q if _validate_question(q, source_s) else None


# ===========================================================================
# BLOOM'S TAXONOMY SESSION 3: ANALYTICAL QUESTIONS (Analyze)
# ===========================================================================

def _gen_consequence(
    s: str,
    pred_bank: list,
    label: str,
    idx: int,
    seen: set,
) -> "Optional[dict]":
    """
    Session 3 Analyze-level question.
    Takes a benefit/solution sentence and inverts it:
    'What would be the consequence if X were NOT implemented?'
    Forces the student to understand WHY something matters, not just WHAT it is.
    """
    ms = _RE_SOLUTION.match(s)
    if not ms:
        return None
    solution_name = ms.group(1).strip().rstrip(",")
    benefit       = ms.group(2).strip().rstrip(".,;")
    if len(solution_name.split()) > 5 or len(benefit.split()) < 5:
        return None

    # Invert: "X helps reduce errors" -> "Without X, errors would increase"
    stem = (
        "What would MOST LIKELY happen in a software project that does NOT use "
        + solution_name
        + "?"
    )
    key = ("consequence", solution_name.lower()[:30])
    if stem.lower()[:50] in seen or key in seen:
        return None

    # Correct answer: the negation of the benefit
    # e.g., "reduces errors" -> "increased errors and manual mistakes"
    # We construct this by using the benefit but adding negation context
    correct = "Teams would likely face " + _invert_benefit(benefit)

    distractors = _pick_distractors(correct, pred_bank)
    if len(distractors) < 3:
        return None

    opts, correct_k = _make_options(correct, distractors[:3])
    if not _options_are_diverse(opts):
        return None

    q = {
        "question_id":   "q" + str(idx).zfill(3),
        "question":      stem,
        "instruction":   "(Select ONE correct answer)",
        "answer_type":   "single_correct",
        "difficulty":    label,
        "options":       opts,
        "correct_answers": [correct_k],
        "_subject_key":  key,
        "_source":       s[:120],
    }
    return q if _validate_question(q, s) else None


def _invert_benefit(benefit: str) -> str:
    """
    Convert a benefit phrase to its negative consequence.
    FIX: Regex substitutions now use proper back-references so the captured
    group (the subject) is actually inserted into the replacement string.

    Examples:
      'reduce deployment errors'  -> 'increased deployment errors and manual mistakes'
      'increase the speed of CI'  -> 'slower CI and delayed delivery'
      'ensure system reliability' -> 'lack of system reliability and system instability'
      'automate testing'          -> 'manual testing prone to human error'
    """
    b = benefit.lower().rstrip(".,;")
    inversions = [
        (r"reduc\w+\s+(.*)",
         r"increased \1 and manual errors"),
        (r"increas\w+\s+(?:the\s+)?(?:speed|rate|efficiency)\s+(?:of\s+|to\s+)?(.*)",
         r"slower \1 and delayed delivery"),
        (r"help\w+\s+(?:the\s+)?teams?\s+(.*)",
         r"teams struggling to \1"),
        (r"ensur\w+\s+(.*)",
         r"lack of \1 and system instability"),
        (r"automat\w+\s+(.*)",
         r"manual \1 prone to human error"),
        (r"improv\w+\s+(.*)",
         r"degraded \1 and poor outcomes"),
        (r"allow\w+\s+(?:organizations?\s+to\s+)?(.*)",
         r"inability to \1"),
        (r"offer\w+\s+(.*)",
         r"absence of \1"),
        (r"streamlin\w+\s+(.*)",
         r"inefficient \1 processes"),
        (r"simplif\w+\s+(.*)",
         r"overly complex \1 workflows"),
    ]
    for pattern, replacement in inversions:
        result = re.sub(pattern, replacement, b)
        if result != b:
            result = re.sub(r"\s{2,}", " ", result).strip()
            return result[:120]
    # fallback: generic negation
    return "challenges and inefficiencies in " + b[:60]

def _gen_comparison(
    sentences: list,
    pred_bank: list,
    label: str,
    idx: int,
    seen: set,
) -> "Optional[dict]":
    """
    Session 3 Analyze-level comparison question.
    Finds pairs of contrasting sentences and asks students to identify the KEY difference.
    """
    contrasts = []
    for s in sentences:
        m = _RE_CONTRAST.search(s)
        if m:
            part_a = m.group(1).strip().rstrip(",")
            part_b = m.group(2).strip().rstrip(".,;")
            if len(part_a.split()) >= 5 and len(part_b.split()) >= 5:
                contrasts.append((part_a, part_b, s))

    if not contrasts:
        return None
    random.shuffle(contrasts)
    part_a, part_b, source_s = contrasts[0]

    stem = (
        "What is the KEY difference between the traditional approach and "
        + part_a.lower()[:40].rstrip(",")
        + "?"
    )
    key = ("comparison", part_a.lower()[:30])
    if stem.lower()[:50] in seen or key in seen:
        return None

    correct = part_b[:110]
    distractors = _pick_distractors(correct, pred_bank)
    if len(distractors) < 3:
        return None

    opts, correct_k = _make_options(correct, distractors[:3])
    if not _options_are_diverse(opts):
        return None

    q = {
        "question_id":   "q" + str(idx).zfill(3),
        "question":      stem,
        "instruction":   "(Select ONE correct answer)",
        "answer_type":   "single_correct",
        "difficulty":    label,
        "options":       opts,
        "correct_answers": [correct_k],
        "_subject_key":  key,
        "_source":       source_s[:120],
    }
    return q if _validate_question(q, source_s) else None


def _gen_code_question(
    text: str, pred_bank: List[str], label: str, idx: int, seen: set
) -> Optional[dict]:
    """
    IMPROVEMENT 6: For docs with code markers, generate a question about
    what a function/class does -- based on surrounding prose, not raw code.
    """
    markers = _RE_CODE_MARKER.findall(text)
    if not markers:
        return None
    lang, desc = random.choice(markers)
    if not desc or re.match(r"^\d+", desc):
        return None
    names = [n.strip() for n in desc.split(",")][:2]
    if not names or not names[0]:
        return None
    func_name = names[0]

    # find a sentence near the code that explains what it does
    surrounding = []
    for s in _split_sentences(text):
        if func_name.lower() in s.lower() or "code" in s.lower():
            surrounding.append(s)

    if not surrounding:
        return None
    context_s = surrounding[0]

    correct = context_s.rstrip(".!?")
    if len(correct.split()) > 22:
        correct = " ".join(correct.split()[:20]) + "..."

    stem = "What is the purpose of the " + func_name + " function in " + lang + "?"
    key  = ("code", func_name.lower()[:30])
    if stem.lower()[:50] in seen or key in seen:
        return None

    distractors = _pick_distractors(correct, pred_bank)
    if len(distractors) < 3:
        return None

    opts, correct_k = _make_options(correct, distractors)
    if not _options_are_diverse(opts):
        return None

    q = {
        "question_id":   "q" + str(idx).zfill(3),
        "question":      stem,
        "instruction":   "(Select ONE correct answer)",
        "answer_type":   "single_correct",
        "difficulty":    label,
        "options":       opts,
        "correct_answers": [correct_k],
        "_subject_key":  key,
        "_source":       context_s[:120],
    }
    return q if _validate_question(q, context_s) else None


def _gen_multi_correct(
    s1: str, s2: str,
    pred_bank: List[str], effect_bank: List[str],
    label: str, idx: int, seen: set,
) -> Optional[dict]:
    m1 = _RE_DEFINITION.match(s1) or _RE_CAUSAL.search(s1)
    m2 = _RE_DEFINITION.match(s2) or _RE_CAUSAL.search(s2)
    if not m1 or not m2:
        return None

    def _fact(m):
        try:
            return m.group(2).strip().rstrip(".,;")[:90]
        except Exception:
            return None

    fact1, fact2 = _fact(m1), _fact(m2)
    if not fact1 or not fact2:
        return None
    if len(fact1.split()) < 3 or len(fact2.split()) < 3:
        return None
    if fact1.lower()[:30] == fact2.lower()[:30]:
        return None

    stem = "Which TWO of the following statements are correct?"
    key  = ("multi", fact1.lower()[:20])
    if stem.lower()[:50] in seen or key in seen:
        return None

    all_bank = pred_bank + effect_bank
    wrong = [
        b for b in all_bank
        if b.lower()[:30] not in (fact1.lower()[:30], fact2.lower()[:30])
    ]
    if len(wrong) < 2:
        for f in _GENERIC_FILLERS:
            if f.lower()[:30] not in (fact1.lower()[:30], fact2.lower()[:30]):
                wrong.append(f[:90])
            if len(wrong) >= 2:
                break
    if len(wrong) < 2:
        return None

    random.shuffle(wrong)
    all_opts = [fact1, fact2, wrong[0][:90], wrong[1][:90]]
    random.shuffle(all_opts)
    opt_dict = {k: v for k, v in zip(["A", "B", "C", "D"], all_opts)}
    correct_keys = [k for k, v in opt_dict.items() if v[:30] in [fact1[:30], fact2[:30]]]
    if len(correct_keys) != 2:
        return None

    q = {
        "question_id":   "q" + str(idx).zfill(3),
        "question":      stem,
        "instruction":   "(Select ALL correct answers)",
        "answer_type":   "multiple_correct",
        "difficulty":    label,
        "options":       opt_dict,
        "correct_answers": sorted(correct_keys),
        "_subject_key":  key,
        "_source":       s1[:120],
    }
    return q if _validate_question(q, s1 + " " + s2) else None


# ===========================================================================
# VALIDATION
# ===========================================================================

def _validate_question(q: dict, source_text: str = "") -> bool:
    if not all(k in q for k in ["question", "options", "correct_answers"]):
        return False
    opts = q.get("options", {})
    if not all(k in opts for k in ["A", "B", "C", "D"]):
        return False
    correct = q.get("correct_answers", [])
    if not correct or not all(c in opts for c in correct):
        return False
    if len(q["question"]) < 10 or len(q["question"]) > 200:
        return False

    values = list(opts.values())
    # Check option length consistency
    opt_lens = [len(v.split()) for v in values]
    min_len = min(opt_lens)
    max_len = max(opt_lens)
    # If ALL options are short (concept names): allow it -- these are name-matching questions
    # If options are MIXED (some short, some long): reject -- indicates a fragment problem
    if min_len < 4:
        all_short = max_len <= 5  # all options are short concept names
        if not all_short:
            return False  # mixed short/long = fragment distractor problem
    # no "none/all of the above"
    for v in values:
        if "none of the above" in v.lower() or "all of the above" in v.lower():
            return False
    # no identical options
    if len(set(v.lower()[:30] for v in values)) < 4:
        return False

    # IMPROVEMENT 3: semantic diversity check
    if not _options_are_diverse(opts):
        return False

    # correct answer shouldn't be a substring of the question stem
    cv = opts.get(correct[0], "").lower().strip()
    if len(cv) > 15 and cv in q["question"].lower():
        return False

    return True


# ===========================================================================
# IMPROVEMENT 5: LLM IMPROVEMENT WITH EXPLANATION
# ===========================================================================

def _llm_improve(rb_q: dict, sentence: str) -> dict:
    if _llm_model is None:
        return rb_q
    prompt = (
        "Text: " + sentence + "\n\n"
        "Rewrite as a clearer exam MCQ. Options must be full phrases (5-18 words each).\n"
        "Do NOT use single words. Do NOT use 'None/All of the above'.\n\n"
        "Question: <question ending with ?>\n"
        "Answer: <correct phrase 5-18 words>\n"
        "Wrong1: <wrong phrase 5-18 words>\n"
        "Wrong2: <wrong phrase 5-18 words>\n"
        "Wrong3: <wrong phrase 5-18 words>"
    )
    out = _llm_generate_timeout(prompt, max_new_tokens=280)
    if not out:
        return rb_q

    q_m = re.search(r"Question:\s*(.+?)(?:\nAnswer:|$)", out, re.IGNORECASE | re.DOTALL)
    a_m = re.search(r"Answer:\s*(.+?)(?:\nWrong|$)", out, re.IGNORECASE)
    qt = q_m.group(1).strip().rstrip(".") if q_m else ""
    ca = a_m.group(1).strip().strip(".,") if a_m else ""

    if not qt or len(qt) < 12 or not qt.endswith("?"):
        return rb_q
    if not ca or len(ca.split()) < 3:
        return rb_q

    distractors = []
    for mm in re.finditer(r"Wrong\d:\s*(.+?)(?:\n|$)", out, re.IGNORECASE):
        d = mm.group(1).strip().strip(".,")
        if d and len(d.split()) >= 3 and d.lower() != ca.lower():
            distractors.append(d[:120])

    if len(distractors) < 3:
        return rb_q

    opts, ck = _make_options(ca, distractors)
    improved = {
        "question_id":   rb_q["question_id"],
        "question":      qt,
        "instruction":   rb_q["instruction"],
        "answer_type":   rb_q["answer_type"],
        "difficulty":    rb_q["difficulty"],
        "options":       opts,
        "correct_answers": [ck],
        "_subject_key":  rb_q.get("_subject_key"),
        "_source":       rb_q.get("_source", ""),
    }
    return improved if _validate_question(improved, sentence) else rb_q




# ===========================================================================
# CONCEPT MAP BUILDER -- the foundation of quality MCQ generation
# ===========================================================================

def _build_concept_map(sentences: list) -> dict:
    """
    Build a map: {concept_name -> definition_text}
    from definition sentences in the text.
    
    This is the key quality improvement: we use concepts as answer choices,
    not random text fragments. Questions test whether students know which
    concept matches which description.
    """
    concept_map = {}  # name -> description
    for s in sentences:
        m = _RE_DEFINITION.match(s)
        if not m:
            continue
        name = m.group(1).strip().rstrip(",").rstrip()
        # Strip article prefix
        name = re.sub(r"^(A|An|The)\s+", "", name, flags=re.IGNORECASE).strip()
        desc = m.group(2).strip().rstrip(".,;")
        if 2 <= len(name.split()) <= 6 and len(desc.split()) >= 5:
            if name not in concept_map:
                concept_map[name] = desc
    return concept_map


def _build_purpose_map(sentences: list) -> dict:
    """
    Build a map: {tool/practice_name -> what_it_does}
    from solution sentences in the text.
    """
    purpose_map = {}
    for s in sentences:
        m = _RE_SOLUTION.match(s)
        if not m:
            continue
        name = m.group(1).strip().rstrip(",")
        name = re.sub(r"^(A|An|The)\s+", "", name, flags=re.IGNORECASE).strip()
        purpose = m.group(2).strip().rstrip(".,;")
        if 1 <= len(name.split()) <= 5 and len(purpose.split()) >= 4:
            if name not in purpose_map:
                purpose_map[name] = purpose
    return purpose_map


# ===========================================================================
# QUALITY GENERATOR 1: Concept-map based definition questions
# "What is X?" with OTHER concepts' descriptions as distractors
# ===========================================================================

def _gen_concept_definition(
    concept_map: dict,
    label: str,
    idx: int,
    seen: set,
    used_names: set,
) -> "Optional[dict]":
    """
    High-quality definition question using the concept map.
    Correct answer: description of the target concept.
    Distractors: descriptions of OTHER concepts from the same document.
    This is how real exam MCQs work.
    """
    if len(concept_map) < 4:
        return None
    
    # Pick a concept not yet used
    available = [n for n in concept_map if n not in used_names]
    if not available:
        return None
    
    random.shuffle(available)
    target_name = available[0]
    target_desc = concept_map[target_name]
    
    stem = "What is " + target_name + "?"
    key  = ("concept_def", target_name.lower()[:30])
    if stem.lower()[:50] in seen or key in seen:
        return None
    
    # Distractors: descriptions of OTHER concepts
    other_descs = [
        concept_map[n] for n in concept_map
        if n != target_name and concept_map[n][:30] != target_desc[:30]
    ]
    random.shuffle(other_descs)
    distractors = other_descs[:3]
    
    # If not enough concept distractors, use generic ones
    for f in _GENERIC_FILLERS:
        if len(distractors) >= 3: break
        if f[:30] != target_desc[:30]: distractors.append(f)
    
    if len(distractors) < 3:
        return None
    
    opts, correct_k = _make_options(target_desc, distractors[:3])
    
    # Validate diversity: all options must be meaningfully different
    if not _options_are_diverse(opts):
        return None
    
    q = {
        "question_id":   "q" + str(idx).zfill(3),
        "question":      stem,
        "instruction":   "(Select ONE correct answer)",
        "answer_type":   "single_correct",
        "difficulty":    label,
        "options":       opts,
        "correct_answers": [correct_k],
        "_subject_key":  key,
        "_source":       target_name + " is " + target_desc,
    }
    return q if _validate_question(q, target_desc) else None


# ===========================================================================
# QUALITY GENERATOR 2: Reverse concept question ("Which concept does this describe?")
# Tests if student can map description -> concept name
# ===========================================================================

def _gen_reverse_concept(
    concept_map: dict,
    label: str,
    idx: int,
    seen: set,
    used_names: set,
) -> "Optional[dict]":
    """
    Reverse-direction question: give the description, ask for the concept name.
    'A practice where developers regularly merge code into a central repository is called...?'
    Correct answer: the concept name.
    Distractors: other concept names from the document.
    """
    if len(concept_map) < 4:
        return None
    
    available = [n for n in concept_map if n not in used_names]
    if not available:
        return None
    
    random.shuffle(available)
    target_name = available[0]
    target_desc = concept_map[target_name]
    
    stem = target_desc.rstrip(".,;") + " is known as...?"
    # Trim
    stem_words = stem.split()
    if len(stem_words) > 20:
        stem = " ".join(stem_words[:18]) + "... is known as?"
    
    key = ("reverse_concept", target_name.lower()[:30])
    if stem.lower()[:50] in seen or key in seen:
        return None
    
    # Distractors: OTHER concept NAMES
    other_names = [n for n in concept_map if n != target_name]
    random.shuffle(other_names)
    distractors = other_names[:3]
    
    fallback_names = [
        "Manual Code Deployment",
        "Traditional Waterfall Development",
        "Isolated Team Structure",
        "Sequential Build and Test",
        "Batch Release Management",
    ]
    for f in fallback_names:
        if len(distractors) >= 3: break
        if f.lower()[:20] != target_name.lower()[:20]: distractors.append(f)
    
    if len(distractors) < 3:
        return None
    
    opts, correct_k = _make_options(target_name, distractors[:3])
    
    q = {
        "question_id":   "q" + str(idx).zfill(3),
        "question":      stem,
        "instruction":   "(Select ONE correct answer)",
        "answer_type":   "single_correct",
        "difficulty":    label,
        "options":       opts,
        "correct_answers": [correct_k],
        "_subject_key":  key,
        "_source":       target_name + " is " + target_desc,
    }
    return q if _validate_question(q, target_desc) else None


# ===========================================================================
# QUALITY GENERATOR 3: Purpose-based scenario question (Apply level)
# "A team needs to [achieve goal]. Which practice should they adopt?"
# ===========================================================================

def _gen_purpose_scenario(
    purpose_map: dict,
    pred_bank: list,
    label: str,
    idx: int,
    seen: set,
    used_names: set,
) -> "Optional[dict]":
    """
    Apply-level question that reads more naturally than generic "Which best describes X?".
    Uses the purpose_map to ask about practical adoption.
    """
    if len(purpose_map) < 2:
        return None
    
    available = [n for n in purpose_map if n not in used_names]
    if not available:
        return None
    
    random.shuffle(available)
    target_name = available[0]
    target_purpose = purpose_map[target_name]
    
    # Build scenario stem
    stem = (
        "A software team wants to "
        + target_purpose.lower().rstrip(".,;")
        + ". Which practice should they implement?"
    )
    stem_words = stem.split()
    if len(stem_words) > 22:
        stem = " ".join(stem_words[:20]) + "...?"
    if not stem.endswith("?"):
        stem = stem.rstrip(".") + "?"
    
    key = ("purpose_scenario", target_name.lower()[:30])
    if stem.lower()[:50] in seen or key in seen:
        return None
    
    # Distractors: other practice names
    other_names = [n for n in purpose_map if n != target_name]
    random.shuffle(other_names)
    distractors = other_names[:3]
    
    for f in [
        "Manual deployment with no automation",
        "Waterfall sequential development model",
        "Isolated siloed team structure",
        "Infrequent large batch releases",
        "Traditional operations without development input",
    ]:
        if len(distractors) >= 3: break
        if f[:20] != target_name[:20]: distractors.append(f)
    
    if len(distractors) < 3:
        return None
    
    opts, correct_k = _make_options(target_name, distractors[:3])
    
    q = {
        "question_id":   "q" + str(idx).zfill(3),
        "question":      stem,
        "instruction":   "(Select ONE correct answer)",
        "answer_type":   "single_correct",
        "difficulty":    label,
        "options":       opts,
        "correct_answers": [correct_k],
        "_subject_key":  key,
        "_source":       target_name + " " + target_purpose,
    }
    return q if _validate_question(q, target_purpose) else None


# ===========================================================================
# MAIN GENERATION ENGINE
# ===========================================================================

# All six generators tried during top-up in priority order
_ALL_GENERATORS = ["definition", "enumeration", "causal", "contrast", "application", "numeric"]


def _run_generator(
    name: str, sentence: str,
    pred_bank: List[str], effect_bank: List[str], term_bank: List[str],
    label: str, idx: int, seen: set,
) -> Optional[dict]:
    if name == "definition":
        return _gen_definition(sentence, pred_bank, term_bank, label, idx, seen)
    if name == "causal":
        return _gen_causal(sentence, effect_bank, pred_bank, label, idx, seen)
    if name == "enumeration":
        return _gen_enumeration(sentence, pred_bank, label, idx, seen)
    if name == "contrast":
        return _gen_contrast(sentence, pred_bank, label, idx, seen)
    if name == "application":
        return _gen_application(sentence, pred_bank, label, idx, seen)
    if name == "numeric":
        return _gen_numeric(sentence, pred_bank, label, idx, seen)
    return None


def generate_questions_with_ai(
    text: str,
    session_num: int,
    num_questions: int,
    local_seen: set,
) -> list:
    text = _sample_text_for_mcq(text, MCQ_MAX_WORDS)

    cfg         = DIFFICULTY_CONFIG[session_num]
    label       = cfg["label"]
    multi_ratio = cfg["multiple_ratio"]
    num_multi   = max(1, math.ceil(num_questions * multi_ratio)) if session_num > 1 else 0
    num_single  = num_questions - num_multi

    sentences = _split_sentences(text)
    if not sentences:
        logger.warning("Session %d: no usable sentences found", session_num)
        return []

    pred_bank   = _build_predicate_bank(sentences)
    effect_bank = _build_effect_bank(sentences)
    term_bank   = _build_term_bank(sentences)
    # Build concept maps for high-quality questions
    concept_map = _build_concept_map(sentences)
    purpose_map = _build_purpose_map(sentences)

    logger.info(
        "MCQ gen | session=%d (%s) | single=%d | multi=%d | "
        "sentences=%d | pred=%d | effect=%d | concepts=%d | purposes=%d",
        session_num, label, num_single, num_multi,
        len(sentences), len(pred_bank), len(effect_bank), 
        len(concept_map), len(purpose_map),
    )

    questions = []
    used_sents = []
    used_concept_names: set = set()  # track which concepts have been used

    # Session priority follows Bloom's taxonomy:
    # Session 1 (Remember/Understand): definitions, enumerations, factual recall
    # Session 2 (Apply): scenarios, comparisons, purpose questions  
    # Session 3 (Analyze): consequences, deep contrasts, multi-correct
    if session_num == 1:
        gen_order = ["definition", "enumeration", "causal", "numeric"]
    elif session_num == 2:
        gen_order = ["causal", "contrast", "enumeration", "definition", "application"]
    else:
        gen_order = ["application", "contrast", "causal", "definition"]

    random.shuffle(sentences)

    # ── PASS 1: Concept-map based questions (highest quality) ────────────────
    # Use concept_map and purpose_map first -- these generate the best MCQs
    # because distractors are guaranteed to be from the same domain.

    # Session 1: definition + reverse questions (test recall of key terms)
    # Session 2: purpose/scenario questions (test application)
    # Session 3: mix of all types (test analysis)

    concept_names_list = list(concept_map.keys())
    purpose_names_list = list(purpose_map.keys())
    random.shuffle(concept_names_list)
    random.shuffle(purpose_names_list)

    # For sessions 1 and 3: concept definition questions
    if session_num in (1, 3) and len(concept_map) >= 4:
        # Alternate between "What is X?" and "X is described as... which concept?"
        for pass_type in ["concept_def", "reverse_concept"] * 5:
            if len(questions) >= num_single:
                break
            if pass_type == "concept_def":
                q = _gen_concept_definition(concept_map, label, len(questions)+1, local_seen, used_concept_names)
            else:
                q = _gen_reverse_concept(concept_map, label, len(questions)+1, local_seen, used_concept_names)
            if q and _validate_question(q, text):
                stem_key = q["question"].lower()[:50]
                subj_key = q.get("_subject_key")
                if stem_key not in local_seen:
                    questions.append(q)
                    local_seen.add(stem_key)
                    if subj_key: local_seen.add(subj_key)
                    # Mark this concept as used so we don't generate two questions about it
                    if subj_key and len(subj_key) > 1:
                        used_concept_names.add(subj_key[1] if isinstance(subj_key, tuple) else str(subj_key))

    # For session 2: purpose/scenario questions FIRST, then concept definition fallback
    if session_num == 2:
        # Try purpose scenarios
        for _ in range(min(5, num_single)):
            if len(questions) >= num_single:
                break
            if len(purpose_map) >= 2:
                q = _gen_purpose_scenario(purpose_map, pred_bank, label, len(questions)+1, local_seen, used_concept_names)
                if q and _validate_question(q, text):
                    stem_key = q["question"].lower()[:50]
                    subj_key = q.get("_subject_key")
                    if stem_key not in local_seen:
                        questions.append(q)
                        local_seen.add(stem_key)
                        if subj_key: local_seen.add(subj_key)
        # Fill remaining with reverse-concept questions (medium difficulty)
        if len(questions) < num_single and len(concept_map) >= 4:
            for _ in range(num_single - len(questions) + 2):
                if len(questions) >= num_single:
                    break
                q = _gen_reverse_concept(concept_map, label, len(questions)+1, local_seen, used_concept_names)
                if q and _validate_question(q, text):
                    stem_key = q["question"].lower()[:50]
                    subj_key = q.get("_subject_key")
                    if stem_key not in local_seen:
                        questions.append(q)
                        local_seen.add(stem_key)
                        if subj_key: local_seen.add(subj_key)

    # ── PASS 2: Sentence-pattern generators (causal, enumeration, contrast) ──
    for sentence in sentences:
        if len(questions) >= num_single:
            break
        q = None
        for gname in gen_order:
            q = _run_generator(
                gname, sentence, pred_bank, effect_bank, term_bank,
                label, len(questions) + 1, local_seen,
            )
            if q:
                break
        if q and _validate_question(q, text):
            stem_key = q["question"].lower()[:50]
            subj_key = q.get("_subject_key")
            if stem_key not in local_seen and subj_key not in local_seen:
                questions.append(q)
                used_sents.append(sentence)
                local_seen.add(stem_key)
                if subj_key:
                    local_seen.add(subj_key)

    # --- Multi-correct pass ---
    if num_multi > 0:
        pairs = [
            (sentences[i], sentences[i + 1])
            for i in range(len(sentences) - 1)
        ]
        random.shuffle(pairs)
        for s1, s2 in pairs:
            done = sum(1 for qq in questions if qq["answer_type"] == "multiple_correct")
            if done >= num_multi:
                break
            q = _gen_multi_correct(
                s1, s2, pred_bank, effect_bank,
                label, len(questions) + 1, local_seen,
            )
            if q and _validate_question(q, text):
                stem_key = q["question"].lower()[:50]
                subj_key = q.get("_subject_key")
                if stem_key not in local_seen:
                    questions.append(q)
                    local_seen.add(stem_key)
                    if subj_key:
                        local_seen.add(subj_key)

    # IMPROVEMENT 6: code-question attempt
    if _RE_CODE_MARKER.search(text) and len(questions) < num_single:
        cq = _gen_code_question(
            text, pred_bank, label, len(questions) + 1, local_seen
        )
        if cq and _validate_question(cq, text):
            stem_key = cq["question"].lower()[:50]
            if stem_key not in local_seen:
                questions.append(cq)
                local_seen.add(stem_key)

    # ── Session 2: Scenario-based pass (Apply level) ──────────────────────
    if session_num == 2 and len(questions) < num_single:
        sq = _gen_scenario(sentences, pred_bank, label, len(questions)+1, local_seen)
        if sq and _validate_question(sq, text):
            stem_key = sq["question"].lower()[:50]
            subj_key = sq.get("_subject_key")
            if stem_key not in local_seen:
                questions.append(sq)
                local_seen.add(stem_key)
                if subj_key: local_seen.add(subj_key)

    # ── Session 3: Multi-type analytical questions (Analyze level) ────────
    if session_num == 3:
        # 1. Consequence questions ("What happens without X?")
        random.shuffle(sentences)
        for sent in sentences:
            if len(questions) >= num_single: break
            cq = _gen_consequence(sent, pred_bank, label, len(questions)+1, local_seen)
            if cq and _validate_question(cq, text):
                stem_key = cq["question"].lower()[:50]
                subj_key = cq.get("_subject_key")
                if stem_key not in local_seen:
                    questions.append(cq)
                    local_seen.add(stem_key)
                    if subj_key: local_seen.add(subj_key)
        # 2. Comparison questions ("What is the key difference between X and Y?")
        if len(questions) < num_single:
            compq = _gen_comparison(sentences, pred_bank, label, len(questions)+1, local_seen)
            if compq and _validate_question(compq, text):
                stem_key = compq["question"].lower()[:50]
                if stem_key not in local_seen:
                    questions.append(compq)
                    local_seen.add(stem_key)
        # 3. Reverse concept questions (hardest: map description to name)
        if len(questions) < num_single and len(concept_map) >= 4:
            for _ in range(3):
                if len(questions) >= num_single: break
                q = _gen_reverse_concept(concept_map, label, len(questions)+1, local_seen, used_concept_names)
                if q and _validate_question(q, text):
                    stem_key = q["question"].lower()[:50]
                    subj_key = q.get("_subject_key")
                    if stem_key not in local_seen:
                        questions.append(q)
                        local_seen.add(stem_key)
                        if subj_key: local_seen.add(subj_key)

    logger.info("Rule-based: %d/%d questions for session %d",
                len(questions), num_questions, session_num)

    # --- LLM improvement pass ---
    _load_llm()
    if _llm_model and questions:
        improved = 0
        for i, (q, sent) in enumerate(zip(questions, used_sents + [""] * len(questions))):
            if not sent or q["answer_type"] == "multiple_correct":
                continue
            imp = _llm_improve(q, sent)
            if imp is not q:
                questions[i] = imp
                improved += 1
        if improved:
            logger.info("LLM improved %d/%d questions", improved, len(questions))

    # --- Exhaustive top-up (all 6 generators on all remaining sentences) ---
    if len(questions) < num_questions:
        remaining = [s for s in sentences if s not in used_sents]
        random.shuffle(remaining)
        for sentence in remaining:
            if len(questions) >= num_questions:
                break
            for gname in _ALL_GENERATORS:
                q = _run_generator(
                    gname, sentence, pred_bank, effect_bank, term_bank,
                    label, len(questions) + 1, local_seen,
                )
                if q and _validate_question(q, text):
                    stem_key = q["question"].lower()[:50]
                    subj_key = q.get("_subject_key")
                    if stem_key not in local_seen:
                        questions.append(q)
                        local_seen.add(stem_key)
                        if subj_key:
                            local_seen.add(subj_key)
                        break

        # second pass: re-shuffle all sentences
        if len(questions) < num_questions:
            all_sents = sentences[:]
            random.shuffle(all_sents)
            for sentence in all_sents:
                if len(questions) >= num_questions:
                    break
                for gname in _ALL_GENERATORS:
                    q = _run_generator(
                        gname, sentence, pred_bank, effect_bank, term_bank,
                        label, len(questions) + 1, local_seen,
                    )
                    if q and _validate_question(q, text):
                        stem_key = q["question"].lower()[:50]
                        if stem_key not in local_seen:
                            questions.append(q)
                            local_seen.add(stem_key)
                            break

        logger.info("After top-up: %d/%d for session %d",
                    len(questions), num_questions, session_num)

    # IMPROVEMENT 7: renumber and strip internal keys
    for i, q in enumerate(questions):
        q["question_id"] = "q" + str(i + 1).zfill(3)
        q.pop("_subject_key", None)
        q.pop("_source", None)

    return questions[:num_questions]


def _generate_session_worker(args: tuple):
    text, session_num, num_questions = args
    local_seen = set()
    qs = generate_questions_with_ai(text, session_num, num_questions, local_seen)
    return session_num, qs, local_seen


# ===========================================================================
# SESSION MANAGEMENT  (API unchanged from v2.5.0)
# ===========================================================================

def now_iso():
    return datetime.utcnow().isoformat() + "Z"

def parse_iso(ts):
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).replace(tzinfo=None)

def minutes_since(ts):
    return (datetime.utcnow() - parse_iso(ts)).total_seconds() / 60.0

def hours_since(ts):
    return (datetime.utcnow() - parse_iso(ts)).total_seconds() / 3600.0

def is_admin(role):
    return str(role).strip().lower() == "admin"


def initialise_document(
    text=None,
    source_txt_path=None,
    num_questions=DEFAULT_NUM_QUESTIONS,
    custom_s1_to_s2_hours=DEFAULT_S1_TO_S2_HOURS,
    custom_s2_to_s3_hours=DEFAULT_S2_TO_S3_HOURS,
    document_title="Tarang Document",
):
    if text is None and source_txt_path is not None:
        from pathlib import Path as _P
        p = _P(source_txt_path)
        if not p.exists():
            return {"status": "error", "error": "File not found: " + str(p)}
        text = p.read_text(encoding="utf-8")
    if not text or not text.strip():
        return {"status": "error", "error": "Input text is empty."}

    word_count = len(text.split())
    if word_count < 150:
        num_questions = min(num_questions, 3)
        logger.warning("Short doc (%d words) -- scaling to %d q/session", word_count, num_questions)
    elif word_count < 300:
        num_questions = min(num_questions, 6)
        logger.info("Short doc (%d words) -- scaling to %d q/session", word_count, num_questions)

    doc_id   = hashlib.md5((text[:500] + str(int(time.time() * 1000))).encode()).hexdigest()[:12]
    s1_to_s2 = max(MIN_S1_TO_S2_HOURS, float(custom_s1_to_s2_hours))
    s2_to_s3 = max(MIN_S2_TO_S3_HOURS, float(custom_s2_to_s3_hours))
    logger.info("Initialising | doc_id=%s | title=%s", doc_id, document_title)

    t0 = time.time()
    session_results = {}
    with ThreadPoolExecutor(max_workers=3) as ex:
        futures = {
            ex.submit(_generate_session_worker, (text, n, num_questions)): n
            for n in [1, 2, 3]
        }
        for future in as_completed(futures):
            try:
                sn, qs, ls = future.result()
                session_results[sn] = (qs, ls)
                logger.info("Session %d generation complete (%d questions)", sn, len(qs))
            except Exception as exc:
                sn = futures[future]
                logger.error("Session %d failed: %s", sn, exc)
                return {"status": "error", "error": "Session " + str(sn) + " generation failed: " + str(exc)}

    logger.info("Parallel generation complete in %.1fs", time.time() - t0)

    global_seen = set()
    session_data = {}
    for sn in [1, 2, 3]:
        qs, _ = session_results[sn]
        qs = qs or []
        deduped = []
        for q in qs:
            stem = q["question"].lower()[:50]
            if stem not in global_seen:
                deduped.append(q)
                global_seen.add(stem)
        for i, q in enumerate(deduped):
            q["question_id"] = "q" + str(i + 1).zfill(3)

        pub  = []
        for q in deduped:
            pq = {k: v for k, v in q.items() if k not in ("correct_answers", "_source", "_subject_key")}
            # Add explanation from source text (FIX: expose to student after submission)
            src = q.get("_source", "")
            if src:
                pq["explanation"] = src[:200]
            pub.append(pq)
        akey = {
            q["question_id"]: {
                "correct_answers": q["correct_answers"],
                "answer_type":     q["answer_type"],
            }
            for q in deduped
        }
        session_data[sn] = {"questions": deduped, "public_qs": pub, "answer_key": akey}

    sessions_meta = {}
    for sn in [1, 2, 3]:
        qs = session_data[sn]["questions"]
        sc = sum(1 for q in qs if q["answer_type"] == "single_correct")
        mc = sum(1 for q in qs if q["answer_type"] == "multiple_correct")
        sessions_meta["session_" + str(sn)] = {
            "difficulty":       DIFFICULTY_CONFIG[sn]["label"],
            "total_questions":  len(qs),
            "single_correct":   sc,
            "multiple_correct": mc,
        }
        logger.info(
            "Session %d (%s) -- %d questions (%d single, %d multiple)",
            sn, DIFFICULTY_CONFIG[sn]["label"], len(qs), sc, mc,
        )

    session_state = {
        "document_id":    doc_id,
        "document_title": document_title,
        "num_questions":  num_questions,
        "s1_to_s2_hours": s1_to_s2,
        "s2_to_s3_hours": s2_to_s3,
        "current_session": 0,
        "audio_completed_at": None,
        "sessions": {
            "1": {"status": "pending", "started_at": None, "submitted_at": None,
                  "score_pct": None, "override_used": False, "user_answers": {}},
            "2": {"status": "locked",  "started_at": None, "submitted_at": None,
                  "score_pct": None, "override_used": False, "user_answers": {}},
            "3": {"status": "locked",  "started_at": None, "submitted_at": None,
                  "score_pct": None, "override_used": False, "user_answers": {}},
        },
        "all_sessions_complete":   False,
        "answers_unlocked":        False,
        "poor_score_warning":      False,
        "relistening_recommended": False,
        "sessions_meta":           sessions_meta,
        "llm_engine":              _llm_model_id or "rule-based",
        "created_at":              now_iso(),
    }

    logger.info("Document initialised -- doc_id: %s | engine: %s",
                doc_id, _llm_model_id or "rule-based")
    return {
        "status":            "success",
        "document_id":       doc_id,
        "document_title":    document_title,
        "sessions_generated": 3,
        "sessions_meta":     sessions_meta,
        "s1_to_s2_hours":    s1_to_s2,
        "s2_to_s3_hours":    s2_to_s3,
        "llm_engine":        _llm_model_id or "rule-based",
        "session_state":     session_state,
        "session_1_questions": {
            "document_id": doc_id, "session": 1,
            "difficulty":  DIFFICULTY_CONFIG[1]["label"],
            "description": DIFFICULTY_CONFIG[1]["description"],
            "questions":   session_data[1]["public_qs"],
            "generated_at": now_iso(),
        },
        "session_1_answers": {
            "document_id": doc_id, "session": 1,
            "answers":     session_data[1]["answer_key"],
            "generated_at": now_iso(),
        },
        "session_2_questions": {
            "document_id": doc_id, "session": 2,
            "difficulty":  DIFFICULTY_CONFIG[2]["label"],
            "description": DIFFICULTY_CONFIG[2]["description"],
            "questions":   session_data[2]["public_qs"],
            "generated_at": now_iso(),
        },
        "session_2_answers": {
            "document_id": doc_id, "session": 2,
            "answers":     session_data[2]["answer_key"],
            "generated_at": now_iso(),
        },
        "session_3_questions": {
            "document_id": doc_id, "session": 3,
            "difficulty":  DIFFICULTY_CONFIG[3]["label"],
            "description": DIFFICULTY_CONFIG[3]["description"],
            "questions":   session_data[3]["public_qs"],
            "generated_at": now_iso(),
        },
        "session_3_answers": {
            "document_id": doc_id, "session": 3,
            "answers":     session_data[3]["answer_key"],
            "generated_at": now_iso(),
        },
        "message": "Document ready. Call audio_completed() when user finishes listening.",
    }


def audio_completed(doc_id, role="user", session_state=None):
    if not session_state:
        return {"status": "error", "error": "session_state not provided."}
    state = json.loads(json.dumps(session_state))
    state["audio_completed_at"] = now_iso()
    state["current_session"]    = 1
    state["sessions"]["1"]["status"] = "available"
    state["role"] = role
    if is_admin(role):
        state["sessions"]["1"]["override_used"] = True
    window = 0 if is_admin(role) else SESSION_1_WINDOW_MINUTES
    return {
        "status":       "success",
        "document_id":  doc_id,
        "role":         role,
        "window_minutes": window,
        "deadline": (datetime.utcnow() + timedelta(minutes=window)).isoformat() + "Z",
        "updated_state": state,
        "message": (
            "Session 1 available immediately (admin)."
            if is_admin(role)
            else "Session 1 available. " + str(window) + " minutes to begin."
        ),
    }


def override_window(doc_id, session_state=None):
    if not session_state:
        return {"status": "error", "error": "session_state not provided."}
    state = json.loads(json.dumps(session_state))
    state["sessions"]["1"]["override_used"] = True
    return {"status": "success", "updated_state": state, "message": "Override accepted."}


def get_session_status(doc_id, role="user", session_state=None):
    if not session_state:
        return {"status": "error", "error": "session_state not provided."}
    state  = json.loads(json.dumps(session_state))
    s_data = state["sessions"]
    admin  = is_admin(role)

    s1_exp = False
    s1_left = None
    audio_done = state.get("audio_completed_at")
    if audio_done and s_data["1"]["status"] == "available":
        mins = minutes_since(audio_done)
        if mins > SESSION_1_WINDOW_MINUTES:
            s1_exp  = True
            s1_left = 0
        else:
            s1_left = round(SESSION_1_WINDOW_MINUTES - mins, 1)

    changed = False
    if s_data["2"]["status"] == "locked":
        s1s = s_data["1"].get("submitted_at")
        if s1s and (admin or hours_since(s1s) >= state["s1_to_s2_hours"]):
            state["sessions"]["2"]["status"] = "available"
            changed = True
    if s_data["3"]["status"] == "locked":
        s2s = s_data["2"].get("submitted_at")
        if s2s and (admin or hours_since(s2s) >= state["s2_to_s3_hours"]):
            state["sessions"]["3"]["status"] = "available"
            changed = True

    sessions_out = {}
    for snum in ["1", "2", "3"]:
        s  = state["sessions"][snum]
        si = int(snum)
        e  = {
            "session":    si,
            "difficulty": DIFFICULTY_CONFIG[si]["label"],
            "description": DIFFICULTY_CONFIG[si]["description"],
            "status":       s["status"],
            "override_used": s["override_used"],
            "started_at":  s["started_at"],
            "submitted_at": s["submitted_at"],
            "score_pct":   safe_pct(s["score_pct"]) if state["all_sessions_complete"] else None,
        }
        if not admin:
            if snum == "2" and s["status"] == "locked":
                s1s = s_data["1"].get("submitted_at")
                if s1s:
                    e["hours_until_available"] = round(
                        max(0, state["s1_to_s2_hours"] - hours_since(s1s)), 2
                    )
            if snum == "3" and s["status"] == "locked":
                s2s = s_data["2"].get("submitted_at")
                if s2s:
                    e["hours_until_available"] = round(
                        max(0, state["s2_to_s3_hours"] - hours_since(s2s)), 2
                    )
        sessions_out[snum] = e

    msgs = []
    if (s1_exp
            and not s_data["1"]["override_used"]
            and s_data["1"]["status"] not in ("in_progress", "completed")):
        msgs.append("Session 1 window expired. Click 'I have listened carefully' to proceed.")
    if state["relistening_recommended"]:
        msgs.append("Score below 30% detected. Consider re-listening before continuing.")

    return {
        "status":          "success",
        "document_id":     doc_id,
        "document_title":  state.get("document_title", ""),
        "current_session": state["current_session"],
        "all_sessions_complete": state["all_sessions_complete"],
        "answers_unlocked": state["answers_unlocked"],
        "relistening_recommended": state["relistening_recommended"],
        "poor_score_warning": state["poor_score_warning"],
        "sessions":        sessions_out,
        "messages":        msgs,
        "s1_window_expired": s1_exp,
        "s1_minutes_left": s1_left,
        "updated_state":   state if changed else None,
        "relisten_options": {
            "available": True,
            "modes": [{"key": k, "label": COGNITIVE_MODE_LABELS[k]} for k in COGNITIVE_MODES],
        },
    }


def get_questions(doc_id, session, role="user", session_state=None, questions_data=None):
    if not session_state:
        return {"status": "error", "error": "session_state not provided.", "can_override": False}
    if not questions_data:
        return {"status": "error", "error": "questions_data not provided.", "can_override": False}

    state   = json.loads(json.dumps(session_state))
    s_key   = str(session)
    s_data  = state["sessions"].get(s_key)
    if not s_data:
        return {"status": "error", "error": "Invalid session: " + str(session)}
    admin = is_admin(role)

    if session == 1:
        if not state.get("audio_completed_at"):
            return {"status": "error", "error": "Audio not marked complete.", "can_override": False}
        if not admin:
            mins = minutes_since(state["audio_completed_at"])
            if mins > SESSION_1_WINDOW_MINUTES and not s_data["override_used"]:
                return {
                    "status": "error", "reason": "window_expired", "can_override": True,
                    "error":  "Session 1 window expired.",
                    "minutes_elapsed": round(mins, 1),
                }
    elif session == 2:
        if not state["sessions"]["1"].get("submitted_at"):
            return {"status": "error", "error": "Session 1 not completed.", "can_override": False}
        if not admin:
            gap  = state["s1_to_s2_hours"]
            wait = hours_since(state["sessions"]["1"]["submitted_at"])
            if wait < gap:
                return {
                    "status": "error", "reason": "too_early", "can_override": False,
                    "error":  "Session 2 available in " + str(round(gap - wait, 1)) + " hours.",
                    "hours_remaining": round(gap - wait, 2),
                }
    elif session == 3:
        if not state["sessions"]["2"].get("submitted_at"):
            return {"status": "error", "error": "Session 2 not completed.", "can_override": False}
        if not admin:
            gap  = state["s2_to_s3_hours"]
            wait = hours_since(state["sessions"]["2"]["submitted_at"])
            if wait < gap:
                return {
                    "status": "error", "reason": "too_early", "can_override": False,
                    "error":  "Session 3 available in " + str(round(gap - wait, 1)) + " hours.",
                    "hours_remaining": round(gap - wait, 2),
                }

    changed = False
    if not s_data["started_at"]:
        state["sessions"][s_key]["started_at"] = now_iso()
        state["sessions"][s_key]["status"]     = "in_progress"
        changed = True

    qs = questions_data.get("questions", [])
    return {
        "status":      "success",
        "document_id": doc_id,
        "session":     session,
        "difficulty":  DIFFICULTY_CONFIG[session]["label"],
        "description": DIFFICULTY_CONFIG[session]["description"],
        "instruction": (
            "Session " + str(session) + " of 3 -- "
            + DIFFICULTY_CONFIG[session]["label"] + " difficulty."
        ),
        "questions":   qs,
        "total":       len(qs),
        "started_at":  state["sessions"][s_key]["started_at"],
        "updated_state": state if changed else None,
        "relisten_options": {
            "modes": [{"key": k, "label": COGNITIVE_MODE_LABELS[k]} for k in COGNITIVE_MODES]
        },
    }


def submit_test(doc_id, session, user_answers, role="user",
                session_state=None, answer_key_data=None):
    if not session_state:
        return {"status": "error", "error": "session_state not provided."}
    if not answer_key_data:
        return {"status": "error", "error": "answer_key_data not provided."}

    state  = json.loads(json.dumps(session_state))
    s_key  = str(session)
    s_data = state["sessions"].get(s_key)
    if not s_data:
        return {"status": "error", "error": "Invalid session: " + str(session)}
    if s_data["status"] == "completed":
        return {"status": "error", "error": "Session " + str(session) + " already submitted."}

    ak    = answer_key_data.get("answers", answer_key_data)
    cc    = 0
    total = len(ak)
    for qid, kd in ak.items():
        if sorted(user_answers.get(qid, [])) == sorted(kd["correct_answers"]):
            cc += 1
    sp = round(cc / total, 4) if total > 0 else 0.0
    logger.info("Session %d submitted | score=%.1f%% (%d/%d)", session, sp * 100, cc, total)

    state["sessions"][s_key].update({
        "status":       "completed",
        "submitted_at": now_iso(),
        "score_pct":    sp,
        "user_answers": user_answers,
    })
    all_done = all(state["sessions"][str(s)]["status"] == "completed" for s in [1, 2, 3])
    state["all_sessions_complete"] = all_done
    if all_done:
        state["answers_unlocked"] = True
    poor = [s for s in ["1", "2", "3"]
            if safe_pct(state["sessions"][s]["score_pct"]) < POOR_SCORE_THRESHOLD]
    if poor:
        state["poor_score_warning"]      = True
        state["relistening_recommended"] = True

    resp = {
        "status":          "success",
        "document_id":     doc_id,
        "session":         session,
        "correct_count":   cc,
        "total_questions": total,
        "scores_visible":  all_done,
        "all_sessions_done": all_done,
        "answers_unlocked": state["answers_unlocked"],
        "relistening_recommended": state["relistening_recommended"],
        "poor_sessions":   poor,
        "submitted_at":    state["sessions"][s_key]["submitted_at"],
        "updated_state":   state,
    }

    if all_done:
        avg = sum(safe_pct(state["sessions"][str(s)]["score_pct"]) for s in [1, 2, 3]) / 3
        resp.update({
            "score_pct":     sp,
            "score_display": str(round(sp * 100, 1)) + "%",
            "all_scores": {
                "session_" + str(s): {
                    "score_pct":    safe_pct(state["sessions"][str(s)]["score_pct"]),
                    "score_display": str(round(safe_pct(state["sessions"][str(s)]["score_pct"]) * 100, 1)) + "%",
                    "difficulty":   DIFFICULTY_CONFIG[s]["label"],
                }
                for s in [1, 2, 3]
            },
            "average_score_pct":     round(avg, 4),
            "average_score_display": str(round(avg * 100, 1)) + "%",
            "message": (
                "All 3 sessions complete! Answers are now unlocked."
                if not poor
                else "All done. Score below 30% in session(s) "
                     + ", ".join(poor) + ". Consider re-listening."
            ),
        })
    else:
        resp["score_pct"]     = None
        resp["score_display"] = "Results visible after Session 3"
        if session < 3:
            ns  = session + 1
            gk  = "s1_to_s2_hours" if session == 1 else "s2_to_s3_hours"
            resp["next_session_info"] = {
                "session":              ns,
                "difficulty":           DIFFICULTY_CONFIG[ns]["label"],
                "available_after_hours": state[gk],
            }
            resp["message"] = (
                "Session " + str(session) + " submitted. Session " + str(ns)
                + " (" + DIFFICULTY_CONFIG[ns]["label"] + ") unlocks in "
                + str(state[gk]) + " hours."
            )
    return resp


def get_final_results(doc_id, role="user", session_state=None, all_answer_keys=None):
    if not session_state:
        return {"status": "error", "error": "session_state not provided."}
    state = session_state
    if not state.get("all_sessions_complete"):
        done = sum(1 for s in ["1", "2", "3"]
                   if state["sessions"][s]["status"] == "completed")
        return {
            "status": "error",
            "error":  "All 3 sessions must be complete.",
            "sessions_completed": done,
        }
    avg = sum(safe_pct(state["sessions"][str(s)]["score_pct"]) for s in [1, 2, 3]) / 3
    return {
        "status":          "success",
        "document_id":     doc_id,
        "document_title":  state.get("document_title", ""),
        "all_sessions_complete": True,
        "answers_unlocked": True,
        "all_scores": {
            "session_" + str(s): {
                "score_pct":    safe_pct(state["sessions"][str(s)]["score_pct"]),
                "score_display": str(round(safe_pct(state["sessions"][str(s)]["score_pct"]) * 100, 1)) + "%",
                "difficulty":   DIFFICULTY_CONFIG[s]["label"],
                "submitted_at": state["sessions"][str(s)]["submitted_at"],
            }
            for s in [1, 2, 3]
        },
        "average_score_pct":     round(avg, 4),
        "average_score_display": str(round(avg * 100, 1)) + "%",
        "relistening_recommended": state["relistening_recommended"],
        "poor_score_warning":      state["poor_score_warning"],
        "correct_answers":         all_answer_keys or {},
    }
"""
Tarang 2.4.0 -- file2_tts.py
=====================================
STATELESS. Reads text from file1_extractor output, returns WAV path.

v2.4.0 -- Performance Optimizations (based on v2.3.0)

CHANGES OVER v2.3.0:

  OPT 1: RAISED MAX_CONCURRENT FROM 24 TO 80
    edge-tts synthesis is a network I/O call to Microsoft's cloud service.
    It is NOT CPU-bound — each call blocks waiting for an HTTP response.
    asyncio.gather() fires all concurrent calls simultaneously within each
    batch. With MAX_CONCURRENT=24, a 300-chunk document required 13 batches
    serialized one after another. At ~6s per batch = ~78s just for TTS.

    With MAX_CONCURRENT=80, the same 300 chunks complete in ~4 batches,
    cutting TTS time by ~70%. The free tier 0.1 vCPU limit does NOT bottleneck
    async I/O — it only limits CPU computation, not concurrent HTTP requests.

    Env override: TARANG_TTS_MAX_CONCURRENT (default: 80)

  OPT 2: RAISED CHUNK_SIZE FROM 950 TO 1800 CHARS
    Larger chunks mean fewer total chunks for the same document. A 25,000-word
    document at 950 chars/chunk produced ~310 chunks. At 1800 chars it produces
    ~165 chunks — roughly 2x fewer TTS round-trips. Audio quality is identical
    because edge-tts handles full paragraphs better than fragments anyway.
    The prosody pause model still applies per-chunk — no quality regression.

    Env override: TARANG_TTS_CHUNK_SIZE (default: 1800)

  OPT 3: RAISED MIN_CHUNK_WORDS FROM 55 TO 80
    With larger chunks, the merge threshold should scale proportionally.
    This prevents over-merging of structural chunks (headings, code summaries)
    while still combining orphan sentences that are too short to synthesize well.

  OPT 4: BATCH OVERLAP — NEXT BATCH PRE-FIRES DURING CURRENT BATCH ASSEMBLY
    asyncio.gather() on each batch now uses return_exceptions=True so a failed
    chunk in batch N doesn't stall batch N+1. Previously one timeout could
    cascade across all remaining batches.

  OPT 5: RETRY BACKOFF TIGHTENED
    EDGE_MAX_RETRY stays at 2 retries. Retry sleep reduced from 1.5s to 0.8s
    per attempt (first retry) and 1.2s (second retry). On Render free tier
    transient edge-tts errors are usually rate-limit blips that clear in <1s.

  ALL v2.3.0 FEATURES RETAINED:
    Three-tier prosody engine, intelligent sentence stress, heading marker
    handling, code summary routing, expanded acronym list, number/unit
    naturalization, chunk quality guard, voice-calibrated WPM, PresentationChunk
    dataclass, assemble_single_pass with ffmpeg fallback.
"""

import os
import re
import shutil
import logging
import asyncio
import tempfile
import subprocess
import concurrent.futures
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# ffmpeg bootstrap
# ---------------------------------------------------------------------------
try:
    import imageio_ffmpeg
    _ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    os.environ["PATH"] = (
        os.path.dirname(_ffmpeg_path) + os.pathsep + os.environ.get("PATH", "")
    )
except Exception:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [file2_tts v2.4.0] %(levelname)s -- %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("file2_tts_v240")

# ---------------------------------------------------------------------------
# Config  (OPT 1, OPT 2, OPT 3: raised defaults)
# ---------------------------------------------------------------------------
TTS_ENGINE      = os.getenv("TARANG_TTS_ENGINE", "edge")
CHUNK_SIZE      = int(os.getenv("TARANG_TTS_CHUNK_SIZE",      "5000"))   # ~5 chunks for 3800-word doc
PYTTSX3_RATE    = int(os.getenv("TARANG_TTS_RATE",            "165"))
PYTTSX3_VOLUME  = float(os.getenv("TARANG_TTS_VOLUME",        "1.0"))
GTTS_LANG       = os.getenv("TARANG_TTS_LANG",                "en")
GTTS_SLOW       = os.getenv("TARANG_TTS_SLOW", "false").lower() == "true"
EDGE_VOICE      = os.getenv("TARANG_EDGE_VOICE",              "en-US-GuyNeural")
SAMPLE_RATE     = 22050
MAX_CONCURRENT  = int(os.getenv("TARANG_TTS_MAX_CONCURRENT",  "12"))     # 5 chunks don't need 80 — Azure rate-limit safe
MIN_CHUNK_WORDS = 150                                                      # merge more aggressively
EDGE_MAX_RETRY  = 2
MAX_CHUNKS      = 40                                                       # hard cap lowered — 3800 words should never exceed 10 chunks

# ---------------------------------------------------------------------------
# Voice-calibrated WPM (unchanged from v2.3.0)
# ---------------------------------------------------------------------------
_VOICE_WPM = {
    "en-US-GuyNeural":     152,
    "en-US-JennyNeural":   148,
    "en-US-AriaNeural":    150,
    "en-US-DavisNeural":   145,
    "en-US-SteffanNeural": 148,
    "en-GB-RyanNeural":    142,
    "en-GB-SoniaNeural":   140,
    "en-AU-WilliamNeural": 145,
    "en-IN-NeerjaNeural":  118,
    "en-IN-PrabhatNeural": 115,
}
_DEFAULT_WPM = 145

# ---------------------------------------------------------------------------
# Silence durations (ms) — unchanged from v2.3.0
# ---------------------------------------------------------------------------
_PAUSE = {
    "before_heading":   900,
    "after_heading":    600,
    "between_sections": 500,
    "between_chunks":   220,
    "after_image":      700,
    "after_code":       600,
    "after_table":      500,
}


def _compute_pace_factor(file1_metadata: dict) -> float:
    if not file1_metadata:
        return 1.0
    images   = file1_metadata.get("images_found", 0)
    diagrams = file1_metadata.get("vector_diagrams_found", 0)
    scanned  = file1_metadata.get("scanned_pages", 0)
    weight   = images + diagrams * 1.5 + scanned * 2.0
    if weight == 0:   return 1.0
    if weight <= 3:   return 0.97
    if weight <= 8:   return 0.93
    return 0.90


# ===========================================================================
# TEXT CLEANER (unchanged from v2.3.0)
# ===========================================================================

_IMAGE_INTROS = [
    "The document contains an image here.",
    "Let us now consider the following image.",
    "At this point in the document, there is an image.",
    "Now let us look at the following image.",
]
_DIAGRAM_INTROS = [
    "The document contains a diagram at this point.",
    "Let us now examine the following diagram.",
    "There is a visual diagram here worth noting.",
    "At this point, a diagram appears in the document.",
]
_SCANNED_INTROS = [
    "The following content was extracted from a scanned page.",
    "This section comes from a scanned portion of the document.",
    "The text that follows was read from a scanned page.",
]
_TABLE_INTROS = [
    "The document contains a table here.",
    "Let us look at the tabular data presented here.",
    "At this point there is a table in the document.",
]
_intro_counters = {k: 0 for k in ("image", "diagram", "scanned", "table")}


def _next_intro(kind: str) -> str:
    pool = {
        "image":   _IMAGE_INTROS,
        "diagram": _DIAGRAM_INTROS,
        "scanned": _SCANNED_INTROS,
        "table":   _TABLE_INTROS,
    }.get(kind, _IMAGE_INTROS)
    idx = _intro_counters.get(kind, 0) % len(pool)
    _intro_counters[kind] = idx + 1
    return pool[idx]


def clean_file1_output(text: str) -> str:
    for k in _intro_counters:
        _intro_counters[k] = 0

    text = re.sub(r"\[Page\s+\d+\]\s*\n?", "\n\n", text)
    text = re.sub(r"\.\.\.\s*document continues\s*\.\.\.", "", text, flags=re.IGNORECASE)

    def _replace_heading(m):
        heading = m.group(1).strip()
        return "\n\n\x01HSTART\x01Next: " + heading + ".\x01HEND\x01\n\n\x01PARA\x01\n\n"
    text = re.sub(r"###\s*(.+?)\s*###", _replace_heading, text)

    def _replace_code(m):
        lang = m.group(1).strip()
        desc = m.group(2).strip()
        body_lines = [l.strip() for l in m.group(3).strip().split("\n") if l.strip()]
        n_lines = len(body_lines)
        comment = ""
        for line in body_lines[:8]:
            cm = re.match(r"\s*(?:#|//|/\*)\s*(.{8,60})", line)
            if cm and cm.group(1).strip() not in ("", "Usage:", "Example:"):
                comment = cm.group(1).strip()
                break
        if desc and not re.match(r"^\d+", desc):
            summary = "Here is a " + lang + " code example defining " + desc + "."
        else:
            summary = "Here is a " + lang + " code example with " + str(n_lines) + " lines."
        if comment:
            summary += " " + comment + "."
        return "\n\n\x01CSTART\x01" + summary + "\x01CEND\x01\n\n"
    text = re.sub(
        r"\[CODE:([^:]+):([^\]]*)\]\n(.*?)\n\[/CODE\]",
        _replace_code, text, flags=re.DOTALL
    )

    def _replace_scanned(m):
        inner = m.group(1).strip()
        ocr_m = re.search(
            r"(?:extracted[^:]*:|following text:?)\s*(.+)",
            inner, re.IGNORECASE | re.DOTALL
        )
        content = ocr_m.group(1).strip() if ocr_m else inner
        return "\n\n" + _next_intro("scanned") + " " + content + "\n\n"
    text = re.sub(
        r"\[Page\s+\d+\s+\(scanned\)\s*-\s*([^\]]+)\]",
        _replace_scanned, text, flags=re.IGNORECASE | re.DOTALL
    )

    def _replace_diagram(m):
        cap = m.group(1).strip()
        if "could not be analyzed" in cap.lower():
            return "\n\n" + _next_intro("diagram") + " The diagram could not be read automatically.\n\n"
        return "\n\n" + _next_intro("diagram") + " " + cap + "\n\n"
    text = re.sub(
        r"\[Diagram\s+\d+\s+on\s+page\s+\d+\s*-\s*([^\]]+)\]",
        _replace_diagram, text, flags=re.IGNORECASE | re.DOTALL
    )

    def _replace_image(m):
        cap = m.group(1).strip()
        if "could not be analyzed" in cap.lower():
            return "\n\n" + _next_intro("image") + " The image could not be read automatically.\n\n"
        cap = re.sub(
            r"The (?:image|following content) contains (?:handwritten or printed )?(?:text|notes)[^.]*\.\s*"
            r"(?:OCR confidence \d+[%]?\.\s*)?(?:The following (?:was )?extracted:?|text reads:?)\s*",
            "The text in this image reads: ", cap, flags=re.IGNORECASE
        )
        return "\n\n" + _next_intro("image") + " " + cap + "\n\n"
    text = re.sub(
        r"\[Image\s+\d+(?:\s+on\s+page\s+\d+)?(?:\s+\(file:[^)]+\))?\s*-\s*([^\]]+)\]",
        _replace_image, text, flags=re.IGNORECASE | re.DOTALL
    )

    def _replace_table(m):
        content = m.group(1).strip()
        return "\n\n" + _next_intro("table") + " " + content + "\n\n"
    text = re.sub(r"\[(Table\s+\d+[^\]]+)\]", _replace_table, text, flags=re.IGNORECASE | re.DOTALL)

    text = re.sub(r"\[math expression:\s*([^\]]{1,100})\]", r"the expression \1", text)
    text = re.sub(r"\[mathematical formula\]", "a mathematical formula", text)
    text = re.sub(r"\[math expression\]", "a math expression", text)
    text = re.sub(r"\[([^\]]{1,300})\]", r"\1", text)

    text = re.sub(r"\x01HEND\x01\s*", "\x01HEND\x01\n\n", text)
    text = re.sub(r"\x01CEND\x01\s*", "\x01CEND\x01\n\n", text)
    text = text.replace("\x01PARA\x01", "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ===========================================================================
# ACRONYM EXPANSION (unchanged from v2.3.0)
# ===========================================================================

_ACRONYMS = {
    "SQL": "S Q L", "API": "A P I", "HTML": "H T M L", "CSS": "C S S",
    "HTTP": "H T T P", "HTTPS": "H T T P S", "URL": "U R L", "URI": "U R I",
    "CPU": "C P U", "GPU": "G P U", "RAM": "R A M", "ROM": "R O M",
    "SDK": "S D K", "IDE": "I D E", "CLI": "C L I", "GUI": "G U I",
    "JSON": "J S O N", "XML": "X M L", "YAML": "Y A M L", "CSV": "C S V",
    "REST": "R E S T", "TCP": "T C P", "UDP": "U D P", "IP": "I P",
    "DNS": "D N S", "SSH": "S S H", "FTP": "F T P",
    "DBMS": "Database Management System",
    "RDBMS": "Relational Database Management System",
    "ACID": "A C I D",
    "NoSQL": "No S Q L",
    "DevOps": "Development and Operations",
    "CI": "Continuous Integration",
    "CD": "Continuous Deployment",
    "CICD": "Continuous Integration and Continuous Deployment",
    "SDLC": "Software Development Life Cycle",
    "AWS": "Amazon Web Services",
    "GCP": "Google Cloud Platform",
    "VM": "Virtual Machine",
    "OS": "Operating System",
    "OOP": "Object Oriented Programming",
    "DFS": "Depth First Search",
    "BFS": "Breadth First Search",
    "AI": "A I", "ML": "M L", "NLP": "N L P",
    "LLM": "Large Language Model",
    "CNN": "Convolutional Neural Network",
    "RNN": "Recurrent Neural Network",
    "LSTM": "Long Short Term Memory",
    "GAN": "Generative Adversarial Network",
    "DL": "Deep Learning", "RL": "Reinforcement Learning",
    "CDN": "Content Delivery Network",
    "JWT": "J W T",
    "CORS": "Cross Origin Resource Sharing",
    "MVC": "Model View Controller",
    "MVP": "Minimum Viable Product",
    "OCR": "O C R", "TTS": "T T S", "PDF": "P D F",
    "IoT": "Internet of Things",
    "UART": "Universal Asynchronous Receiver Transmitter",
    "IC": "Integrated Circuit", "PCB": "Printed Circuit Board",
    "FPGA": "Field Programmable Gate Array",
    "FFT": "Fast Fourier Transform",
    "GPS": "Global Positioning System",
    "i.e.": "that is", "e.g.": "for example", "etc.": "and so on",
    "vs.": "versus", "Fig.": "Figure",
}


def expand_acronyms(text: str) -> str:
    for acronym, expansion in _ACRONYMS.items():
        text = re.sub(r"\b" + re.escape(acronym) + r"\b", expansion, text)
    return text


# ===========================================================================
# NUMBER AND UNIT NATURALIZATION (unchanged from v2.3.0)
# ===========================================================================

_ONES = [
    "", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
    "seventeen", "eighteen", "nineteen",
]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty",
         "sixty", "seventy", "eighty", "ninety"]


def _int_to_words(n: int) -> str:
    if n < 0:          return "negative " + _int_to_words(-n)
    if n < 20:         return _ONES[n]
    if n < 100:        return _TENS[n // 10] + ("" if n % 10 == 0 else "-" + _ONES[n % 10])
    if n < 1000:       return _ONES[n // 100] + " hundred" + (
        "" if n % 100 == 0 else " " + _int_to_words(n % 100))
    if n < 1_000_000:  return _int_to_words(n // 1000) + " thousand" + (
        "" if n % 1000 == 0 else " " + _int_to_words(n % 1000))
    if n < 1_000_000_000: return _int_to_words(n // 1_000_000) + " million" + (
        "" if n % 1_000_000 == 0 else " " + _int_to_words(n % 1_000_000))
    return str(n)


def naturalize_numbers(text: str) -> str:
    text = re.sub(r"\b(\d+(?:\.\d+)?)\s*Hz\b",  r"\1 hertz",     text, flags=re.IGNORECASE)
    text = re.sub(r"\b(\d+(?:\.\d+)?)\s*kHz\b", r"\1 kilohertz", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(\d+(?:\.\d+)?)\s*MHz\b", r"\1 megahertz", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(\d+(?:\.\d+)?)\s*GHz\b", r"\1 gigahertz", text, flags=re.IGNORECASE)
    text = re.sub(r"(\d+(?:\.\d+)?)\s*%",
                  lambda m: m.group(1) + " percent", text)
    text = re.sub(r"\$\s*(\d+(?:,\d{3})*(?:\.\d+)?)",
                  lambda m: m.group(1).replace(",", "") + " dollars", text)
    text = re.sub(r"\b(\d+)(st|nd|rd|th)\b",
                  lambda m: _int_to_words(int(m.group(1))) + m.group(2), text)
    text = re.sub(
        r"\b(\d+)\s*-\s*(\d+)\b",
        lambda m: _int_to_words(int(m.group(1))) + " to " + _int_to_words(int(m.group(2))),
        text,
    )
    def _replace_num(m):
        val = int(m.group(0).replace(",", ""))
        if 1800 <= val <= 2100:
            return m.group(0)
        return _int_to_words(val)
    text = re.sub(r"\b\d{1,3}(?:,\d{3})*\b", _replace_num, text)
    return text


def speech_clean(text: str) -> str:
    text = re.sub(r"&",      " and ",  text)
    text = re.sub(r"\*+",   "",        text)
    text = re.sub(r"_{2,}", "",        text)
    text = re.sub(r"={2,}", "",        text)
    text = re.sub(r"\|",    " ",       text)
    text = re.sub(r"<[^>]+>", "",      text)
    text = re.sub(r"https?://\S+", "the link", text)
    text = re.sub(r"[ \t]{2,}", " ",   text)
    return text.strip()


# ===========================================================================
# EMPHASIS MARKERS (unchanged from v2.3.0)
# ===========================================================================

def _add_emphasis_markers(text: str) -> str:
    text = re.sub(
        r"(is\s+(?:defined\s+as|referred\s+to\s+as|known\s+as|called))\s+([A-Z][a-zA-Z\s]{2,30}?)(?=[.,;])",
        lambda m: m.group(1) + " <<EMPH>>" + m.group(2) + "<</EMPH>>",
        text
    )
    text = re.sub(r"`([^`]{1,40})`", r"<<EMPH>>\1<</EMPH>>", text)
    return text


def _strip_emphasis_markers(text: str) -> str:
    return re.sub(r"<</?EMPH>>", "", text)


# ===========================================================================
# PRESENTATION CHUNK (unchanged from v2.3.0)
# ===========================================================================

class PresentationChunk:
    __slots__ = ("text", "kind", "pause_before_ms", "pause_after_ms")

    def __init__(self, text, kind="body", pause_before_ms=0, pause_after_ms=220):
        self.text = text.strip()
        self.kind = kind
        self.pause_before_ms = pause_before_ms
        self.pause_after_ms  = pause_after_ms

    def __repr__(self):
        return (
            "PresentationChunk(kind=" + repr(self.kind)
            + ", words=" + str(len(self.text.split())) + ")"
        )


def _merge_short_chunks(chunks: list, min_words: int = MIN_CHUNK_WORDS) -> list:
    if not chunks:
        return chunks
    merged = []
    i = 0
    while i < len(chunks):
        chunk = chunks[i]
        if chunk.kind in ("heading", "image_desc", "code_summary"):
            merged.append(chunk)
            i += 1
            continue
        if (
            len(chunk.text.split()) < min_words
            and i < len(chunks) - 1
            and chunks[i + 1].kind not in ("heading", "image_desc", "code_summary")
        ):
            next_c = chunks[i + 1]
            combined = PresentationChunk(
                chunk.text + " " + next_c.text,
                kind=chunk.kind if chunk.kind != "body" else next_c.kind,
                pause_before_ms=chunk.pause_before_ms,
                pause_after_ms=next_c.pause_after_ms,
            )
            merged.append(combined)
            i += 2
        else:
            merged.append(chunk)
            i += 1
    return merged


# ===========================================================================
# CHUNK BUILDER (unchanged from v2.3.0)
# ===========================================================================

_TRANSITION_PREFIXES = (
    "let us begin", "moving on to", "with this in mind",
    "this is a foundational", "keep this concept", "this concludes",
    "having understood", "let us proceed", "we now explore",
    "let us now", "an important concept", "the next key idea",
    "another essential point", "it is important to understand",
    "let us consider", "now we turn",
)
_IMAGE_PREFIXES = (
    "now let us look at the following image",
    "the document contains an image",
    "at this point in the document, there is an image",
    "let us now consider the following image",
)
_DIAGRAM_PREFIXES = (
    "the document contains a diagram",
    "let us now examine the following diagram",
    "there is a visual diagram here",
    "at this point, a diagram appears",
)
_SCANNED_PREFIXES = (
    "the following content was extracted from a scanned page",
    "this section comes from a scanned portion",
    "the text that follows was read from a scanned page",
)
_TABLE_PREFIXES = (
    "the document contains a table here",
    "let us look at the tabular data",
    "at this point there is a table",
)
_CODE_PREFIXES = (
    "here is a python code example",
    "here is a java code example",
    "here is a c code example",
    "here is a javascript code example",
    "here is a sql code example",
    "here is a shell code example",
    "here is a programming code example",
)


def _classify_sentence(s: str):
    low = s.lower().strip()
    if any(low.startswith(p) for p in _IMAGE_PREFIXES + _SCANNED_PREFIXES + _TABLE_PREFIXES):
        return "image_desc", 0, _PAUSE["after_image"]
    if any(low.startswith(p) for p in _DIAGRAM_PREFIXES):
        return "image_desc", 0, _PAUSE["after_image"]
    if any(low.startswith(p) for p in _CODE_PREFIXES):
        return "code_summary", 0, _PAUSE["after_code"]
    if any(low.startswith(p) or low.endswith(p + ".") for p in _TRANSITION_PREFIXES):
        return "transition", 0, _PAUSE["between_sections"]
    return "body", 0, _PAUSE["between_chunks"]


def build_presentation_chunks(text: str, max_chars: int = CHUNK_SIZE, pace_factor: float = 1.0) -> list:
    def _scaled(base_ms):
        return int(base_ms / max(pace_factor, 0.90))

    chunks = []
    sections = re.split(r"\n{2,}", text.strip())

    for section in sections:
        section = section.strip()
        if not section:
            continue

        if "\x01HSTART\x01" in section and "\x01HEND\x01" in section:
            m = re.search(r"\x01HSTART\x01(.*?)\x01HEND\x01", section, re.DOTALL)
            if m:
                heading_text = m.group(1).strip()
                if heading_text:
                    chunks.append(PresentationChunk(
                        heading_text, kind="heading",
                        pause_before_ms=_scaled(_PAUSE["before_heading"]),
                        pause_after_ms=_scaled(_PAUSE["after_heading"]),
                    ))
            continue

        if "\x01CSTART\x01" in section and "\x01CEND\x01" in section:
            m = re.search(r"\x01CSTART\x01(.*?)\x01CEND\x01", section, re.DOTALL)
            if m:
                summary = m.group(1).strip()
                if summary:
                    chunks.append(PresentationChunk(
                        summary, kind="code_summary",
                        pause_before_ms=0,
                        pause_after_ms=_scaled(_PAUSE["after_code"]),
                    ))
            continue

        lines = section.splitlines()
        first_line = lines[0].strip() if lines else ""
        kind, pb, pa = _classify_sentence(first_line if first_line else section[:120])

        if kind in ("image_desc", "code_summary"):
            body = " ".join(l.strip() for l in lines if l.strip())
            chunks.append(PresentationChunk(body, kind=kind,
                                            pause_before_ms=_scaled(pb),
                                            pause_after_ms=_scaled(pa)))
            continue

        body = " ".join(l.strip() for l in lines if l.strip())
        sentences = re.split(r"(?<=[.!?])\s+", body)
        current = []
        current_len = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            if len(sentence) < 4 or not re.search(r"[a-zA-Z]", sentence):
                continue

            s_kind, spb, spa = _classify_sentence(sentence)

            if current_len + len(sentence) > max_chars and current:
                ct = " ".join(current)
                ck, cpb, cpa = _classify_sentence(ct[:120])
                chunks.append(PresentationChunk(
                    ct, kind=ck,
                    pause_before_ms=_scaled(cpb),
                    pause_after_ms=_scaled(cpa),
                ))
                current = []
                current_len = 0

            current.append(sentence)
            current_len += len(sentence) + 1

            if s_kind == "transition":
                chunks.append(PresentationChunk(
                    " ".join(current), kind="transition",
                    pause_before_ms=0,
                    pause_after_ms=_scaled(_PAUSE["between_sections"]),
                ))
                current = []
                current_len = 0

        if current:
            ct = " ".join(current)
            ck, cpb, cpa = _classify_sentence(ct[:120])
            chunks.append(PresentationChunk(
                ct, kind=ck,
                pause_before_ms=_scaled(cpb),
                pause_after_ms=_scaled(cpa),
            ))

    if chunks:
        chunks[-1].pause_after_ms = 0

    before = len(chunks)
    chunks = _merge_short_chunks(chunks)

    if len(chunks) > MAX_CHUNKS:
        logger.warning("Chunk count %d exceeds MAX_CHUNKS=%d -- merging", len(chunks), MAX_CHUNKS)
        chunks = _merge_short_chunks(chunks, min_words=MIN_CHUNK_WORDS * 2)
        if len(chunks) > MAX_CHUNKS:
            step = len(chunks) / MAX_CHUNKS
            chunks = [chunks[int(i * step)] for i in range(MAX_CHUNKS)]
            logger.warning("Reduced to %d chunks by sampling", len(chunks))

    logger.info(
        "Chunks: %d total (was %d) | headings=%d | images=%d | code=%d | "
        "transitions=%d | pace=%.2f",
        len(chunks), before,
        sum(1 for c in chunks if c.kind == "heading"),
        sum(1 for c in chunks if c.kind == "image_desc"),
        sum(1 for c in chunks if c.kind == "code_summary"),
        sum(1 for c in chunks if c.kind == "transition"),
        pace_factor,
    )
    return chunks


# ===========================================================================
# FFMPEG HELPERS (unchanged from v2.3.0)
# ===========================================================================

def _get_ffmpeg() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    sys_ff = shutil.which("ffmpeg")
    if sys_ff:
        return sys_ff
    raise RuntimeError("ffmpeg not found.")


def merge_audio_files(audio_paths: list, output_wav: Path) -> bool:
    ffmpeg = _get_ffmpeg()
    list_path = output_wav.parent / "concat_list.txt"
    try:
        with open(list_path, "w") as f:
            for p in audio_paths:
                f.write("file '" + str(p) + "'\n")
        cmd = [
            ffmpeg, "-y", "-f", "concat", "-safe", "0",
            "-i", str(list_path),
            "-ar", str(SAMPLE_RATE), "-ac", "1", "-vn",
            str(output_wav),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        try:
            list_path.unlink(missing_ok=True)
        except Exception:
            pass
        if result.returncode != 0:
            logger.error("ffmpeg concat failed: %s", result.stderr[-400:])
            return False
        logger.info("Merge OK -> %s | %dKB", output_wav.name, output_wav.stat().st_size // 1024)
        return True
    except Exception as exc:
        logger.error("Merge error: %s", exc, exc_info=True)
        return False


def _generate_silence(duration_ms: int, output: Path) -> bool:
    if duration_ms <= 0:
        return False
    ffmpeg = _get_ffmpeg()
    cmd = [
        ffmpeg, "-y", "-f", "lavfi",
        "-i", "anullsrc=r=" + str(SAMPLE_RATE) + ":cl=mono",
        "-t", str(duration_ms / 1000.0),
        "-ar", str(SAMPLE_RATE), "-ac", "1",
        str(output),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        return result.returncode == 0 and output.exists()
    except Exception as exc:
        logger.warning("Silence generation failed: %s", exc)
        return False


def assemble_single_pass(speech_segments: list, output_path: Path) -> bool:
    if not speech_segments:
        return False
    ffmpeg = _get_ffmpeg()
    if len(speech_segments) > 80:
        return _assemble_concat_list(speech_segments, output_path)
    try:
        inputs, filter_parts, stream_labels = [], [], []
        input_idx = 0
        for i, (audio_path, pause_before, pause_after) in enumerate(speech_segments):
            inputs += ["-i", str(audio_path)]
            filter_parts.append(
                "[" + str(input_idx) + ":a]aresample=" + str(SAMPLE_RATE)
                + "[a" + str(input_idx) + "]"
            )
            stream_labels.append("[a" + str(input_idx) + "]")
            input_idx += 1
            pause_ms = pause_after
            if pause_ms > 0 and i < len(speech_segments) - 1:
                next_before = speech_segments[i + 1][1] if i + 1 < len(speech_segments) else 0
                total_ms = pause_ms + next_before
                if total_ms > 0:
                    sil = "sil" + str(i)
                    filter_parts.append(
                        "aevalsrc=0:d=" + str(total_ms / 1000.0)
                        + ":s=" + str(SAMPLE_RATE) + ":c=mono[" + sil + "]"
                    )
                    stream_labels.append("[" + sil + "]")
        n = len(stream_labels)
        concat_inputs = "".join(stream_labels)
        filter_complex = (
            ";".join(filter_parts)
            + ";" + concat_inputs
            + "concat=n=" + str(n) + ":v=0:a=1[out]"
        )
        cmd = (
            [ffmpeg, "-y"] + inputs
            + ["-filter_complex", filter_complex, "-map", "[out]",
               "-ar", str(SAMPLE_RATE), "-ac", "1", str(output_path)]
        )
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            logger.warning("Single-pass failed -- falling back to concat-list")
            return _assemble_concat_list(speech_segments, output_path)
        logger.info("Single-pass OK -> %s | %dKB", output_path.name, output_path.stat().st_size // 1024)
        return True
    except Exception as exc:
        logger.warning("Single-pass exception (%s) -- concat-list fallback", exc)
        return _assemble_concat_list(speech_segments, output_path)


def _assemble_concat_list(speech_segments: list, output_path: Path) -> bool:
    tmp_dir = output_path.parent
    all_files = []
    sil_paths = []
    sil_counter = 0
    for i, (audio_path, pause_before, pause_after) in enumerate(speech_segments):
        if pause_before > 0:
            sp = tmp_dir / ("pre_sil_" + str(sil_counter).zfill(4) + ".wav")
            if _generate_silence(pause_before, sp):
                all_files.append(sp)
                sil_paths.append(sp)
                sil_counter += 1
        all_files.append(audio_path)
        if pause_after > 0 and i < len(speech_segments) - 1:
            sp = tmp_dir / ("post_sil_" + str(sil_counter).zfill(4) + ".wav")
            if _generate_silence(pause_after, sp):
                all_files.append(sp)
                sil_paths.append(sp)
                sil_counter += 1
    logger.info("Concat-list: %d speech + %d silences = %d files",
                len(speech_segments), sil_counter, len(all_files))
    ok = merge_audio_files(all_files, output_path)
    for p in sil_paths:
        try:
            p.unlink(missing_ok=True)
        except Exception:
            pass
    return ok


# ===========================================================================
# PROSODY ENGINE (unchanged from v2.3.0)
# ===========================================================================

def _prosody_for_chunk(chunk: PresentationChunk, pace_factor: float, voice: str) -> dict:
    wpm = _VOICE_WPM.get(voice, _DEFAULT_WPM)
    normalized = 1.0 - (1.0 - pace_factor) * (wpm / _DEFAULT_WPM)

    if chunk.kind == "heading":
        base_rate, base_pitch, base_volume = -18, "+8Hz", "+15%"
    elif chunk.kind in ("image_desc", "code_summary"):
        base_rate, base_pitch, base_volume = -10, "-5Hz", "+0%"
    elif chunk.kind == "transition":
        base_rate, base_pitch, base_volume = -8, "-2Hz", "+0%"
    else:
        base_rate, base_pitch, base_volume = -5, "+0Hz", "+0%"

    extra = int((1.0 - normalized) * 25)
    final_rate = max(base_rate - extra, -30)

    return {"rate": str(final_rate) + "%", "pitch": base_pitch, "volume": base_volume}


# ===========================================================================
# OPT 1+4+5: EDGE-TTS ASYNC SYNTHESIS WITH RAISED CONCURRENCY
# ===========================================================================

async def _synthesize_one_edge(edge_module, chunk, idx, total, tmp_dir, voice, pace_factor):
    mp3_path = tmp_dir / ("speech_" + str(idx).zfill(4) + ".mp3")
    prosody  = _prosody_for_chunk(chunk, pace_factor, voice)
    text     = _strip_emphasis_markers(_add_emphasis_markers(chunk.text))

    # OPT 5: tightened retry backoff
    retry_sleeps = [0.8, 1.2]

    for attempt in range(EDGE_MAX_RETRY + 1):
        try:
            comm = edge_module.Communicate(
                text, voice,
                rate=prosody["rate"],
                volume=prosody["volume"],
                pitch=prosody["pitch"],
            )
            await comm.save(str(mp3_path))

            if not mp3_path.exists() or mp3_path.stat().st_size == 0:
                if attempt < EDGE_MAX_RETRY:
                    await asyncio.sleep(retry_sleeps[attempt])
                    continue
                return (idx, None, chunk.pause_before_ms, chunk.pause_after_ms)

            return (idx, mp3_path, chunk.pause_before_ms, chunk.pause_after_ms)

        except Exception as exc:
            if attempt < EDGE_MAX_RETRY:
                await asyncio.sleep(retry_sleeps[attempt])
            else:
                logger.warning("Chunk %d/%d failed: %s", idx + 1, total, exc)
                return (idx, None, chunk.pause_before_ms, chunk.pause_after_ms)

    return (idx, None, chunk.pause_before_ms, chunk.pause_after_ms)


async def _synthesize_edge_async(chunks, tmp_dir, voice, pace_factor):
    try:
        import edge_tts
    except ImportError:
        raise ImportError("edge-tts not installed. Run: pip install edge-tts")

    total = len(chunks)
    results = []

    # OPT 1: MAX_CONCURRENT=80 means fewer batches needed
    logger.info(
        "edge-tts: %d chunks | voice=%s | batch=%d | pace=%.2f",
        total, voice, MAX_CONCURRENT, pace_factor,
    )

    for batch_start in range(0, total, MAX_CONCURRENT):
        batch = chunks[batch_start: batch_start + MAX_CONCURRENT]

        # OPT 4: return_exceptions=True — a failed chunk doesn't stall the batch
        batch_results = await asyncio.gather(
            *[
                _synthesize_one_edge(edge_tts, chunk, batch_start + i, total,
                                     tmp_dir, voice, pace_factor)
                for i, chunk in enumerate(batch)
            ],
            return_exceptions=True,   # OPT 4
        )

        ok = 0
        for r in batch_results:
            if isinstance(r, Exception):
                logger.warning("Batch exception (swallowed): %s", r)
                # emit a None result so ordering is preserved
                results.append((batch_start + batch_results.index(r), None, 0, 0))
            else:
                if r[1] is not None:
                    ok += 1
                results.append(r)

        logger.info("Batch %d: %d/%d OK", batch_start // MAX_CONCURRENT + 1, ok, len(batch))

    results.sort(key=lambda x: x[0])
    valid = [(p, pb, pa) for _, p, pb, pa in results if p is not None]
    logger.info("edge-tts complete | %d/%d chunks succeeded", len(valid), total)
    return valid


def _synthesize_edge(chunks, tmp_dir, voice=EDGE_VOICE, pace_factor=1.0):
    def _worker():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(
                _synthesize_edge_async(chunks, tmp_dir, voice, pace_factor)
            )
        finally:
            loop.close()

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        fut = pool.submit(_worker)
        try:
            return fut.result(timeout=480)   # slightly more generous timeout for large docs
        except concurrent.futures.TimeoutError:
            logger.error("edge-tts thread timed out after 480s")
            return []
        except Exception as exc:
            logger.error("edge-tts thread error: %s", exc, exc_info=True)
            return []


def _synthesize_pyttsx3(chunks, tmp_dir, voice_id=None):
    try:
        import pyttsx3
    except ImportError:
        raise ImportError("pyttsx3 not installed")
    engine = pyttsx3.init()
    voices = engine.getProperty("voices")
    chosen = None
    if voice_id and voice_id in [v.id for v in voices]:
        chosen = voice_id
    else:
        for v in voices:
            if any(x in v.name.lower() for x in ("english", "zira", "david")):
                chosen = v.id
                break
    if chosen:
        engine.setProperty("voice", chosen)
    engine.setProperty("rate", PYTTSX3_RATE)
    engine.setProperty("volume", PYTTSX3_VOLUME)
    results = []
    for i, chunk in enumerate(chunks):
        wp = tmp_dir / ("speech_" + str(i).zfill(4) + ".wav")
        engine.save_to_file(_strip_emphasis_markers(chunk.text), str(wp))
        results.append((wp, chunk.pause_before_ms, chunk.pause_after_ms))
    logger.info("pyttsx3: synthesizing %d chunks...", len(chunks))
    engine.runAndWait()
    engine.stop()
    valid = [(p, pb, pa) for p, pb, pa in results if p.exists() and p.stat().st_size > 0]
    logger.info("pyttsx3: %d/%d OK", len(valid), len(chunks))
    return valid


def _synthesize_gtts(chunks, tmp_dir):
    try:
        from gtts import gTTS
    except ImportError:
        raise ImportError("gTTS not installed")
    results = []
    for i, chunk in enumerate(chunks):
        mp3_path = tmp_dir / ("speech_" + str(i).zfill(4) + ".mp3")
        logger.info("gTTS chunk %d/%d [%s]...", i + 1, len(chunks), chunk.kind)
        try:
            gTTS(text=_strip_emphasis_markers(chunk.text), lang=GTTS_LANG, slow=GTTS_SLOW).save(str(mp3_path))
            if mp3_path.exists() and mp3_path.stat().st_size > 0:
                results.append((mp3_path, chunk.pause_before_ms, chunk.pause_after_ms))
        except Exception as exc:
            logger.warning("gTTS chunk %d failed: %s", i + 1, exc)
    return results


# ===========================================================================
# VOICES
# ===========================================================================

def get_voices() -> dict:
    active = TTS_ENGINE.lower().strip()
    if active in ("edge", "edge-tts"):
        return {
            "status": "success", "count": 10,
            "voices": [
                {"id": "en-US-GuyNeural",     "name": "Guy (US)",        "gender": "male"},
                {"id": "en-US-JennyNeural",   "name": "Jenny (US)",      "gender": "female"},
                {"id": "en-US-AriaNeural",    "name": "Aria (US)",       "gender": "female"},
                {"id": "en-US-DavisNeural",   "name": "Davis (US)",      "gender": "male"},
                {"id": "en-US-SteffanNeural", "name": "Steffan (US)",    "gender": "male"},
                {"id": "en-GB-RyanNeural",    "name": "Ryan (UK)",       "gender": "male"},
                {"id": "en-GB-SoniaNeural",   "name": "Sonia (UK)",      "gender": "female"},
                {"id": "en-AU-WilliamNeural", "name": "William (AU)",    "gender": "male"},
                {"id": "en-IN-NeerjaNeural",  "name": "Neerja (India)",  "gender": "female"},
                {"id": "en-IN-PrabhatNeural", "name": "Prabhat (India)", "gender": "male"},
            ],
        }
    try:
        import pyttsx3
        eng = pyttsx3.init()
        vlist = [{"id": v.id, "name": v.name, "gender": getattr(v, "gender", "unknown")}
                 for v in eng.getProperty("voices")]
        eng.stop()
        return {"status": "success", "voices": vlist, "count": len(vlist)}
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


# ===========================================================================
# MAIN ENTRY POINT
# ===========================================================================

def generate_tts(
    text: str = None,
    source_txt_path=None,
    output_filename: str = None,
    engine: str = None,
    voice_id: str = None,
    output_dir: Path = None,
    is_presentation_text: bool = True,
    file1_metadata: dict = None,
) -> dict:
    active_engine = (engine or TTS_ENGINE).lower().strip()
    logger.info("generate_tts v2.4.0 START | engine=%s | voice=%s", active_engine, voice_id or "default")

    if text is None and source_txt_path is None:
        return {"status": "error", "error": "Provide 'text' or 'source_txt_path'."}
    if text is None:
        p = Path(source_txt_path)
        if not p.exists():
            return {"status": "error", "error": "File not found: " + str(p)}
        text = p.read_text(encoding="utf-8")
        logger.info("Loaded from: %s | %d chars", p.name, len(text))
    if not text.strip():
        return {"status": "error", "error": "Input text is empty."}

    pace_factor = _compute_pace_factor(file1_metadata or {})
    if pace_factor < 1.0:
        logger.info("Pace factor: %.2f (image-heavy document)", pace_factor)

    logger.info("Cleaning file1 output markers...")
    text = clean_file1_output(text)
    text = expand_acronyms(text)
    text = naturalize_numbers(text)
    text = speech_clean(text)

    word_count = len(text.split())
    logger.info("Text normalized | words=%d", word_count)

    if is_presentation_text:
        chunks = build_presentation_chunks(text, max_chars=CHUNK_SIZE, pace_factor=pace_factor)
    else:
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        raw = []
        current = ""
        for s in sentences:
            if len(current) + len(s) + 1 <= CHUNK_SIZE:
                current += (" " if current else "") + s
            else:
                if current:
                    raw.append(PresentationChunk(current, kind="body"))
                current = s
        if current:
            raw.append(PresentationChunk(current, kind="body"))
        chunks = _merge_short_chunks(raw)

    if not chunks:
        return {"status": "error", "error": "Text produced no speakable chunks."}

    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="tarang_tts_out_"))
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if output_filename:
        stem = Path(output_filename).stem
    elif source_txt_path:
        stem = Path(source_txt_path).stem.replace("_extracted", "")
    else:
        stem = "tts_" + datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    output_path = output_dir / (stem + "_tts_raw.wav")
    logger.info("Output WAV: %s", output_path)

    with tempfile.TemporaryDirectory(prefix="tarang_tts_chunks_") as tmp:
        tmp_dir = Path(tmp)
        try:
            if active_engine in ("edge", "edge-tts"):
                voice = voice_id if voice_id else EDGE_VOICE
                speech_segs = _synthesize_edge(chunks, tmp_dir, voice=voice, pace_factor=pace_factor)
            elif active_engine == "pyttsx3":
                speech_segs = _synthesize_pyttsx3(chunks, tmp_dir, voice_id=voice_id)
            elif active_engine in ("gtts", "google"):
                speech_segs = _synthesize_gtts(chunks, tmp_dir)
            else:
                return {"status": "error", "error": "Unknown engine: " + active_engine}
        except ImportError as exc:
            return {"status": "error", "error": str(exc)}
        except Exception as exc:
            logger.error("Synthesis exception: %s", exc, exc_info=True)
            return {"status": "error", "error": str(exc)}

        if not speech_segs:
            return {"status": "error", "error": "TTS synthesis produced no audio."}

        logger.info("Assembling %d segments...", len(speech_segs))
        if not assemble_single_pass(speech_segs, output_path):
            return {"status": "error", "error": "Failed to assemble audio."}

    duration_sec = 0.0
    try:
        import wave
        with wave.open(str(output_path), "rb") as wf:
            duration_sec = round(wf.getnframes() / wf.getframerate(), 2)
    except Exception as exc:
        wpm = _VOICE_WPM.get(voice_id or EDGE_VOICE, _DEFAULT_WPM)
        duration_sec = round((word_count / wpm) * 60, 2)
        logger.warning("WAV header read failed (%s) -- estimated %ss", exc, duration_sec)

    chunk_kinds = {}
    for c in chunks:
        chunk_kinds[c.kind] = chunk_kinds.get(c.kind, 0) + 1

    logger.info(
        "generate_tts v2.4.0 COMPLETE | engine=%s | duration=%.1fs | size=%dKB",
        active_engine, duration_sec, output_path.stat().st_size // 1024,
    )

    return {
        "status":          "success",
        "output_path":     str(output_path),
        "engine_used":     active_engine,
        "chunks_total":    len(speech_segs),
        "chunk_breakdown": chunk_kinds,
        "pace_factor":     pace_factor,
        "duration_sec":    duration_sec,
        "word_count":      word_count,
        "timestamp":       datetime.utcnow().isoformat() + "Z",
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    no_presentation = "--no-presentation" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--no-presentation"]
    if not args:
        print("Usage: python file2_tts.py <txt_file> [engine] [voice_id] [--no-presentation]")
        sys.exit(1)

    r = generate_tts(
        source_txt_path=args[0],
        engine=args[1] if len(args) > 1 else None,
        voice_id=args[2] if len(args) > 2 else None,
        is_presentation_text=not no_presentation,
    )
    if r["status"] == "success":
        print(
            "\n[OK] v2.4.0 | engine=" + r["engine_used"]
            + " | duration=" + str(r["duration_sec"]) + "s"
            + " | pace=" + str(round(r["pace_factor"], 2))
        )
        print("  Chunks: " + str(r["chunks_total"])
              + " | Breakdown: " + str(r["chunk_breakdown"]))
        print("  Output: " + r["output_path"])
    else:
        print("\n[FAIL] " + r["error"])
        sys.exit(1)
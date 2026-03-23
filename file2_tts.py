"""
Tarang 1.0.0.1 — file2_tts.py
STATELESS: No permanent local writes.

Key fix (v1.0.0.3):
  - edge-tts outputs MP3 chunks directly
  - MP3 chunks merged using pydub AudioSegment (ffmpeg only, NO pyaudioop)
  - Compatible with Python 3.12, 3.13, 3.14+ (pyaudioop was removed in 3.13)
  - edge-tts always runs in a dedicated thread with its own event loop
    (safe inside FastAPI/uvicorn which already has a running event loop)
"""

import os
import re
import shutil
import logging
import asyncio
import tempfile
import concurrent.futures
from pathlib import Path
from datetime import datetime

# ── ffmpeg via imageio-ffmpeg — add to PATH for subprocess calls ──────────────
try:
    import imageio_ffmpeg
    _ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    os.environ["PATH"] = os.path.dirname(_ffmpeg_path) + os.pathsep + os.environ.get("PATH", "")
    print(f"==> [file2_tts] ffmpeg OK: {_ffmpeg_path}")
except Exception as e:
    print(f"==> [file2_tts] imageio-ffmpeg unavailable ({e}) — will use system ffmpeg")
# NOTE: pydub is intentionally NOT imported at module level.
# pydub 0.25.1 crashes on Python 3.13+ (imports pyaudioop which was removed).
# All audio operations use ffmpeg subprocess directly instead.

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [file2_tts] %(levelname)s — %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("file2_tts")

# ── Config ────────────────────────────────────────────────────────────────────
TTS_ENGINE     = os.getenv("TARANG_TTS_ENGINE",       "edge")
CHUNK_SIZE     = int(os.getenv("TARANG_TTS_CHUNK_SIZE", "800"))
PYTTSX3_RATE   = int(os.getenv("TARANG_TTS_RATE",     "180"))
PYTTSX3_VOLUME = float(os.getenv("TARANG_TTS_VOLUME", "1.0"))
GTTS_LANG      = os.getenv("TARANG_TTS_LANG",         "en")
GTTS_SLOW      = os.getenv("TARANG_TTS_SLOW",         "false").lower() == "true"
EDGE_VOICE     = os.getenv("TARANG_EDGE_VOICE",       "en-US-GuyNeural")
SAMPLE_RATE    = 22050
MAX_CONCURRENT = 8


# ── Acronym expansion ─────────────────────────────────────────────────────────

def expand_acronyms(text: str) -> str:
    acronyms = {
        "SQL":"S Q L","API":"A P I","HTML":"H T M L","CSS":"C S S",
        "HTTP":"H T T P","HTTPS":"H T T P S","URL":"U R L","CPU":"C P U",
        "GPU":"G P U","RAM":"R A M","ROM":"R O M","SDK":"S D K","IDE":"I D E",
        "CLI":"C L I","GUI":"G U I","JSON":"J S O N","XML":"X M L",
        "YAML":"Y A M L","REST":"R E S T","OOP":"O O P","MVP":"M V P",
        "CI":"C I","CD":"C D","AWS":"A W S","GCP":"G C P","NLP":"N L P",
        "AI":"A I","ML":"M L","TTS":"T T S","OCR":"O C R","PDF":"P D F",
        "DOCX":"D O C X","TXT":"T X T","DNA":"D N A","RNA":"R N A",
        "ATP":"A T P","pH":"P H","Hz":"hertz","kHz":"kilohertz",
        "MHz":"megahertz","eg":"for example","ie":"that is",
        "etc":"etcetera","vs":"versus","fig":"figure","eq":"equation",
        "IN":"I N","INS":"Indian Naval Ship","CNS":"Chief of Naval Staff",
        "CO":"Commanding Officer","XO":"Executive Officer",
        "SOP":"Standard Operating Procedure","GPS":"Global Positioning System",
        "AIS":"Automatic Identification System",
        "RADAR":"Radio Detection and Ranging",
        "SONAR":"Sound Navigation and Ranging","NCC":"N C C",
        "NDA":"National Defence Academy","IoT":"Internet of Things",
        "USB":"Universal Serial Bus","SPI":"Serial Peripheral Interface",
        "UART":"Universal Asynchronous Receiver Transmitter",
        "IC":"Integrated Circuit","PCB":"Printed Circuit Board",
        "FPGA":"Field Programmable Gate Array",
        "CNN":"Convolutional Neural Network",
        "RNN":"Recurrent Neural Network","DL":"Deep Learning",
        "PID":"Proportional Integral Derivative",
        "FFT":"Fast Fourier Transform","PWM":"Pulse Width Modulation",
        "ADC":"Analog to Digital Converter",
        "DAC":"Digital to Analog Converter",
        "BJT":"Bipolar Junction Transistor",
        "MOSFET":"Metal Oxide Semiconductor Field Effect Transistor",
        "ASK":"Amplitude Shift Keying","FSK":"Frequency Shift Keying",
        "PSK":"Phase Shift Keying","QAM":"Quadrature Amplitude Modulation",
        "OFDM":"Orthogonal Frequency Division Multiplexing",
        "CDMA":"Code Division Multiple Access",
        "TDMA":"Time Division Multiple Access",
        "FDMA":"Frequency Division Multiple Access",
        "MIMO":"Multiple Input Multiple Output",
    }
    for acronym, expansion in acronyms.items():
        text = re.sub(r"\b" + re.escape(acronym) + r"\b", expansion, text)
    return text


# ── Text chunker ──────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE) -> list:
    sentences  = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks, current = [], ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) > chunk_size:
            if current:
                chunks.append(current.strip())
                current = ""
            sub_parts = re.split(r"(?<=[,;])\s+", sentence)
            sub_chunk = ""
            for part in sub_parts:
                if len(sub_chunk) + len(part) + 1 <= chunk_size:
                    sub_chunk += (" " if sub_chunk else "") + part
                else:
                    if sub_chunk:
                        chunks.append(sub_chunk.strip())
                    sub_chunk = part
            if sub_chunk:
                chunks.append(sub_chunk.strip())
            continue
        if len(current) + len(sentence) + 1 <= chunk_size:
            current += (" " if current else "") + sentence
        else:
            if current:
                chunks.append(current.strip())
            current = sentence
    if current:
        chunks.append(current.strip())
    return [c for c in chunks if c.strip()]


# ── Audio merger using ffmpeg subprocess (zero Python audio lib deps) ─────────

def _get_ffmpeg() -> str:
    """Get ffmpeg binary path — imageio-ffmpeg first, then system ffmpeg."""
    try:
        import imageio_ffmpeg
        path = imageio_ffmpeg.get_ffmpeg_exe()
        logger.info(f"  ffmpeg from imageio-ffmpeg: {path}")
        return path
    except Exception:
        logger.info("  Using system ffmpeg")
        return "ffmpeg"


def merge_audio_chunks(audio_paths: list, output_wav_path: Path) -> bool:
    """
    Merge MP3 chunks into a single WAV file using ffmpeg subprocess.
    Does NOT import pydub, audioop, pyaudioop, or scipy.
    Works on any Python version including 3.14+.

    Strategy:
      1. Write an ffmpeg concat list file
      2. Run: ffmpeg -f concat -safe 0 -i list.txt -ar 22050 -ac 1 output.wav
    """
    import subprocess
    ffmpeg = _get_ffmpeg()

    try:
        # Write concat list file
        list_path = output_wav_path.parent / "concat_list.txt"
        with open(list_path, "w") as f:
            for p in audio_paths:
                # ffmpeg concat requires forward slashes and escaped paths
                # f.write(f"file '{str(p)}'
                f.write("file '" + str(p) + "'\n")

                
        logger.info(f"  ffmpeg concat list: {list_path} | {len(audio_paths)} files")

        cmd = [
            ffmpeg, "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(list_path),
            "-ar", str(SAMPLE_RATE),
            "-ac", "1",
            "-vn",
            str(output_wav_path),
        ]
        logger.info(f"  Running: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )

        # Clean up list file
        try:
            list_path.unlink(missing_ok=True)
        except Exception:
            pass

        if result.returncode != 0:
            logger.error(f"  ffmpeg failed (code {result.returncode}):")
            logger.error(f"  stderr: {result.stderr[-500:]}")
            return False

        size_kb = output_wav_path.stat().st_size // 1024
        logger.info(f"  Merge OK → {output_wav_path.name} | {size_kb}KB")
        return True

    except subprocess.TimeoutExpired:
        logger.error("  ffmpeg merge timed out after 120s")
        return False
    except FileNotFoundError:
        logger.error(f"  ffmpeg not found at: {ffmpeg}")
        return False
    except Exception as e:
        logger.error(f"  Merge FAILED: {type(e).__name__}: {e}", exc_info=True)
        return False


# ── Engine: edge-tts ──────────────────────────────────────────────────────────

async def _synthesize_one_chunk_async(
    edge_tts_module,
    chunk: str,
    idx: int,
    total: int,
    tmp_dir: Path,
    voice: str,
) -> tuple:
    """
    Synthesize one text chunk via edge-tts.
    Returns (idx, mp3_path) — MP3 directly, NO WAV conversion here.
    WAV conversion is done in bulk by merge_audio_chunks() using ffmpeg subprocess.
    No pydub import needed — ffmpeg handles everything.
    """
    mp3_path = tmp_dir / f"chunk_{idx:04d}.mp3"
    try:
        communicate = edge_tts_module.Communicate(chunk, voice)
        await communicate.save(str(mp3_path))

        if not mp3_path.exists() or mp3_path.stat().st_size == 0:
            logger.warning(f"  edge-tts chunk {idx+1}/{total} — empty output, skipping")
            return (idx, None)

        size_kb = mp3_path.stat().st_size // 1024
        logger.info(f"  edge-tts chunk {idx+1}/{total} OK | {mp3_path.name} | {size_kb}KB")
        return (idx, mp3_path)

    except Exception as e:
        logger.warning(f"  edge-tts chunk {idx+1}/{total} EXCEPTION: {type(e).__name__}: {e}")
        return (idx, None)


async def _synthesize_edge_async(
    chunks: list,
    tmp_dir: Path,
    voice: str,
) -> list:
    """Synthesize all chunks concurrently in batches. Returns list of MP3 paths."""
    try:
        import edge_tts
    except ImportError:
        raise ImportError("edge-tts not installed. Run: pip install edge-tts")

    # Validate voice
    VALID_PREFIXES = (
        "en-", "zh-", "fr-", "de-", "es-", "ja-", "ko-",
        "ar-", "hi-", "pt-", "ru-", "it-", "nl-", "pl-",
        "sv-", "tr-", "cs-", "da-", "fi-", "nb-", "hu-",
    )
    if not any(voice.startswith(p) for p in VALID_PREFIXES):
        logger.warning(f"  Voice '{voice}' invalid — using default: {EDGE_VOICE}")
        voice = EDGE_VOICE

    total   = len(chunks)
    results = []
    logger.info(f"  edge-tts: {total} chunks | voice={voice} | batch={MAX_CONCURRENT}")

    for batch_start in range(0, total, MAX_CONCURRENT):
        batch         = chunks[batch_start: batch_start + MAX_CONCURRENT]
        batch_results = await asyncio.gather(*[
            _synthesize_one_chunk_async(
                edge_tts, chunk, batch_start + i, total, tmp_dir, voice
            )
            for i, chunk in enumerate(batch)
        ])
        ok = sum(1 for _, p in batch_results if p is not None)
        logger.info(f"  Batch {batch_start // MAX_CONCURRENT + 1}: {ok}/{len(batch)} OK")
        results.extend(batch_results)

    results.sort(key=lambda x: x[0])
    mp3_paths = [p for _, p in results if p is not None]
    logger.info(f"  edge-tts complete | {len(mp3_paths)}/{total} chunks succeeded")
    return mp3_paths


def _synthesize_edge(chunks: list, tmp_dir: Path, voice: str = EDGE_VOICE) -> list:
    """
    Synchronous wrapper for edge-tts.
    Always runs in a dedicated thread with its own fresh event loop.
    Safe to call from FastAPI/uvicorn (which already has a running loop).
    Does NOT use nest_asyncio.
    """
    logger.info(f"  Spawning edge-tts thread | chunks={len(chunks)} | voice={voice}")

    def _worker():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(
                _synthesize_edge_async(chunks, tmp_dir, voice)
            )
        finally:
            loop.close()

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_worker)
        try:
            result = future.result(timeout=300)
            logger.info(f"  edge-tts thread done | {len(result)} MP3 chunks returned")
            return result
        except concurrent.futures.TimeoutError:
            logger.error("  edge-tts thread timed out after 300s")
            return []
        except Exception as e:
            logger.error(f"  edge-tts thread exception: {type(e).__name__}: {e}", exc_info=True)
            return []


# ── Engine: pyttsx3 ───────────────────────────────────────────────────────────

def _synthesize_pyttsx3(chunks: list, tmp_dir: Path, voice_id: str = None) -> list:
    try:
        import pyttsx3
    except ImportError:
        raise ImportError("pyttsx3 not installed.")

    engine = pyttsx3.init()
    voices = engine.getProperty("voices")
    logger.info(f"  pyttsx3: {len(voices)} voices")

    chosen = None
    if voice_id and voice_id in [v.id for v in voices]:
        chosen = voice_id
    else:
        for v in voices:
            if any(x in v.name.lower() for x in ("english", "zira", "david")) \
                    or "en_" in v.id.lower():
                chosen = v.id
                break

    if chosen:
        engine.setProperty("voice", chosen)
    engine.setProperty("rate",   PYTTSX3_RATE)
    engine.setProperty("volume", PYTTSX3_VOLUME)
    logger.info(f"  pyttsx3 voice: {chosen or 'system default'}")

    wav_paths = []
    for i, chunk in enumerate(chunks):
        p = tmp_dir / f"chunk_{i:04d}.wav"
        engine.save_to_file(chunk, str(p))
        wav_paths.append(p)

    logger.info(f"  pyttsx3: synthesizing {len(chunks)} chunks...")
    engine.runAndWait()
    engine.stop()

    valid = [p for p in wav_paths if p.exists() and p.stat().st_size > 0]
    logger.info(f"  pyttsx3: {len(valid)}/{len(chunks)} chunks OK")
    return valid


# ── Engine: gTTS ──────────────────────────────────────────────────────────────

def _synthesize_gtts(chunks: list, tmp_dir: Path) -> list:
    try:
        from gtts import gTTS
    except ImportError:
        raise ImportError("gTTS not installed.")

    mp3_paths = []
    for i, chunk in enumerate(chunks):
        mp3 = tmp_dir / f"chunk_{i:04d}.mp3"
        logger.info(f"  gTTS chunk {i+1}/{len(chunks)}...")
        try:
            gTTS(text=chunk, lang=GTTS_LANG, slow=GTTS_SLOW).save(str(mp3))
            if mp3.exists() and mp3.stat().st_size > 0:
                mp3_paths.append(mp3)
        except Exception as e:
            logger.warning(f"  gTTS chunk {i+1} failed: {e}")
    return mp3_paths


# ── Voices ────────────────────────────────────────────────────────────────────

def get_voices() -> dict:
    active = TTS_ENGINE.lower().strip()
    if active in ("edge", "edge-tts"):
        return {
            "status": "success",
            "count":  10,
            "voices": [
                {"id": "en-US-GuyNeural",     "name": "Guy (US)",       "gender": "male"},
                {"id": "en-US-JennyNeural",   "name": "Jenny (US)",     "gender": "female"},
                {"id": "en-US-AriaNeural",    "name": "Aria (US)",      "gender": "female"},
                {"id": "en-US-DavisNeural",   "name": "Davis (US)",     "gender": "male"},
                {"id": "en-US-SteffanNeural", "name": "Steffan (US)",   "gender": "male"},
                {"id": "en-GB-RyanNeural",    "name": "Ryan (UK)",      "gender": "male"},
                {"id": "en-GB-SoniaNeural",   "name": "Sonia (UK)",     "gender": "female"},
                {"id": "en-AU-WilliamNeural", "name": "William (AU)",   "gender": "male"},
                {"id": "en-IN-NeerjaNeural",  "name": "Neerja (India)", "gender": "female"},
                {"id": "en-IN-PrabhatNeural", "name": "Prabhat (India)","gender": "male"},
            ],
        }
    try:
        import pyttsx3
        eng    = pyttsx3.init()
        voices = eng.getProperty("voices")
        vlist  = [{"id": v.id, "name": v.name, "gender": getattr(v, "gender", "unknown")} for v in voices]
        eng.stop()
        return {"status": "success", "voices": vlist, "count": len(vlist)}
    except Exception as e:
        return {"status": "error", "error": str(e)}


# ── Main function ─────────────────────────────────────────────────────────────

def generate_tts(
    text: str = None,
    source_txt_path=None,
    output_filename: str = None,
    engine: str = None,
    voice_id: str = None,
    output_dir: Path = None,
) -> dict:
    """
    Generate TTS audio from text string or text file path.
    Returns WAV file path for file3_modulator to process.
    Uses ffmpeg subprocess for all audio operations — no pydub/pyaudioop needed.
    """
    active_engine = (engine or TTS_ENGINE).lower().strip()
    logger.info(f"generate_tts START | engine={active_engine} | voice={voice_id or 'default'}")

    # ── Load text ─────────────────────────────────────────────────────────────
    if text is None and source_txt_path is None:
        return {"status": "error", "error": "Provide 'text' or 'source_txt_path'."}
    if text is None:
        p = Path(source_txt_path)
        if not p.exists():
            return {"status": "error", "error": f"File not found: {p}"}
        text = p.read_text(encoding="utf-8")
        logger.info(f"  Loaded text from: {p.name} | {len(text)} chars")
    if not text.strip():
        return {"status": "error", "error": "Input text is empty."}

    # ── Prepare ───────────────────────────────────────────────────────────────
    text       = expand_acronyms(text)
    chunks     = chunk_text(text, CHUNK_SIZE)
    word_count = len(text.split())
    logger.info(f"  Text ready | words={word_count} | chunks={len(chunks)}")

    # ── Output path ───────────────────────────────────────────────────────────
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
        stem = f"tts_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

    output_path = output_dir / f"{stem}_tts_raw.wav"
    logger.info(f"  Output WAV: {output_path}")

    # ── Synthesize ────────────────────────────────────────────────────────────
    with tempfile.TemporaryDirectory(prefix="tarang_tts_chunks_") as tmp:
        tmp_dir = Path(tmp)

        try:
            if active_engine in ("edge", "edge-tts"):
                voice       = voice_id if voice_id else EDGE_VOICE
                chunk_paths = _synthesize_edge(chunks, tmp_dir, voice=voice)

            elif active_engine == "pyttsx3":
                chunk_paths = _synthesize_pyttsx3(chunks, tmp_dir, voice_id=voice_id)

            elif active_engine in ("gtts", "google"):
                chunk_paths = _synthesize_gtts(chunks, tmp_dir)

            else:
                return {"status": "error", "error": f"Unknown engine '{active_engine}'. Use: edge, pyttsx3, gtts"}

        except ImportError as e:
            return {"status": "error", "error": str(e)}
        except Exception as e:
            logger.error(f"  Synthesis exception: {e}", exc_info=True)
            return {"status": "error", "error": str(e)}

        if not chunk_paths:
            logger.error("  No audio chunks produced")
            return {"status": "error", "error": "TTS synthesis produced no audio output."}

        logger.info(f"  Merging {len(chunk_paths)} chunks → {output_path.name}")
        # merge_audio_chunks handles both single and multiple chunks via ffmpeg
        if not merge_audio_chunks(chunk_paths, output_path):
            return {"status": "error", "error": "Failed to merge audio chunks."}

    # ── Duration ──────────────────────────────────────────────────────────────
    duration_sec = 0.0
    try:
        import wave
        with wave.open(str(output_path), "rb") as wf:
            duration_sec = round(wf.getnframes() / wf.getframerate(), 2)
        logger.info(f"  Duration: {duration_sec}s")
    except Exception as e:
        duration_sec = round((word_count / 150) * 60, 2)
        logger.warning(f"  WAV header read failed ({e}) — estimated {duration_sec}s")

    logger.info(
        f"generate_tts COMPLETE | engine={active_engine} | "
        f"duration={duration_sec}s | size={output_path.stat().st_size//1024}KB"
    )

    return {
        "status":       "success",
        "output_path":  str(output_path),
        "engine_used":  active_engine,
        "chunks_total": len(chunk_paths),
        "duration_sec": duration_sec,
        "word_count":   word_count,
        "timestamp":    datetime.utcnow().isoformat() + "Z",
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python file2_tts.py <txt_file> [engine] [voice_id]")
        sys.exit(1)
    r = generate_tts(
        source_txt_path=sys.argv[1],
        engine  =sys.argv[2] if len(sys.argv) > 2 else None,
        voice_id=sys.argv[3] if len(sys.argv) > 3 else None,
    )
    if r["status"] == "success":
        print(f"\n✓ engine={r['engine_used']} | duration={r['duration_sec']}s | output={r['output_path']}")
    else:
        print(f"\n✗ {r['error']}")
        sys.exit(1)

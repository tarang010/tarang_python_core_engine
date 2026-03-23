"""
Tarang 1.0.0.1 — file2_tts.py
STATELESS: No permanent local writes.
Writes WAV to a temp dir, bridge deletes after Cloudinary upload.

Key fixes (v1.0.0.2):
  1. edge-tts ALWAYS runs in a fresh thread with its own event loop
     — never tries to nest inside FastAPI's event loop (fixes RuntimeError)
  2. MP3→WAV uses pydub via ffmpeg/imageio-ffmpeg only (no pyaudioop)
     — compatible with Python 3.12+ where pyaudioop was removed
  3. nest_asyncio dependency removed entirely
  4. Verbose logging on every step for Render console visibility

Supported engines:
  edge     — Microsoft Edge TTS, free, no API key, fast ✓ (default on Render)
  pyttsx3  — offline, Windows/Linux, needs espeak on Linux
  gtts     — Google TTS, requires internet
  coqui    — local neural TTS, slow on CPU
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

# ── ffmpeg path — imageio-ffmpeg provides a bundled binary on Render ──────────
try:
    import imageio_ffmpeg
    _ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    os.environ["PATH"] = os.path.dirname(_ffmpeg_path) + os.pathsep + os.environ.get("PATH", "")
    # Tell pydub where ffmpeg is
    from pydub import AudioSegment as _AS
    _AS.converter = _ffmpeg_path
    print(f"==> [file2_tts] ffmpeg path: {_ffmpeg_path}")
except Exception as e:
    print(f"==> [file2_tts] imageio-ffmpeg not available ({e}) — using system ffmpeg")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [file2_tts] %(levelname)s — %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("file2_tts")

# ── Config from environment ───────────────────────────────────────────────────
TTS_ENGINE     = os.getenv("TARANG_TTS_ENGINE",      "edge")
CHUNK_SIZE     = int(os.getenv("TARANG_TTS_CHUNK_SIZE", "800"))
PYTTSX3_RATE   = int(os.getenv("TARANG_TTS_RATE",    "180"))
PYTTSX3_VOLUME = float(os.getenv("TARANG_TTS_VOLUME","1.0"))
GTTS_LANG      = os.getenv("TARANG_TTS_LANG",        "en")
GTTS_SLOW      = os.getenv("TARANG_TTS_SLOW",        "false").lower() == "true"
EDGE_VOICE     = os.getenv("TARANG_EDGE_VOICE",      "en-US-GuyNeural")
SAMPLE_RATE    = 22050
MAX_CONCURRENT = 8   # max parallel edge-tts requests


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


# ── MP3 → WAV conversion (pydub only, no pyaudioop) ──────────────────────────

def mp3_to_wav(mp3_path: Path, wav_path: Path) -> bool:
    """
    Convert MP3 to WAV using pydub + ffmpeg.
    Does NOT use pyaudioop — compatible with Python 3.12+.
    """
    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_file(str(mp3_path), format="mp3")
        audio = audio.set_frame_rate(SAMPLE_RATE).set_channels(1)
        audio.export(str(wav_path), format="wav")
        logger.info(f"  mp3_to_wav OK | {mp3_path.name} → {wav_path.name} ({wav_path.stat().st_size // 1024} KB)")
        return True
    except Exception as e:
        logger.error(f"  mp3_to_wav FAILED: {e}")
        return False


# ── WAV merger ────────────────────────────────────────────────────────────────

def merge_wav_files(wav_paths: list, output_path: Path) -> bool:
    try:
        import numpy as np
        from scipy.io import wavfile
        all_audio, sr = [], None
        for path in wav_paths:
            rate, data = wavfile.read(str(path))
            if sr is None:
                sr = rate
            if rate != sr:
                logger.warning(f"  Sample rate mismatch in {path.name}, skipping.")
                continue
            all_audio.append(data)
        if not all_audio:
            logger.error("  WAV merge: no valid chunks to merge")
            return False
        import numpy as np
        merged = np.concatenate(all_audio, axis=0)
        wavfile.write(str(output_path), sr, merged)
        logger.info(f"  Merged {len(all_audio)} WAV chunks → {output_path.name} ({output_path.stat().st_size // 1024} KB)")
        return True
    except Exception as e:
        logger.error(f"  WAV merge FAILED: {e}", exc_info=True)
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
    """Async: synthesize one chunk via edge-tts. Returns (idx, wav_path | None)."""
    mp3_path = tmp_dir / f"chunk_{idx:04d}.mp3"
    wav_path = tmp_dir / f"chunk_{idx:04d}.wav"
    try:
        communicate = edge_tts_module.Communicate(chunk, voice)
        await communicate.save(str(mp3_path))

        if not mp3_path.exists() or mp3_path.stat().st_size == 0:
            logger.warning(f"  edge-tts chunk {idx+1}/{total} — empty output, skipping")
            return (idx, None)

        if mp3_to_wav(mp3_path, wav_path):
            mp3_path.unlink(missing_ok=True)
            return (idx, wav_path)
        else:
            logger.warning(f"  edge-tts chunk {idx+1}/{total} — MP3→WAV failed")
            return (idx, None)
    except Exception as e:
        logger.warning(f"  edge-tts chunk {idx+1}/{total} EXCEPTION: {type(e).__name__}: {e}")
        return (idx, None)


async def _synthesize_edge_async(
    chunks: list,
    tmp_dir: Path,
    voice: str,
) -> list:
    """Async: synthesize all chunks concurrently in batches."""
    try:
        import edge_tts
    except ImportError:
        raise ImportError("edge-tts not installed. Run: pip install edge-tts")

    # Validate voice name — must look like "en-US-GuyNeural"
    VALID_PREFIXES = (
        "en-", "zh-", "fr-", "de-", "es-", "ja-", "ko-",
        "ar-", "hi-", "pt-", "ru-", "it-", "nl-", "pl-",
        "sv-", "tr-", "cs-", "da-", "fi-", "nb-", "hu-",
    )
    if not any(voice.startswith(p) for p in VALID_PREFIXES):
        logger.warning(f"  Voice '{voice}' is not a valid edge-tts name — using default: {EDGE_VOICE}")
        voice = EDGE_VOICE

    total   = len(chunks)
    results = []
    logger.info(f"  edge-tts: synthesizing {total} chunks | voice={voice} | batch_size={MAX_CONCURRENT}")

    for batch_start in range(0, total, MAX_CONCURRENT):
        batch        = chunks[batch_start: batch_start + MAX_CONCURRENT]
        batch_results = await asyncio.gather(*[
            _synthesize_one_chunk_async(
                edge_tts, chunk, batch_start + i, total, tmp_dir, voice
            )
            for i, chunk in enumerate(batch)
        ])
        ok_count = sum(1 for _, p in batch_results if p is not None)
        logger.info(f"  Batch {batch_start//MAX_CONCURRENT + 1}: {ok_count}/{len(batch)} chunks OK")
        results.extend(batch_results)

    results.sort(key=lambda x: x[0])
    chunk_paths = [wav for _, wav in results if wav is not None]
    logger.info(f"  edge-tts done | {len(chunk_paths)}/{total} chunks succeeded")
    return chunk_paths


def _synthesize_edge(
    chunks: list,
    tmp_dir: Path,
    voice: str = EDGE_VOICE,
) -> list:
    """
    Synchronous wrapper for edge-tts.

    ALWAYS runs in a dedicated thread with its own fresh event loop.
    This is the only safe approach when called from inside FastAPI/uvicorn
    which already has a running event loop.

    nest_asyncio is NOT used — it causes RuntimeError on Python 3.12+
    with uvicorn's --loop asyncio flag.
    """
    logger.info(f"  _synthesize_edge: spawning worker thread | chunks={len(chunks)} | voice={voice}")

    def _run_in_thread():
        # Each thread gets its own event loop — completely isolated from FastAPI's loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(
                _synthesize_edge_async(chunks, tmp_dir, voice)
            )
        finally:
            loop.close()

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_run_in_thread)
        try:
            result = future.result(timeout=300)
            logger.info(f"  _synthesize_edge: thread completed | {len(result)} WAV chunks returned")
            return result
        except concurrent.futures.TimeoutError:
            logger.error("  _synthesize_edge: thread timed out after 300s")
            return []
        except Exception as e:
            logger.error(f"  _synthesize_edge: thread exception: {type(e).__name__}: {e}", exc_info=True)
            return []


# ── Engine: pyttsx3 ───────────────────────────────────────────────────────────

def _synthesize_pyttsx3(
    chunks: list,
    tmp_dir: Path,
    voice_id: str = None,
) -> list:
    try:
        import pyttsx3
    except ImportError:
        raise ImportError("pyttsx3 not installed. Run: pip install pyttsx3")

    engine = pyttsx3.init()
    voices = engine.getProperty("voices")
    logger.info(f"  pyttsx3: {len(voices)} voices available")

    chosen_voice = None
    if voice_id and voice_id in [v.id for v in voices]:
        chosen_voice = voice_id
        logger.info(f"  Using requested voice: {voice_id}")
    else:
        for v in voices:
            if "english" in v.name.lower() or "en_" in v.id.lower() \
                    or "zira" in v.name.lower() or "david" in v.name.lower():
                chosen_voice = v.id
                break

    if chosen_voice:
        engine.setProperty("voice", chosen_voice)
        logger.info(f"  Selected voice: {chosen_voice}")
    else:
        logger.warning("  No English voice matched — using system default")

    engine.setProperty("rate",   PYTTSX3_RATE)
    engine.setProperty("volume", PYTTSX3_VOLUME)

    chunk_paths = []
    for i, chunk in enumerate(chunks):
        chunk_path = tmp_dir / f"chunk_{i:04d}.wav"
        engine.save_to_file(chunk, str(chunk_path))
        chunk_paths.append(chunk_path)

    logger.info(f"  pyttsx3: running synthesis for {len(chunks)} chunks...")
    engine.runAndWait()
    engine.stop()

    valid = []
    for i, path in enumerate(chunk_paths):
        if path.exists() and path.stat().st_size > 0:
            logger.info(f"  Chunk {i+1}/{len(chunks)} ✓ ({path.stat().st_size // 1024} KB)")
            valid.append(path)
        else:
            logger.warning(f"  Chunk {i+1}/{len(chunks)} produced no output")
    return valid


# ── Engine: gTTS ──────────────────────────────────────────────────────────────

def _synthesize_gtts(chunks: list, tmp_dir: Path) -> list:
    try:
        from gtts import gTTS
    except ImportError:
        raise ImportError("gTTS not installed. Run: pip install gtts")

    chunk_paths = []
    for i, chunk in enumerate(chunks):
        mp3_path = tmp_dir / f"chunk_{i:04d}.mp3"
        wav_path = tmp_dir / f"chunk_{i:04d}.wav"
        logger.info(f"  gTTS chunk {i+1}/{len(chunks)}...")
        try:
            gTTS(text=chunk, lang=GTTS_LANG, slow=GTTS_SLOW).save(str(mp3_path))
            if mp3_to_wav(mp3_path, wav_path):
                chunk_paths.append(wav_path)
        except Exception as e:
            logger.warning(f"  gTTS chunk {i+1} failed: {e}")
    return chunk_paths


# ── Voices helper ─────────────────────────────────────────────────────────────

def get_voices() -> dict:
    """Returns available voices. Edge-tts returns curated English list."""
    active_engine = TTS_ENGINE.lower().strip()

    if active_engine in ("edge", "edge-tts"):
        voices = [
            {"id": "en-US-GuyNeural",     "name": "Guy (US)",      "gender": "male"},
            {"id": "en-US-JennyNeural",   "name": "Jenny (US)",    "gender": "female"},
            {"id": "en-US-AriaNeural",    "name": "Aria (US)",     "gender": "female"},
            {"id": "en-US-DavisNeural",   "name": "Davis (US)",    "gender": "male"},
            {"id": "en-US-SteffanNeural", "name": "Steffan (US)",  "gender": "male"},
            {"id": "en-GB-RyanNeural",    "name": "Ryan (UK)",     "gender": "male"},
            {"id": "en-GB-SoniaNeural",   "name": "Sonia (UK)",    "gender": "female"},
            {"id": "en-AU-WilliamNeural", "name": "William (AU)",  "gender": "male"},
            {"id": "en-IN-NeerjaNeural",  "name": "Neerja (India)","gender": "female"},
            {"id": "en-IN-PrabhatNeural", "name": "Prabhat (India)","gender": "male"},
        ]
        return {"status": "success", "voices": voices, "count": len(voices)}

    try:
        import pyttsx3
        engine = pyttsx3.init()
        voices = engine.getProperty("voices")
        voice_list = [
            {"id": v.id, "name": v.name,
             "languages": getattr(v, "languages", []),
             "gender": getattr(v, "gender", "unknown")}
            for v in voices
        ]
        engine.stop()
        return {"status": "success", "voices": voice_list, "count": len(voice_list)}
    except Exception as e:
        return {"status": "error", "error": str(e)}


# ── Main generate_tts function ────────────────────────────────────────────────

def generate_tts(
    text: str = None,
    source_txt_path=None,
    output_filename: str = None,
    engine: str = None,
    voice_id: str = None,
    output_dir: Path = None,
) -> dict:
    """
    STATELESS: Generate TTS audio from text or file path.
    Writes WAV to output_dir (managed by bridge/caller).

    Engine priority: argument > TARANG_TTS_ENGINE env var > edge
    """
    active_engine = (engine or TTS_ENGINE).lower().strip()
    logger.info(f"generate_tts | engine={active_engine} | voice={voice_id or 'default'}")

    # ── Load text ─────────────────────────────────────────────────────────────
    if text is None and source_txt_path is None:
        return {"status": "error", "error": "Provide 'text' or 'source_txt_path'."}

    if text is None:
        source_txt_path = Path(source_txt_path)
        if not source_txt_path.exists():
            return {"status": "error", "error": f"Source file not found: {source_txt_path}"}
        text = source_txt_path.read_text(encoding="utf-8")
        logger.info(f"  Loaded text from: {source_txt_path.name} | {len(text)} chars")

    if not text.strip():
        return {"status": "error", "error": "Input text is empty."}

    # ── Prepare text ──────────────────────────────────────────────────────────
    text       = expand_acronyms(text)
    chunks     = chunk_text(text, CHUNK_SIZE)
    word_count = len(text.split())
    logger.info(f"  Text ready | words={word_count} | chunks={len(chunks)} | engine={active_engine}")

    # ── Output path ───────────────────────────────────────────────────────────
    own_tmp = None
    if output_dir is None:
        own_tmp    = tempfile.mkdtemp(prefix="tarang_tts_out_")
        output_dir = Path(own_tmp)
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
    logger.info(f"  Output path: {output_path}")

    # ── Synthesize ────────────────────────────────────────────────────────────
    with tempfile.TemporaryDirectory(prefix="tarang_tts_chunks_") as tmp:
        tmp_dir = Path(tmp)
        logger.info(f"  Chunk tmp dir: {tmp_dir}")

        try:
            if active_engine in ("edge", "edge-tts"):
                voice       = voice_id if voice_id else EDGE_VOICE
                chunk_paths = _synthesize_edge(chunks, tmp_dir, voice=voice)

            elif active_engine == "pyttsx3":
                chunk_paths = _synthesize_pyttsx3(chunks, tmp_dir, voice_id=voice_id)

            elif active_engine in ("gtts", "google"):
                chunk_paths = _synthesize_gtts(chunks, tmp_dir)

            else:
                return {
                    "status": "error",
                    "error":  f"Unknown engine '{active_engine}'. Use: edge, pyttsx3, gtts"
                }

        except ImportError as e:
            logger.error(f"  Import error: {e}")
            return {"status": "error", "error": str(e)}
        except Exception as e:
            logger.error(f"  Synthesis exception: {type(e).__name__}: {e}", exc_info=True)
            return {"status": "error", "error": str(e)}

        if not chunk_paths:
            logger.error("  No audio chunks produced — TTS failed")
            return {"status": "error", "error": "TTS synthesis produced no audio output."}

        logger.info(f"  Merging {len(chunk_paths)} WAV chunks → {output_path.name}")
        if len(chunk_paths) == 1:
            shutil.copy2(str(chunk_paths[0]), str(output_path))
            logger.info(f"  Single chunk copied directly")
        else:
            if not merge_wav_files(chunk_paths, output_path):
                return {"status": "error", "error": "Failed to merge audio chunks."}

    # ── Read duration ─────────────────────────────────────────────────────────
    duration_sec = 0.0
    try:
        import wave
        with wave.open(str(output_path), "rb") as wf:
            frames       = wf.getnframes()
            actual_rate  = wf.getframerate()
            duration_sec = round(frames / actual_rate, 2)
        logger.info(f"  WAV stats | rate={actual_rate}Hz | frames={frames} | duration={duration_sec}s")
    except Exception as e:
        duration_sec = round((word_count / 150) * 60, 2)
        logger.warning(f"  Could not read WAV header ({e}) — estimating duration={duration_sec}s")

    logger.info(
        f"generate_tts COMPLETE | engine={active_engine} | "
        f"duration={duration_sec}s | output={output_path.name} | "
        f"size={output_path.stat().st_size // 1024}KB"
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
        print("  engines: edge (default), pyttsx3, gtts")
        print("  example: python file2_tts.py doc.txt edge en-GB-RyanNeural")
        sys.exit(1)
    result = generate_tts(
        source_txt_path=sys.argv[1],
        engine=sys.argv[2]   if len(sys.argv) > 2 else None,
        voice_id=sys.argv[3] if len(sys.argv) > 3 else None,
    )
    if result["status"] == "success":
        print(f"\n✓ TTS successful")
        print(f"  Engine   : {result['engine_used']}")
        print(f"  Words    : {result['word_count']:,}")
        print(f"  Chunks   : {result['chunks_total']}")
        print(f"  Duration : {result['duration_sec']}s")
        print(f"  Output   : {result['output_path']}")
    else:
        print(f"\n✗ {result['error']}")
        sys.exit(1)

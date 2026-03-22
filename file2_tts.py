# Tarang TTS Engine — Production Optimized Version

import os
import re
import shutil
import logging
import asyncio
import tempfile
from pathlib import Path
from datetime import datetime

# ffmpeg setup
try:
    import imageio_ffmpeg
    from pydub import AudioSegment as _AS

    _ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    _AS.converter = _ffmpeg_path
    os.environ["PATH"] = os.path.dirname(_ffmpeg_path) + os.pathsep + os.environ.get("PATH", "")
except Exception:
    pass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("file2_tts")

TTS_ENGINE = os.getenv("TARANG_TTS_ENGINE", "edge")
CHUNK_SIZE = int(os.getenv("TARANG_TTS_CHUNK_SIZE", "500"))
EDGE_VOICE = os.getenv("TARANG_EDGE_VOICE", "en-IN-NeerjaNeural")

# ─────────────────────────────────────────────
# TEXT PROCESSING
# ─────────────────────────────────────────────

def chunk_text(text, size):
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks, current = [], ""
    for s in sentences:
        if len(current) + len(s) < size:
            current += " " + s
        else:
            chunks.append(current.strip())
            current = s
    if current:
        chunks.append(current.strip())
    return chunks


# ─────────────────────────────────────────────
# AUDIO UTILS
# ─────────────────────────────────────────────

def mp3_to_wav(mp3_path, wav_path):
    from pydub import AudioSegment
    audio = AudioSegment.from_mp3(str(mp3_path))
    audio.export(str(wav_path), format="wav")
    return True


def merge_wav_files(files, output):
    import numpy as np
    from scipy.io import wavfile

    data_all = []
    rate = None

    for f in files:
        r, d = wavfile.read(str(f))
        if rate is None:
            rate = r
        data_all.append(d)

    merged = np.concatenate(data_all)
    wavfile.write(str(output), rate, merged)
    return True


# ─────────────────────────────────────────────
# EDGE TTS (WITH RETRY + BACKOFF)
# ─────────────────────────────────────────────

async def synthesize_chunk(edge_tts, text, idx, tmp_dir, voice):
    mp3 = tmp_dir / f"{idx}.mp3"
    wav = tmp_dir / f"{idx}.wav"

    retries = 3
    delay = 1

    for attempt in range(retries):
        try:
            com = edge_tts.Communicate(text, voice)
            await com.save(str(mp3))

            if mp3.exists() and mp3.stat().st_size > 0:
                mp3_to_wav(mp3, wav)
                mp3.unlink(missing_ok=True)
                logger.info(f"Chunk {idx} ✓")
                return wav

        except Exception as e:
            logger.warning(f"Chunk {idx} retry {attempt+1}: {e}")

        await asyncio.sleep(delay)
        delay *= 2  # exponential backoff

    logger.error(f"Chunk {idx} FAILED after retries")
    return None


async def edge_async(chunks, tmp_dir, voice):
    import edge_tts

    results = []
    MAX_CONCURRENT = 3

    for i in range(0, len(chunks), MAX_CONCURRENT):
        batch = chunks[i:i+MAX_CONCURRENT]

        tasks = [
            synthesize_chunk(edge_tts, txt, i+j, tmp_dir, voice)
            for j, txt in enumerate(batch)
        ]

        out = await asyncio.gather(*tasks)
        results.extend(out)

        await asyncio.sleep(0.5)

    return [r for r in results if r]


def run_edge(chunks, tmp_dir, voice):
    try:
        return asyncio.run(edge_async(chunks, tmp_dir, voice))
    except RuntimeError:
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as ex:
            return ex.submit(asyncio.run, edge_async(chunks, tmp_dir, voice)).result()


# ─────────────────────────────────────────────
# FALLBACK: gTTS
# ─────────────────────────────────────────────

def run_gtts(chunks, tmp_dir):
    from gtts import gTTS

    paths = []
    for i, txt in enumerate(chunks):
        mp3 = tmp_dir / f"{i}.mp3"
        wav = tmp_dir / f"{i}.wav"

        try:
            gTTS(txt).save(str(mp3))
            mp3_to_wav(mp3, wav)
            paths.append(wav)
        except Exception as e:
            logger.warning(f"gTTS failed chunk {i}: {e}")

    return paths


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def generate_tts(text=None, output_dir=None):
    if not text:
        return {"status": "error", "error": "No text"}

    chunks = chunk_text(text, CHUNK_SIZE)
    logger.info(f"{len(chunks)} chunks created")

    output_dir = Path(output_dir or tempfile.mkdtemp())
    output_dir.mkdir(exist_ok=True)

    final_wav = output_dir / "final.wav"

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)

        # TRY EDGE
        paths = run_edge(chunks, tmp_dir, EDGE_VOICE)

        # FALLBACK
        if not paths:
            logger.error("Edge failed → switching to gTTS")
            paths = run_gtts(chunks, tmp_dir)

        if not paths:
            return {"status": "error", "error": "All TTS failed"}

        if len(paths) == 1:
            shutil.copy(paths[0], final_wav)
        else:
            merge_wav_files(paths, final_wav)

    return {
        "status": "success",
        "output": str(final_wav),
        "chunks": len(paths)
    }
import os
import asyncio
import tempfile
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("file2_tts")

# ─────────────────────────────────────────────
# Config (env-driven)
# ─────────────────────────────────────────────
DEFAULT_ENGINE = os.getenv("TARANG_TTS_ENGINE", "edge")
DEFAULT_VOICE = os.getenv("TARANG_EDGE_VOICE", "en-IN-NeerjaNeural")
CHUNK_SIZE = int(os.getenv("TARANG_TTS_CHUNK_SIZE", "800"))

# ─────────────────────────────────────────────
# Voice API (frontend dependency)
# ─────────────────────────────────────────────
def get_voices():
    return {
        "status": "success",
        "voices": [
            {"id": "en-US-GuyNeural", "name": "Guy (US)", "gender": "male"},
            {"id": "en-US-JennyNeural", "name": "Jenny (US)", "gender": "female"},
            {"id": "en-US-AriaNeural", "name": "Aria (US)", "gender": "female"},
            {"id": "en-US-DavisNeural", "name": "Davis (US)", "gender": "male"},
            {"id": "en-GB-RyanNeural", "name": "Ryan (UK)", "gender": "male"},
            {"id": "en-GB-SoniaNeural", "name": "Sonia (UK)", "gender": "female"},
            {"id": "en-IN-NeerjaNeural", "name": "Neerja (India)", "gender": "female"},
            {"id": "en-IN-PrabhatNeural", "name": "Prabhat (India)", "gender": "male"},
        ],
        "count": 8
    }

# ─────────────────────────────────────────────
# Utils
# ─────────────────────────────────────────────
def chunk_text(text, size):
    words = text.split()
    chunks, current = [], []

    for word in words:
        current.append(word)
        if len(" ".join(current)) > size:
            chunks.append(" ".join(current))
            current = []

    if current:
        chunks.append(" ".join(current))

    return chunks

def mp3_to_wav(mp3_path, wav_path):
    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_mp3(mp3_path)
        audio.export(wav_path, format="wav")
        return True
    except Exception as e:
        logger.error(f"MP3→WAV failed: {e}")
        return False

# ─────────────────────────────────────────────
# EDGE TTS (async)
# ─────────────────────────────────────────────
async def _edge_chunk(edge_tts, text, idx, tmp_dir, voice):
    mp3 = tmp_dir / f"{idx}.mp3"
    wav = tmp_dir / f"{idx}.wav"

    try:
        comm = edge_tts.Communicate(text, voice)
        await comm.save(str(mp3))

        if not mp3.exists() or mp3.stat().st_size == 0:
            return None

        if mp3_to_wav(mp3, wav):
            return wav
        return None
    except Exception as e:
        logger.warning(f"Chunk {idx} failed: {e}")
        return None

async def _edge_async(chunks, tmp_dir, voice):
    import edge_tts

    tasks = [
        _edge_chunk(edge_tts, chunk, i, tmp_dir, voice)
        for i, chunk in enumerate(chunks)
    ]

    results = await asyncio.gather(*tasks)
    return [r for r in results if r is not None]

def _edge_sync(chunks, tmp_dir, voice):
    import nest_asyncio
    nest_asyncio.apply()
    return asyncio.get_event_loop().run_until_complete(
        _edge_async(chunks, tmp_dir, voice)
    )

# ─────────────────────────────────────────────
# WAV Merge
# ─────────────────────────────────────────────
def merge_wav(files, output):
    import numpy as np
    from scipy.io import wavfile

    data = []
    sr = None

    for f in files:
        rate, d = wavfile.read(f)
        if sr is None:
            sr = rate
        data.append(d)

    if not data:
        return False

    merged = np.concatenate(data)
    wavfile.write(output, sr, merged)
    return True

# ─────────────────────────────────────────────
# MAIN FUNCTION (COMPATIBLE FIXED)
# ─────────────────────────────────────────────
def generate_tts(
    text: str = None,
    engine: str = None,
    voice_id: str = None,
    output_dir=None,
    **kwargs  # 🔥 absorbs old params safely
):
    try:
        if not text:
            return {"status": "error", "error": "No text provided"}

        engine = (engine or DEFAULT_ENGINE).lower()
        voice = voice_id or DEFAULT_VOICE

        logger.info(f"TTS start | engine={engine} | voice={voice}")

        chunks = chunk_text(text, CHUNK_SIZE)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)

            if engine in ("edge", "edge-tts"):
                chunk_files = _edge_sync(chunks, tmp_dir, voice)
            else:
                return {"status": "error", "error": f"Unsupported engine: {engine}"}

            if not chunk_files:
                return {"status": "error", "error": "TTS synthesis produced no audio output."}

            output_dir = Path(output_dir or tmp_dir)
            output_path = output_dir / "output.wav"

            if len(chunk_files) == 1:
                output_path.write_bytes(chunk_files[0].read_bytes())
            else:
                if not merge_wav(chunk_files, output_path):
                    return {"status": "error", "error": "Merge failed"}

            return {
                "status": "success",
                "output_path": str(output_path),
                "chunks": len(chunk_files)
            }

    except Exception as e:
        logger.error(f"TTS failed: {e}", exc_info=True)
        return {"status": "error", "error": f"Internal server error: {str(e)}"}
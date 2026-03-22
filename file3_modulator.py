"""
Tarang 1.0.0.1 — file3_modulator.py
STATELESS: Reads WAV from bridge temp dir, writes MP3 to same temp dir.
Deletes the WAV immediately after MP3 is created.
Bridge uploads MP3 to Cloudinary then deletes the entire temp dir.
No sidecar JSON files written.
"""

import os
import logging
import numpy as np
from pathlib import Path
from datetime import datetime

try:
    from scipy.io import wavfile
    from scipy.signal import butter, filtfilt
except ImportError:
    raise ImportError("scipy required. Run: pip install scipy")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [file3_modulator] %(levelname)s — %(message)s"
)
logger = logging.getLogger("file3_modulator")

COGNITIVE_PRESETS = {
    "deep_focus":      {"beat_freq": 14.0, "carrier": 100.0, "depth": 0.15, "description": "Beta (14 Hz) — sustained focus and concentration"},
    "memory":          {"beat_freq": 10.0, "carrier": 100.0, "depth": 0.13, "description": "Alpha (10 Hz) — memory consolidation and recall"},
    "calm":            {"beat_freq":  8.0, "carrier": 100.0, "depth": 0.12, "description": "Alpha (8 Hz) — calm alert state, reduced anxiety"},
    "deep_relaxation": {"beat_freq":  6.0, "carrier": 100.0, "depth": 0.10, "description": "Theta (6 Hz) — deep relaxation, creative thought"},
    "sleep":           {"beat_freq":  4.0, "carrier": 100.0, "depth": 0.10, "description": "Theta/Delta (4 Hz) — drowsy, pre-sleep state"},
}

DEFAULT_STATE = os.getenv("TARANG_COGNITIVE_STATE", "deep_focus")


# ── Audio helpers ─────────────────────────────────────────────────────────────

def load_wav(filepath: Path) -> tuple:
    sample_rate, data = wavfile.read(str(filepath))
    if data.dtype == np.int16:
        audio = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        audio = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.float64:
        audio = data.astype(np.float32)
    else:
        audio = data.astype(np.float32)
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)
        logger.info("Stereo input detected — mixed down to mono.")
    logger.info(f"Loaded WAV: {filepath.name} | rate={sample_rate} Hz | frames={len(audio):,} | duration={len(audio)/sample_rate:.2f}s")
    return sample_rate, audio


def save_stereo_wav(filepath: Path, sample_rate: int, left: np.ndarray, right: np.ndarray):
    left  = np.clip(left,  -1.0, 1.0)
    right = np.clip(right, -1.0, 1.0)
    stereo = np.stack([left, right], axis=1)
    stereo_int16 = (stereo * 32767).astype(np.int16)
    wavfile.write(str(filepath), sample_rate, stereo_int16)
    logger.info(f"Saved stereo WAV → {filepath.name} ({filepath.stat().st_size // 1024} KB)")


def apply_lowpass_filter(audio: np.ndarray, sample_rate: int, cutoff_hz: float = 8000.0) -> np.ndarray:
    nyquist = sample_rate / 2.0
    normal_cutoff = min(cutoff_hz / nyquist, 0.99)
    b, a = butter(4, normal_cutoff, btype="low", analog=False)
    return filtfilt(b, a, audio).astype(np.float32)


def generate_am_carrier(n_frames: int, sample_rate: int, carrier_freq: float, depth: float) -> np.ndarray:
    t = np.arange(n_frames, dtype=np.float64) / sample_rate
    carrier = (1.0 - depth) + depth * np.cos(2.0 * np.pi * carrier_freq * t)
    return carrier.astype(np.float32)


def modulate(audio: np.ndarray, sample_rate: int, beat_freq: float, carrier_freq: float, depth: float) -> tuple:
    n = len(audio)
    logger.info(f"Modulation params — beat: {beat_freq} Hz | carrier L: {carrier_freq} Hz | carrier R: {carrier_freq + beat_freq} Hz | depth: {depth}")
    carrier_left  = generate_am_carrier(n, sample_rate, carrier_freq, depth)
    carrier_right = generate_am_carrier(n, sample_rate, carrier_freq + beat_freq, depth)
    left  = audio * carrier_left
    right = audio * carrier_right
    original_peak = np.max(np.abs(audio)) + 1e-9
    left  = left  * (original_peak / (np.max(np.abs(left))  + 1e-9))
    right = right * (original_peak / (np.max(np.abs(right)) + 1e-9))
    return left.astype(np.float32), right.astype(np.float32)


def add_fade(left: np.ndarray, right: np.ndarray, sample_rate: int,
             fade_in_sec: float = 2.0, fade_out_sec: float = 3.0) -> tuple:
    n = len(left)
    fade_in_samples  = min(int(fade_in_sec  * sample_rate), n // 4)
    fade_out_samples = min(int(fade_out_sec * sample_rate), n // 4)
    fade_in_curve  = (1 - np.cos(np.linspace(0, np.pi, fade_in_samples)))  / 2
    fade_out_curve = (1 + np.cos(np.linspace(0, np.pi, fade_out_samples))) / 2
    envelope = np.ones(n, dtype=np.float32)
    envelope[:fade_in_samples]      = fade_in_curve
    envelope[n - fade_out_samples:] = fade_out_curve
    logger.info(f"Fade applied — in: {fade_in_sec}s, out: {fade_out_sec}s")
    return (left * envelope).astype(np.float32), (right * envelope).astype(np.float32)


# ── Main modulation function ──────────────────────────────────────────────────

def modulate_audio(
    input_wav_path,
    cognitive_state: str = None,
    output_filename: str = None,
    custom_beat_freq: float = None,
    custom_depth: float = None,
    output_dir: Path = None,  # bridge passes its TemporaryDirectory path here
) -> dict:
    """
    STATELESS: Reads WAV from temp path, writes MP3 to same temp dir.
    Deletes WAV immediately after MP3 is ready.
    Bridge uploads MP3 to Cloudinary, then deletes entire temp dir.
    No sidecar JSON written.

    Returns output_path pointing to the MP3 in the temp dir.
    """
    input_wav_path = Path(input_wav_path)
    if not input_wav_path.exists():
        return {"status": "error", "error": f"Input WAV not found: {input_wav_path}"}

    state_key = (cognitive_state or DEFAULT_STATE).lower().strip()
    if state_key not in COGNITIVE_PRESETS:
        return {"status": "error", "error": f"Unknown cognitive state '{state_key}'. Choose from: {list(COGNITIVE_PRESETS.keys())}"}

    preset = COGNITIVE_PRESETS[state_key].copy()
    if custom_beat_freq is not None:
        preset["beat_freq"] = float(custom_beat_freq)
    if custom_depth is not None:
        preset["depth"] = max(0.0, min(1.0, float(custom_depth)))

    logger.info(f"Cognitive state: {state_key} — {preset['description']}")

    try:
        sample_rate, audio = load_wav(input_wav_path)
    except Exception as e:
        return {"status": "error", "error": f"Failed to load WAV: {e}"}

    logger.info("Applying pre-modulation low-pass filter...")
    audio = apply_lowpass_filter(audio, sample_rate, cutoff_hz=8000.0)

    try:
        left, right = modulate(
            audio=audio, sample_rate=sample_rate,
            beat_freq=preset["beat_freq"], carrier_freq=preset["carrier"],
            depth=preset["depth"]
        )
    except Exception as e:
        return {"status": "error", "error": f"Modulation failed: {e}"}

    left, right = add_fade(left, right, sample_rate, fade_in_sec=2.0, fade_out_sec=3.0)

    # Use same temp dir as input WAV (bridge's TemporaryDirectory)
    if output_dir is None:
        output_dir = input_wav_path.parent
    output_dir = Path(output_dir)

    stem     = output_filename or input_wav_path.stem.replace("_tts_raw", "")
    wav_path = output_dir / f"{stem}_modulated.wav"

    try:
        save_stereo_wav(wav_path, sample_rate, left, right)
    except Exception as e:
        return {"status": "error", "error": f"Failed to save modulated WAV: {e}"}

    # ── Convert WAV → MP3, then delete WAV ───────────────────────────────
    mp3_path = wav_path.with_suffix(".mp3")
    try:
        from pydub import AudioSegment
        audio_seg = AudioSegment.from_wav(str(wav_path))
        audio_seg.export(
            str(mp3_path), format="mp3", bitrate="192k",
            parameters=["-ac", "2"]   # keep stereo for binaural
        )
        logger.info(f"MP3 conversion complete — {mp3_path.name} ({mp3_path.stat().st_size // 1024} KB)")
        wav_path.unlink(missing_ok=True)   # delete WAV — not needed anymore
        output_path = mp3_path
    except Exception as e:
        logger.warning(f"MP3 conversion failed ({e}) — keeping WAV")
        output_path = wav_path

    # Also delete the input TTS raw WAV since it's no longer needed
    try:
        input_wav_path.unlink(missing_ok=True)
    except Exception:
        pass

    duration_sec = round(len(left) / sample_rate, 2)
    logger.info(f"Modulation complete — state: {state_key} | beat: {preset['beat_freq']} Hz | duration: {duration_sec}s | output: {output_path.name}")

    return {
        "status":          "success",
        "output_path":     str(output_path),   # temp path — bridge uploads to Cloudinary then deletes
        "cognitive_state": state_key,
        "beat_freq_hz":    preset["beat_freq"],
        "carrier_freq_hz": preset["carrier"],
        "depth":           preset["depth"],
        "duration_sec":    duration_sec,
        "sample_rate":     sample_rate,
        "timestamp":       datetime.utcnow().isoformat() + "Z",
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    print("\nTarang 1.0.0.1 — Binaural Audio Modulator")
    print("==========================================")
    for key, val in COGNITIVE_PRESETS.items():
        print(f"  {key:<18} → {val['description']}")
    print()
    if len(sys.argv) < 2:
        print("Usage: python file3_modulator.py <wav_file> [cognitive_state]")
        sys.exit(1)
    result = modulate_audio(
        input_wav_path=sys.argv[1],
        cognitive_state=sys.argv[2] if len(sys.argv) > 2 else None
    )
    if result["status"] == "success":
        print(f"\n✓ Modulation successful")
        print(f"  State    : {result['cognitive_state']}")
        print(f"  Beat Hz  : {result['beat_freq_hz']}")
        print(f"  Duration : {result['duration_sec']}s")
        print(f"  Output   : {result['output_path']}")
    else:
        print(f"\n✗ {result['error']}")
        sys.exit(1)

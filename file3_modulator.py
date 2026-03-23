"""
Tarang 1.0.0.1 — file3_modulator.py
STATELESS: Reads WAV from bridge temp dir, writes MP3 to same temp dir.
Deletes the WAV immediately after MP3 is created.
Bridge uploads MP3 to Cloudinary then deletes the entire temp dir.
No sidecar JSON files written.

v1.0.0.4 — Memory-safe rewrite:
  All heavy DSP (low-pass filter, AM modulation, stereo pan, fade) is done
  via a single ffmpeg subprocess using lavr/aformat + tremolo filters.
  Peak RAM usage is now ~50MB regardless of audio duration.
  No large numpy arrays are allocated. Safe on Render free tier (512MB limit).
"""

import os
import shutil
import logging
import subprocess
from pathlib import Path
from datetime import datetime

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


# ── ffmpeg resolver ───────────────────────────────────────────────────────────

def _get_ffmpeg() -> str:
    try:
        import imageio_ffmpeg
        path = imageio_ffmpeg.get_ffmpeg_exe()
        logger.info(f"ffmpeg via imageio-ffmpeg: {path}")
        return path
    except Exception:
        pass
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        logger.info(f"ffmpeg via system PATH: {system_ffmpeg}")
        return system_ffmpeg
    raise RuntimeError(
        "ffmpeg not found. Install imageio-ffmpeg (pip install imageio-ffmpeg) "
        "or ensure ffmpeg is on PATH."
    )


def _get_duration(ffmpeg: str, wav_path: Path) -> float:
    """Read audio duration in seconds from WAV header via ffprobe/ffmpeg."""
    probe = subprocess.run(
        [ffmpeg, "-i", str(wav_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    for line in probe.stderr.decode("utf-8", errors="replace").splitlines():
        if "Duration:" in line:
            try:
                dur_str = line.strip().split("Duration:")[1].split(",")[0].strip()
                h, m, s = dur_str.split(":")
                return int(h) * 3600 + int(m) * 60 + float(s)
            except Exception:
                pass
    return 0.0


def _modulate_via_ffmpeg(
    input_wav:  Path,
    output_mp3: Path,
    beat_freq:  float,
    carrier:    float,
    depth:      float,
    fade_in:    float = 2.0,
    fade_out:   float = 3.0,
    bitrate:    str   = "192k",
) -> float:
    """
    Full modulation pipeline in ONE ffmpeg pass — ~50MB RAM regardless of length.

    Filter graph:
      1. lowpass=8kHz          — remove frequencies above speech range
      2. asplit → [L][R]       — duplicate mono to two streams
      3. tremolo on L at carrier Hz          — AM modulation left channel
      4. tremolo on R at (carrier+beat) Hz   — AM modulation right channel
      5. amerge → stereo       — combine to binaural stereo
      6. afade in + afade out  — smooth start/end
      7. libmp3lame → MP3      — encode direct, no intermediate WAV

    tremolo filter: f=frequency(Hz), d=depth(0-1)
    Returns duration_sec parsed from ffmpeg output.
    """
    ffmpeg       = _get_ffmpeg()
    duration_sec = _get_duration(ffmpeg, input_wav)
    fade_out_st  = max(0.0, duration_sec - fade_out) if duration_sec > 0 else 0.0
    d            = min(float(depth), 0.99)  # clamp — 1.0 causes full silence at trough

    logger.info(
        f"ffmpeg modulate | beat={beat_freq}Hz | L={carrier}Hz | R={carrier+beat_freq}Hz "
        f"| depth={d} | duration={duration_sec:.1f}s | fade_out_start={fade_out_st:.1f}s"
    )

    filter_graph = (
        f"[0:a]"
        f"lowpass=f=8000,"
        f"aformat=channel_layouts=mono,"
        f"asplit=2[left_in][right_in];"

        f"[left_in]tremolo=f={carrier:.4f}:d={d:.4f}[left_mod];"

        f"[right_in]tremolo=f={carrier + beat_freq:.4f}:d={d:.4f}[right_mod];"

        f"[left_mod][right_mod]"
        f"amerge=inputs=2,"
        f"afade=t=in:st=0:d={fade_in:.2f},"
        f"afade=t=out:st={fade_out_st:.2f}:d={fade_out:.2f},"
        f"aformat=channel_layouts=stereo"
        f"[out]"
    )

    cmd = [
        ffmpeg, "-y",
        "-i", str(input_wav),
        "-filter_complex", filter_graph,
        "-map", "[out]",
        "-ar", "44100",
        "-ac", "2",
        "-b:a", bitrate,
        "-f", "mp3",
        str(output_mp3),
    ]

    logger.info(f"Running ffmpeg: {input_wav.name} → {output_mp3.name}")
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=600,
    )

    if result.returncode != 0:
        err = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg exited {result.returncode}: {err[-800:]}")

    if not output_mp3.exists() or output_mp3.stat().st_size == 0:
        raise RuntimeError(f"ffmpeg produced empty output: {output_mp3}")

    logger.info(
        f"ffmpeg complete — {output_mp3.name} ({output_mp3.stat().st_size // 1024} KB)"
    )
    return duration_sec


# ── Main modulation function ──────────────────────────────────────────────────

def modulate_audio(
    input_wav_path,
    cognitive_state:  str   = None,
    output_filename:  str   = None,
    custom_beat_freq: float = None,
    custom_depth:     float = None,
    output_dir:       Path  = None,
) -> dict:
    """
    STATELESS: Reads WAV, writes MP3 via ffmpeg (zero large numpy allocations).
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
        return {
            "status": "error",
            "error": f"Unknown cognitive state '{state_key}'. Choose from: {list(COGNITIVE_PRESETS.keys())}"
        }

    preset = COGNITIVE_PRESETS[state_key].copy()
    if custom_beat_freq is not None:
        preset["beat_freq"] = float(custom_beat_freq)
    if custom_depth is not None:
        preset["depth"] = max(0.0, min(1.0, float(custom_depth)))

    logger.info(f"Cognitive state: {state_key} — {preset['description']}")

    if output_dir is None:
        output_dir = input_wav_path.parent
    output_dir = Path(output_dir)

    stem     = output_filename or input_wav_path.stem.replace("_tts_raw", "")
    mp3_path = output_dir / f"{stem}_modulated.mp3"

    try:
        duration_sec = _modulate_via_ffmpeg(
            input_wav  = input_wav_path,
            output_mp3 = mp3_path,
            beat_freq  = preset["beat_freq"],
            carrier    = preset["carrier"],
            depth      = preset["depth"],
            fade_in    = 2.0,
            fade_out   = 3.0,
            bitrate    = "192k",
        )
    except Exception as e:
        return {"status": "error", "error": f"ffmpeg modulation failed: {e}"}

    # Delete the input TTS raw WAV — not needed anymore
    try:
        input_wav_path.unlink(missing_ok=True)
        logger.info(f"Deleted source WAV: {input_wav_path.name}")
    except Exception:
        pass

    logger.info(
        f"Modulation complete — state: {state_key} | beat: {preset['beat_freq']}Hz "
        f"| duration: {duration_sec}s | output: {mp3_path.name}"
    )

    return {
        "status":          "success",
        "output_path":     str(mp3_path),
        "cognitive_state": state_key,
        "beat_freq_hz":    preset["beat_freq"],
        "carrier_freq_hz": preset["carrier"],
        "depth":           preset["depth"],
        "duration_sec":    round(duration_sec, 2),
        "sample_rate":     44100,
        "timestamp":       datetime.utcnow().isoformat() + "Z",
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    print("\nTarang 1.0.0.4 — Binaural Audio Modulator (ffmpeg, memory-safe)")
    print("==================================================================")
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

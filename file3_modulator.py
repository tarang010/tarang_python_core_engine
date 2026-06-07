"""
Tarang 2.0.1 — file3_modulator.py
=====================================
STATELESS: Reads WAV from bridge temp dir, writes MP3 to same temp dir.
Deletes the WAV immediately after MP3 is created.
Bridge uploads MP3 to Cloudinary then deletes the entire temp dir.
No sidecar JSON files written.

v2.0.1 — No logic changes from v2.0.0.
  Version bump only — keeps versioning consistent with bridge.py v2.3.0
  and worker_pool.py v2.3.0 optimization batch.

v2.0.0 — Personality-Driven Mode Suggestion:
  - 5 cognitive presets (deep_focus, memory, calm, deep_relaxation, sleep)
  - suggest_cognitive_state() — takes quiz answers, returns recommended mode + reasoning
  - get_quiz_questions() — returns the personality/learning-style quiz for the frontend
  - EXPANDED_PRESETS — richer metadata per mode (brain_band, hz_range, best_for, avoid_if)
  - Modulation core (ffmpeg pipeline) unchanged from v1.0.0.4
  - Compatible with file1_extractor v2.5.0 and file2_tts v2.4.0 outputs

Mode suggestion logic:
  Users answer 5 quiz questions on the frontend (MERN).
  Frontend sends answers to bridge → bridge calls suggest_cognitive_state().
  Result is stored in user profile (MongoDB). User can override from profile page.
  No ML needed — rule-based scoring matrix maps quiz responses to mode weights.
"""

import os
import shutil
import logging
import subprocess
from pathlib import Path
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [file3_modulator v2.0.1] %(levelname)s — %(message)s"
)
logger = logging.getLogger("file3_modulator")

# ── Cognitive presets ─────────────────────────────────────────────────────────
COGNITIVE_PRESETS = {
    "deep_focus": {
        "beat_freq":   14.0,
        "carrier":     100.0,
        "depth":       0.15,
        "description": "Beta (14 Hz) — sustained focus and concentration",
        "brain_band":  "Beta",
        "hz_range":    "13–30 Hz",
        "best_for":    ["studying", "problem solving", "technical work", "active reading"],
        "avoid_if":    ["anxiety", "stress", "insomnia", "pre-sleep"],
        "color":       "#00d4ff",
    },
    "memory": {
        "beat_freq":   10.0,
        "carrier":     100.0,
        "depth":       0.13,
        "description": "Alpha (10 Hz) — memory consolidation and recall",
        "brain_band":  "Alpha",
        "hz_range":    "8–12 Hz",
        "best_for":    ["memorisation", "language learning", "exam prep", "review sessions"],
        "avoid_if":    ["deep focus tasks", "active problem solving"],
        "color":       "#a78bfa",
    },
    "calm": {
        "beat_freq":   8.0,
        "carrier":     100.0,
        "depth":       0.12,
        "description": "Alpha (8 Hz) — calm alert state, reduced anxiety",
        "brain_band":  "Alpha",
        "hz_range":    "8–12 Hz",
        "best_for":    ["anxious learners", "test anxiety", "relaxed study", "mindful reading"],
        "avoid_if":    ["wanting high energy", "active recall sessions"],
        "color":       "#34d399",
    },
    "deep_relaxation": {
        "beat_freq":   6.0,
        "carrier":     100.0,
        "depth":       0.10,
        "description": "Theta (6 Hz) — deep relaxation, creative thought",
        "brain_band":  "Theta",
        "hz_range":    "4–8 Hz",
        "best_for":    ["creative subjects", "arts", "lateral thinking", "pre-meditation study"],
        "avoid_if":    ["active recall", "analytical tasks", "exam day prep"],
        "color":       "#fb923c",
    },
    "sleep": {
        "beat_freq":   4.0,
        "carrier":     100.0,
        "depth":       0.10,
        "description": "Theta/Delta (4 Hz) — drowsy, pre-sleep state",
        "brain_band":  "Theta/Delta",
        "hz_range":    "0.5–4 Hz",
        "best_for":    ["pre-sleep review", "passive listening", "sleep learning research"],
        "avoid_if":    ["active study", "test prep", "problem solving"],
        "color":       "#818cf8",
    },
}

DEFAULT_STATE = os.getenv("TARANG_COGNITIVE_STATE", "deep_focus")


# ══════════════════════════════════════════════════════════════════════════════
# PERSONALITY QUIZ SYSTEM
# ══════════════════════════════════════════════════════════════════════════════

QUIZ_QUESTIONS = [
    {
        "id":       "q1",
        "question": "When do you usually study or learn best?",
        "options": [
            {"id": "a", "text": "Morning — I'm sharpest early"},
            {"id": "b", "text": "Afternoon — after I've warmed up"},
            {"id": "c", "text": "Evening — when things are quiet"},
            {"id": "d", "text": "Late night — I come alive after dark"},
        ],
    },
    {
        "id":       "q2",
        "question": "How would you describe your current mental state before a study session?",
        "options": [
            {"id": "a", "text": "Focused and energised — ready to go"},
            {"id": "b", "text": "A bit scattered — need to settle in"},
            {"id": "c", "text": "Anxious or stressed — hard to concentrate"},
            {"id": "d", "text": "Tired — pushing through low energy"},
        ],
    },
    {
        "id":       "q3",
        "question": "What is your primary learning goal with this content?",
        "options": [
            {"id": "a", "text": "Deep understanding of concepts"},
            {"id": "b", "text": "Remembering and recalling specific facts"},
            {"id": "c", "text": "Creative or reflective engagement"},
            {"id": "d", "text": "Passive absorption — I'll revisit later"},
        ],
    },
    {
        "id":       "q4",
        "question": "How do you typically respond to audio while studying?",
        "options": [
            {"id": "a", "text": "I love ambient sound — it keeps me in the zone"},
            {"id": "b", "text": "I need total silence — any sound breaks focus"},
            {"id": "c", "text": "Music helps when I'm anxious but not otherwise"},
            {"id": "d", "text": "Sound helps me drift into relaxed listening"},
        ],
    },
    {
        "id":       "q5",
        "question": "Which best describes how you learn most effectively?",
        "options": [
            {"id": "a", "text": "Active recall — testing myself repeatedly"},
            {"id": "b", "text": "Spaced repetition — reviewing over time"},
            {"id": "c", "text": "Immersive reading — long, uninterrupted sessions"},
            {"id": "d", "text": "Relaxed listening — absorbing passively"},
        ],
    },
]

# ── Scoring matrix ────────────────────────────────────────────────────────────
_SCORE_MATRIX = {
    "q1_a": {"deep_focus": 3, "memory": 1},
    "q1_b": {"memory": 2, "deep_focus": 1},
    "q1_c": {"calm": 2, "memory": 2},
    "q1_d": {"deep_relaxation": 2, "sleep": 2},
    "q2_a": {"deep_focus": 4},
    "q2_b": {"memory": 2, "calm": 1},
    "q2_c": {"calm": 4},
    "q2_d": {"deep_relaxation": 2, "sleep": 2},
    "q3_a": {"deep_focus": 4},
    "q3_b": {"memory": 4},
    "q3_c": {"deep_relaxation": 4},
    "q3_d": {"sleep": 3, "deep_relaxation": 1},
    "q4_a": {"deep_focus": 2, "memory": 2},
    "q4_b": {"deep_focus": 1},
    "q4_c": {"calm": 3},
    "q4_d": {"deep_relaxation": 3, "sleep": 1},
    "q5_a": {"deep_focus": 3, "memory": 1},
    "q5_b": {"memory": 4},
    "q5_c": {"deep_focus": 2, "calm": 2},
    "q5_d": {"deep_relaxation": 2, "sleep": 2},
}

_MODE_EXPLANATIONS = {
    "deep_focus": (
        "Based on your answers, you're ready for high-engagement learning. "
        "Beta waves at 14 Hz will keep your mind sharp and concentrated — "
        "perfect for the type of focused work you described."
    ),
    "memory": (
        "Your responses show this is a memory and retention session. "
        "Alpha waves at 10 Hz are the sweet spot for consolidating information "
        "and building strong recall pathways."
    ),
    "calm": (
        "We picked up signs of study stress or anxiety in your answers. "
        "Alpha waves at 8 Hz will ease you into a calm, clear mental state "
        "where learning flows naturally without pressure."
    ),
    "deep_relaxation": (
        "Your learning style and timing suggest a reflective, creative session. "
        "Theta waves at 6 Hz unlock lateral thinking and deep conceptual absorption — "
        "ideal for the immersive engagement you're after."
    ),
    "sleep": (
        "Your answers suggest passive, pre-rest absorption works best for you right now. "
        "Theta/Delta waves at 4 Hz ease you into a receptive state — "
        "great for gentle review before sleep."
    ),
}


def get_quiz_questions() -> dict:
    """
    Return the personality quiz for the MERN frontend to render.
    Frontend displays this as an onboarding step on first upload or profile setup.
    """
    return {
        "status":    "success",
        "version":   "2.0.1",
        "questions": QUIZ_QUESTIONS,
        "total":     len(QUIZ_QUESTIONS),
        "note":      "Send answers as { q1: 'a', q2: 'c', ... } to /suggest-mode",
    }


def suggest_cognitive_state(answers: dict) -> dict:
    """
    Takes quiz answers dict (e.g. {"q1": "a", "q2": "c", "q3": "b", ...})
    Returns the best-matching cognitive mode + scores + explanation.
    """
    mode_scores = {mode: 0 for mode in COGNITIVE_PRESETS}

    for q_id, option in answers.items():
        key = f"{q_id}_{option}"
        if key in _SCORE_MATRIX:
            for mode, delta in _SCORE_MATRIX[key].items():
                mode_scores[mode] = mode_scores.get(mode, 0) + delta
        else:
            logger.warning(f"suggest_cognitive_state: unknown answer key '{key}' — skipped")

    total = sum(mode_scores.values()) or 1
    pct_scores = {m: round((s / total) * 100, 1) for m, s in mode_scores.items()}
    ranked = sorted(pct_scores.items(), key=lambda x: x[1], reverse=True)
    best_mode = ranked[0][0]

    all_modes_out = []
    for mode, pct in ranked:
        preset = COGNITIVE_PRESETS[mode]
        all_modes_out.append({
            "mode":         mode,
            "label":        preset["description"],
            "brain_band":   preset["brain_band"],
            "beat_freq_hz": preset["beat_freq"],
            "score_pct":    pct,
            "best_for":     preset["best_for"],
            "color":        preset["color"],
        })

    logger.info(
        f"Mode suggestion | answers={answers} | "
        f"recommended={best_mode} ({pct_scores[best_mode]}%)"
    )

    return {
        "status":           "success",
        "recommended_mode": best_mode,
        "explanation":      _MODE_EXPLANATIONS[best_mode],
        "confidence_pct":   pct_scores[best_mode],
        "scores":           pct_scores,
        "all_modes":        all_modes_out,
        "can_override":     True,
        "override_note":    "You can change this anytime from your profile settings.",
    }


# ══════════════════════════════════════════════════════════════════════════════
# MODULATION CORE
# ══════════════════════════════════════════════════════════════════════════════

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
      1. lowpass=8kHz           — remove frequencies above speech range
      2. asplit → [L][R]        — duplicate mono to two streams
      3. tremolo on L at carrier Hz           — AM modulation left channel
      4. tremolo on R at (carrier+beat) Hz    — AM modulation right channel
      5. amerge → stereo        — combine to binaural stereo
      6. afade in + afade out   — smooth start/end
      7. libmp3lame → MP3       — encode direct, no intermediate WAV

    Returns duration_sec.
    """
    ffmpeg       = _get_ffmpeg()
    duration_sec = _get_duration(ffmpeg, input_wav)
    fade_out_st  = max(0.0, duration_sec - fade_out) if duration_sec > 0 else 0.0
    d            = min(float(depth), 0.99)

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
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=600,
    )

    if result.returncode != 0:
        err = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg exited {result.returncode}: {err[-800:]}")

    if not output_mp3.exists() or output_mp3.stat().st_size == 0:
        raise RuntimeError(f"ffmpeg produced empty output: {output_mp3}")

    logger.info(f"ffmpeg complete — {output_mp3.name} ({output_mp3.stat().st_size // 1024} KB)")
    return duration_sec


def modulate_audio(
    input_wav_path,
    cognitive_state:  str   = None,
    output_filename:  str   = None,
    custom_beat_freq: float = None,
    custom_depth:     float = None,
    output_dir:       Path  = None,
) -> dict:
    """
    STATELESS: Reads WAV, writes MP3 via ffmpeg.
    Deletes WAV immediately after MP3 is ready.

    cognitive_state can be the suggested mode from suggest_cognitive_state()
    or a user-overridden value from their profile.
    """
    input_wav_path = Path(input_wav_path)
    if not input_wav_path.exists():
        return {"status": "error", "error": f"Input WAV not found: {input_wav_path}"}

    state_key = (cognitive_state or DEFAULT_STATE).lower().strip()
    if state_key not in COGNITIVE_PRESETS:
        return {
            "status": "error",
            "error":  f"Unknown cognitive state '{state_key}'. Choose from: {list(COGNITIVE_PRESETS.keys())}"
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
        "status":           "success",
        "output_path":      str(mp3_path),
        "cognitive_state":  state_key,
        "beat_freq_hz":     preset["beat_freq"],
        "carrier_freq_hz":  preset["carrier"],
        "depth":            preset["depth"],
        "duration_sec":     round(duration_sec, 2),
        "sample_rate":      44100,
        "brain_band":       preset["brain_band"],
        "description":      preset["description"],
        "timestamp":        datetime.utcnow().isoformat() + "Z",
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    print("\nTarang 2.0.1 — Binaural Audio Modulator + Personality Quiz System")
    print("====================================================================")
    for key, val in COGNITIVE_PRESETS.items():
        print(f"  {key:<18} → {val['description']}")
    print()

    if "--quiz" in sys.argv:
        print("\nPersonality Quiz Questions:")
        quiz = get_quiz_questions()
        for q in quiz["questions"]:
            print(f"\n  {q['question']}")
            for opt in q["options"]:
                print(f"    {opt['id']}) {opt['text']}")
        sys.exit(0)

    if "--suggest" in sys.argv:
        answers = {}
        for arg in sys.argv[sys.argv.index("--suggest")+1:]:
            if "=" in arg:
                k, v = arg.split("=", 1)
                answers[k] = v
        result = suggest_cognitive_state(answers)
        print(f"\nRecommended mode: {result['recommended_mode']} ({result['confidence_pct']}%)")
        print(f"Explanation: {result['explanation']}")
        print(f"\nAll scores:")
        for m in result["all_modes"]:
            print(f"  {m['mode']:<18} {m['score_pct']:5.1f}%")
        sys.exit(0)

    if len(sys.argv) < 2:
        print("Usage: python file3_modulator.py <wav_file> [cognitive_state]")
        print("       python file3_modulator.py --quiz")
        print("       python file3_modulator.py --suggest q1=a q2=c q3=b q4=a q5=b")
        sys.exit(1)

    result = modulate_audio(
        input_wav_path=sys.argv[1],
        cognitive_state=sys.argv[2] if len(sys.argv) > 2 else None
    )
    if result["status"] == "success":
        print(f"\n✓ Modulation successful")
        print(f"  State    : {result['cognitive_state']} ({result['brain_band']})")
        print(f"  Beat Hz  : {result['beat_freq_hz']}")
        print(f"  Duration : {result['duration_sec']}s")
        print(f"  Output   : {result['output_path']}")
    else:
        print(f"\n✗ {result['error']}")
        sys.exit(1)
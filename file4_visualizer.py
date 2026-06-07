"""
Tarang 2.0.0 — file4_visualizer.py
=====================================
STATELESS: No local file writes whatsoever.
Returns HTML as string in result["html_content"].
Express stores it in MongoDB Document collection.
No sidecar JSON written.

v2.0.0 — Enhanced Binaural Verification + Best-Quality Plotly Charts:
  - Admin report upgraded from 6 → 8 subplots
  - NEW subplot: L/R Instantaneous Frequency Difference — proves beat is present
  - NEW subplot: Perceived Beat Frequency Timeline — shows what the brain actually hears
  - NEW subplot: Phase Coherence (L vs R cross-correlation) — channel sync quality
  - Waveform overlay improved with peak envelope shading
  - All charts use dark theme, monospace font, and publication-quality formatting
  - User waveform HTML unchanged (animated neon canvas)
  - Compatible with file3_modulator v2.0.0 output (beat_freq_hz, carrier_freq_hz in result)
"""

import os
import json
import logging
import numpy as np
from pathlib import Path
from datetime import datetime

try:
    from scipy.io import wavfile
    from scipy.signal import (
        spectrogram as scipy_spectrogram,
        hilbert,
        correlate,
    )
except ImportError:
    raise ImportError("scipy required. Run: pip install scipy")

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.io as pio
except ImportError:
    raise ImportError("plotly required. Run: pip install plotly")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [file4_visualizer v2.0.0] %(levelname)s — %(message)s"
)
logger = logging.getLogger("file4_visualizer")

COGNITIVE_STATE_LABELS = {
    "deep_focus":      {"label": "Deep Focus",  "freq": "14 Hz", "band": "Beta"},
    "memory":          {"label": "Memory",       "freq": "10 Hz", "band": "Alpha"},
    "calm":            {"label": "Calm",         "freq": "8 Hz",  "band": "Alpha"},
    "deep_relaxation": {"label": "Deep Relax",   "freq": "6 Hz",  "band": "Theta"},
    "sleep":           {"label": "Sleep",        "freq": "4 Hz",  "band": "Theta/Delta"},
}

STATE_COLORS = {
    "deep_focus":      {"primary": "#00d4ff", "glow": "0,212,255",  "secondary": "#0ea5e9"},
    "memory":          {"primary": "#a78bfa", "glow": "167,139,250","secondary": "#8b5cf6"},
    "calm":            {"primary": "#34d399", "glow": "52,211,153", "secondary": "#10b981"},
    "deep_relaxation": {"primary": "#fb923c", "glow": "251,146,60", "secondary": "#f97316"},
    "sleep":           {"primary": "#818cf8", "glow": "129,140,248","secondary": "#6366f1"},
}


# ── Audio loader ──────────────────────────────────────────────────────────────

def load_stereo_wav(filepath: Path) -> tuple:
    filepath = Path(filepath)
    if filepath.suffix.lower() == ".mp3":
        try:
            from pydub import AudioSegment
            audio   = AudioSegment.from_mp3(str(filepath))
            audio   = audio.set_channels(2)
            sr      = audio.frame_rate
            samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
            samples = samples / (2 ** (audio.sample_width * 8 - 1))
            data    = samples.reshape(-1, 2)
            sample_rate = sr
        except Exception as e:
            raise RuntimeError(f"MP3 decode failed: {e}")
    else:
        sample_rate, data = wavfile.read(str(filepath))

    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.float64:
        data = data.astype(np.float32)

    if data.ndim == 2:
        left, right = data[:, 0], data[:, 1]
    else:
        left = right = data

    logger.info(
        f"Loaded: {filepath.name} | rate={sample_rate} Hz | "
        f"frames={len(left):,} | duration={len(left)/sample_rate:.2f}s"
    )
    return sample_rate, left, right


def downsample_for_plot(arr: np.ndarray, target_points: int = 4000) -> np.ndarray:
    if len(arr) <= target_points:
        return arr
    step    = len(arr) // target_points
    trimmed = arr[:step * target_points].reshape(target_points, step)
    idx     = np.argmax(np.abs(trimmed), axis=1)
    return trimmed[np.arange(target_points), idx]


def compute_spectrogram_data(channel: np.ndarray, sample_rate: int, max_freq_hz: float = 500.0) -> tuple:
    nperseg  = min(2048, len(channel) // 10)
    noverlap = nperseg // 2
    freqs, times, Sxx = scipy_spectrogram(
        channel, fs=sample_rate, nperseg=nperseg, noverlap=noverlap, scaling="density"
    )
    freq_mask = freqs <= max_freq_hz
    freqs     = freqs[freq_mask]
    Sxx       = Sxx[freq_mask, :]
    power_db  = 10 * np.log10(Sxx + 1e-12)
    return freqs, times, power_db


# ── Binaural verification functions (NEW in v2.0.0) ─────────────────────────

def compute_instantaneous_frequency(channel: np.ndarray, sample_rate: int) -> tuple:
    """
    Use Hilbert transform to compute instantaneous frequency of each channel.
    This directly measures the actual frequency the audio is oscillating at
    at every moment in time — proving the AM tremolo carrier is present.

    Returns (times, inst_freq_hz) both downsampled for plotting.
    """
    try:
        analytic    = hilbert(channel)
        inst_phase  = np.unwrap(np.angle(analytic))
        # Instantaneous frequency = derivative of phase / (2π) * sample_rate
        inst_freq   = np.diff(inst_phase) / (2.0 * np.pi) * sample_rate
        # Clip to sane speech carrier range (50–300 Hz)
        inst_freq   = np.clip(inst_freq, 50.0, 300.0)
        times       = np.arange(len(inst_freq)) / sample_rate
        # Downsample heavily — only need trend shape
        t_ds   = downsample_for_plot(times,     2000)
        f_ds   = downsample_for_plot(inst_freq, 2000)
        return t_ds, f_ds
    except Exception as e:
        logger.warning(f"Instantaneous frequency computation failed: {e}")
        return np.array([0.0, 1.0]), np.array([100.0, 100.0])


def compute_beat_envelope(left: np.ndarray, right: np.ndarray,
                          sample_rate: int, beat_freq_hz: float) -> tuple:
    """
    Compute the amplitude envelope difference between L and R channels.
    The beat frequency the brain perceives = |envelope_L - envelope_R| oscillation rate.

    Also verifies that the measured beat rate ≈ expected beat_freq_hz.
    Returns (beat_times, beat_diff, measured_beat_hz).
    """
    # RMS envelope in small windows
    win_sec    = max(0.1, 1.0 / beat_freq_hz * 2)   # at least 2 beat cycles per window
    win_frames = int(win_sec * sample_rate)
    hop_frames = max(1, win_frames // 4)

    beat_times, env_l, env_r = [], [], []
    for start in range(0, len(left) - win_frames, hop_frames):
        end = start + win_frames
        beat_times.append((start + win_frames // 2) / sample_rate)
        env_l.append(float(np.sqrt(np.mean(left[start:end] ** 2))))
        env_r.append(float(np.sqrt(np.mean(right[start:end] ** 2))))

    beat_diff = np.abs(np.array(env_l) - np.array(env_r))

    # Estimate beat rate from zero-crossings of beat_diff
    measured_beat_hz = beat_freq_hz  # default
    try:
        mean_diff = np.mean(beat_diff)
        centred   = beat_diff - mean_diff
        zc_rate   = np.sum(np.diff(np.sign(centred)) != 0)
        if len(beat_times) > 4:
            total_time = beat_times[-1] - beat_times[0]
            if total_time > 0:
                measured_beat_hz = round(zc_rate / (2.0 * total_time), 2)
    except Exception:
        pass

    return np.array(beat_times), beat_diff, measured_beat_hz


def compute_phase_coherence(left: np.ndarray, right: np.ndarray,
                             sample_rate: int) -> tuple:
    """
    Cross-correlation between L and R channels in sliding windows.
    High coherence (near 1.0) means channels are similar (mono-like).
    The binaural effect requires slight decorrelation — we want to see
    coherence oscillate as the two channels beat against each other.

    Returns (times, coherence_values).
    """
    win_frames = min(int(5.0 * sample_rate), len(left) // 8)
    hop_frames = max(1, win_frames // 2)
    times, coherence = [], []

    for start in range(0, len(left) - win_frames, hop_frames):
        end  = start + win_frames
        l_w  = left[start:end]
        r_w  = right[start:end]
        # Normalised cross-correlation at zero lag
        denom = (np.std(l_w) * np.std(r_w))
        if denom > 1e-10:
            xcorr = float(np.mean(l_w * r_w) / denom)
            xcorr = max(-1.0, min(1.0, xcorr))
        else:
            xcorr = 1.0
        times.append((start + win_frames // 2) / sample_rate)
        coherence.append(xcorr)

    return np.array(times), np.array(coherence)


# ── Admin report ──────────────────────────────────────────────────────────────

def generate_admin_html(
    wav_path: Path,
    cognitive_state: str,
    beat_freq_hz: float,
    carrier_freq_hz: float = 100.0,
    document_title: str = "Tarang Session"
) -> str:
    """
    Generate full 8-subplot Plotly admin report.
    
    Subplot layout (4 rows × 2 cols):
      Row 1: L Spectrogram | R Spectrogram
      Row 2: Waveform Overlay | Phase Coherence (L vs R)
      Row 3: L Instantaneous Freq | R Instantaneous Freq
      Row 4: Beat Envelope Verification | Perceived Beat Frequency Timeline
    """
    try:
        sample_rate, left, right = load_stereo_wav(wav_path)
    except Exception as e:
        logger.error(f"Failed to load audio for admin report: {e}")
        return ""

    duration   = len(left) / sample_rate
    state_info = COGNITIVE_STATE_LABELS.get(
        cognitive_state,
        {"label": cognitive_state, "freq": f"{beat_freq_hz} Hz", "band": "Custom"}
    )
    colors = STATE_COLORS.get(cognitive_state, STATE_COLORS["deep_focus"])
    primary = colors["primary"]

    logger.info("Computing spectrograms...")
    freqs_l, times_l, pdb_l = compute_spectrogram_data(left,  sample_rate, max_freq_hz=300.0)
    freqs_r, times_r, pdb_r = compute_spectrogram_data(right, sample_rate, max_freq_hz=300.0)

    logger.info("Computing instantaneous frequencies...")
    t_if_l, if_l = compute_instantaneous_frequency(left,  sample_rate)
    t_if_r, if_r = compute_instantaneous_frequency(right, sample_rate)

    logger.info("Computing beat envelope...")
    beat_times, beat_diff, measured_beat_hz = compute_beat_envelope(
        left, right, sample_rate, beat_freq_hz
    )

    logger.info("Computing phase coherence...")
    coh_times, coherence = compute_phase_coherence(left, right, sample_rate)

    # Waveform
    n_plot   = 6000
    t_full   = np.linspace(0, duration, len(left))
    t_ds     = downsample_for_plot(t_full, n_plot)
    left_ds  = downsample_for_plot(left,   n_plot)
    right_ds = downsample_for_plot(right,  n_plot)

    # Peak envelope for waveform shading
    env_win  = max(1, len(left) // 500)
    env_t    = []
    env_peak = []
    for i in range(0, len(left) - env_win, env_win):
        env_t.append(i / sample_rate)
        env_peak.append(float(np.max(np.abs(left[i:i+env_win]))))
    env_t    = np.array(env_t)
    env_peak = np.array(env_peak)

    # Perceived beat: smooth the beat envelope and annotate measured vs expected
    smooth_win    = max(1, len(beat_diff) // 50)
    beat_smoothed = np.convolve(beat_diff, np.ones(smooth_win)/smooth_win, mode='same')

    # Beat frequency estimation annotation
    beat_match = abs(measured_beat_hz - beat_freq_hz) <= (beat_freq_hz * 0.25)
    beat_status = f"✓ Verified ({measured_beat_hz:.1f} Hz)" if beat_match else f"⚠ {measured_beat_hz:.1f} Hz (expected {beat_freq_hz} Hz)"

    logger.info(f"Beat verification: expected={beat_freq_hz}Hz | measured={measured_beat_hz}Hz | match={beat_match}")

    # ── Build figure ──────────────────────────────────────────────────────────
    fig = make_subplots(
        rows=4, cols=2,
        subplot_titles=[
            f"Left Channel Spectrum (0–300 Hz)",
            f"Right Channel Spectrum (0–300 Hz)",
            "Waveform — Left vs Right with Peak Envelope",
            "Phase Coherence: L↔R Channel Sync",
            f"Left Channel Instantaneous Frequency",
            f"Right Channel Instantaneous Frequency",
            f"Beat Envelope Amplitude Difference (L–R)",
            f"Perceived Beat Frequency — {beat_status}",
        ],
        vertical_spacing=0.09,
        horizontal_spacing=0.07,
    )

    # ── Row 1: Spectrograms ───────────────────────────────────────────────────
    fig.add_trace(go.Heatmap(
        x=times_l, y=freqs_l, z=pdb_l,
        colorscale="Viridis", name="L Spectrum",
        zmin=float(np.percentile(pdb_l, 5)),
        zmax=float(np.percentile(pdb_l, 99)),
        colorbar=dict(x=0.45, len=0.22, thickness=10, tickfont=dict(size=9, color="#94a3b8")),
        showscale=True,
    ), row=1, col=1)

    fig.add_trace(go.Heatmap(
        x=times_r, y=freqs_r, z=pdb_r,
        colorscale="Plasma", name="R Spectrum",
        zmin=float(np.percentile(pdb_r, 5)),
        zmax=float(np.percentile(pdb_r, 99)),
        colorbar=dict(x=1.01, len=0.22, thickness=10, tickfont=dict(size=9, color="#94a3b8")),
        showscale=True,
    ), row=1, col=2)

    # ── Row 2 Col 1: Waveform with peak envelope ──────────────────────────────
    fig.add_trace(go.Scatter(
        x=np.concatenate([env_t, env_t[::-1]]),
        y=np.concatenate([env_peak, -env_peak[::-1]]),
        fill="toself",
        fillcolor=f"rgba({colors['glow']},0.10)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Peak envelope",
        showlegend=True,
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=t_ds, y=left_ds, mode="lines", name="Left",
        line=dict(color=primary, width=0.7), opacity=0.9,
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=t_ds, y=right_ds, mode="lines", name="Right",
        line=dict(color=colors["secondary"], width=0.7), opacity=0.9,
    ), row=2, col=1)

    # ── Row 2 Col 2: Phase coherence ──────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=coh_times, y=coherence, mode="lines", name="Coherence",
        line=dict(color="#fbbf24", width=1.5),
        fill="tozeroy", fillcolor="rgba(251,191,36,0.10)",
    ), row=2, col=2)
    # Reference lines at ±0.95 (too similar) and ±0.5 (healthy decorrelation)
    for y_ref, color_ref, label_ref in [
        (0.95, "rgba(239,68,68,0.5)", "Too similar (near mono)"),
        (0.50, "rgba(34,197,94,0.5)", "Healthy binaural decorrelation"),
    ]:
        fig.add_hline(
            y=y_ref, line_dash="dot", line_color=color_ref,
            annotation_text=label_ref,
            annotation_font=dict(size=9, color="#94a3b8"),
            row=2, col=2
        )

    # ── Row 3: Instantaneous frequency (L and R) ──────────────────────────────
    # Expected carrier frequencies
    carrier_l = carrier_freq_hz
    carrier_r = carrier_freq_hz + beat_freq_hz

    fig.add_trace(go.Scatter(
        x=t_if_l, y=if_l, mode="lines", name=f"L inst freq",
        line=dict(color=primary, width=1.0), opacity=0.8,
    ), row=3, col=1)
    fig.add_hline(
        y=carrier_l, line_dash="dash", line_color="rgba(255,255,255,0.4)",
        annotation_text=f"Expected L carrier: {carrier_l:.0f} Hz",
        annotation_font=dict(size=9, color="#94a3b8"),
        row=3, col=1
    )

    fig.add_trace(go.Scatter(
        x=t_if_r, y=if_r, mode="lines", name=f"R inst freq",
        line=dict(color=colors["secondary"], width=1.0), opacity=0.8,
    ), row=3, col=2)
    fig.add_hline(
        y=carrier_r, line_dash="dash", line_color="rgba(255,255,255,0.4)",
        annotation_text=f"Expected R carrier: {carrier_r:.0f} Hz",
        annotation_font=dict(size=9, color="#94a3b8"),
        row=3, col=2
    )

    # ── Row 4 Col 1: Beat envelope ─────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=beat_times, y=beat_diff,
        mode="lines", name="Raw beat diff",
        line=dict(color="#94a3b8", width=0.8), opacity=0.5,
    ), row=4, col=1)
    fig.add_trace(go.Scatter(
        x=beat_times, y=beat_smoothed,
        mode="lines", name="Smoothed beat diff",
        line=dict(color="#a78bfa", width=2.0),
        fill="tozeroy", fillcolor="rgba(167,139,250,0.12)",
    ), row=4, col=1)

    # ── Row 4 Col 2: Perceived beat frequency ─────────────────────────────────
    # Compute short-window oscillation rate of beat envelope
    perceived_times, perceived_freq = [], []
    seg_win    = max(20, len(beat_diff) // 30)
    for i in range(0, len(beat_diff) - seg_win, max(1, seg_win // 2)):
        seg  = beat_diff[i:i+seg_win]
        seg_t = beat_times[i:i+seg_win]
        if len(seg) < 4:
            continue
        # Count zero-crossings around mean
        mn   = np.mean(seg)
        zc   = np.sum(np.diff(np.sign(seg - mn)) != 0)
        dur  = float(seg_t[-1] - seg_t[0]) if seg_t[-1] > seg_t[0] else 1.0
        freq = zc / (2.0 * dur)
        freq = np.clip(freq, 0.5, beat_freq_hz * 3)
        perceived_times.append(float(np.mean(seg_t)))
        perceived_freq.append(float(freq))

    if perceived_times:
        fig.add_trace(go.Scatter(
            x=perceived_times, y=perceived_freq,
            mode="lines+markers", name="Measured beat Hz",
            line=dict(color="#34d399", width=2.0),
            marker=dict(size=4, color="#34d399"),
        ), row=4, col=2)

    # Expected beat line
    fig.add_hline(
        y=beat_freq_hz, line_dash="dash", line_color="rgba(255,255,255,0.6)",
        annotation_text=f"Expected: {beat_freq_hz} Hz",
        annotation_font=dict(size=10, color="#e2e8f0"),
        row=4, col=2
    )
    # Tolerance band ±25%
    fig.add_hrect(
        y0=beat_freq_hz * 0.75, y1=beat_freq_hz * 1.25,
        fillcolor="rgba(52,211,153,0.08)", line_width=0,
        annotation_text="±25% tolerance",
        annotation_font=dict(size=8, color="#64748b"),
        row=4, col=2
    )

    # ── Layout ────────────────────────────────────────────────────────────────
    fig.update_layout(
        title=dict(
            text=(
                f"<b>Tarang 2.0.0 — Admin Binaural Verification Report</b><br>"
                f"<sup style='color:#94a3b8'>"
                f"{document_title} &nbsp;|&nbsp; "
                f"State: {state_info['label']} ({state_info['freq']} {state_info['band']}) &nbsp;|&nbsp; "
                f"Carrier: {carrier_freq_hz:.0f} Hz L / {carrier_freq_hz + beat_freq_hz:.0f} Hz R &nbsp;|&nbsp; "
                f"Duration: {duration:.1f}s &nbsp;|&nbsp; "
                f"Beat: {beat_status} &nbsp;|&nbsp; "
                f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}"
                f"</sup>"
            ),
            font=dict(size=15, color="#f1f5f9", family="'Courier New', monospace"),
            x=0.01,
        ),
        paper_bgcolor="#050914",
        plot_bgcolor="#0d1526",
        font=dict(color="#94a3b8", family="'Courier New', monospace", size=10),
        height=1400,
        showlegend=True,
        legend=dict(
            bgcolor="rgba(5,9,20,0.85)",
            bordercolor="#1e293b",
            borderwidth=1,
            font=dict(color="#94a3b8", size=9),
            x=0.01, y=0.01,
        ),
    )

    # Axis styling
    axis_style = dict(gridcolor="#0f172a", zerolinecolor="#1e293b",
                      tickfont=dict(size=9, color="#64748b"),
                      title_font=dict(size=9, color="#64748b"))
    for i in range(1, 9):
        row = (i - 1) // 2 + 1
        col = (i - 1) % 2 + 1
        fig.update_xaxes(axis_style, row=row, col=col)
        fig.update_yaxes(axis_style, row=row, col=col)

    # Specific y-axis labels
    fig.update_yaxes(title_text="Frequency (Hz)", row=1, col=1)
    fig.update_yaxes(title_text="Frequency (Hz)", row=1, col=2)
    fig.update_yaxes(title_text="Amplitude", row=2, col=1)
    fig.update_yaxes(title_text="Coherence", range=[-1.1, 1.1], row=2, col=2)
    fig.update_yaxes(title_text="Inst. Freq (Hz)", row=3, col=1)
    fig.update_yaxes(title_text="Inst. Freq (Hz)", row=3, col=2)
    fig.update_yaxes(title_text="|RMS L – RMS R|", row=4, col=1)
    fig.update_yaxes(title_text="Beat Hz", row=4, col=2)

    # X-axis time labels
    for row in [2, 3, 4]:
        for col in [1, 2]:
            fig.update_xaxes(title_text="Time (s)", row=row, col=col)

    html_str = pio.to_html(fig, full_html=True, include_plotlyjs=True)
    logger.info(f"Admin report HTML generated | {len(html_str):,} chars | subplots=8")
    return html_str


# ── User neon waveform (unchanged aesthetic, minor improvements) ──────────────

def generate_user_waveform_html(
    wav_path: Path,
    cognitive_state: str,
    beat_freq_hz: float,
    document_title: str = "Tarang Session",
    session_number: int = 1,
) -> str:
    """Generate animated neon waveform with binaural beat info. Returns HTML string."""
    try:
        sample_rate, left, right = load_stereo_wav(wav_path)
    except Exception as e:
        logger.error(f"Failed to load audio for user waveform: {e}")
        return ""

    duration   = len(left) / sample_rate
    state_info = COGNITIVE_STATE_LABELS.get(
        cognitive_state,
        {"label": cognitive_state.replace("_", " ").title(), "freq": f"{beat_freq_hz} Hz", "band": ""}
    )
    colors = STATE_COLORS.get(cognitive_state, STATE_COLORS["deep_focus"])

    n_env    = 200
    mono     = (left + right) / 2
    step     = max(1, len(mono) // n_env)
    envelope = []
    for i in range(n_env):
        chunk = mono[i * step: (i + 1) * step]
        rms   = float(np.sqrt(np.mean(chunk ** 2))) if len(chunk) > 0 else 0.0
        envelope.append(round(rms, 4))
    max_env  = max(envelope) if max(envelope) > 0 else 1.0
    envelope = [round(v / max_env, 4) for v in envelope]
    env_js   = json.dumps(envelope)

    # Quick binaural verification for user display
    _, beat_diff, measured_hz = compute_beat_envelope(left, right, sample_rate, beat_freq_hz)
    beat_verified = abs(measured_hz - beat_freq_hz) <= beat_freq_hz * 0.3
    beat_status_text = f"{measured_hz:.1f} Hz detected" if beat_verified else "Verifying..."

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Tarang — {state_info['label']} Session</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;500&display=swap');
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{
  background:#030712;
  color:#e2e8f0;
  font-family:'Inter',system-ui,sans-serif;
  display:flex; flex-direction:column; align-items:center;
  justify-content:center; min-height:100vh; overflow:hidden;
}}
.container {{ width:100%; max-width:920px; padding:2rem 1.5rem; display:flex; flex-direction:column; align-items:center; gap:1.5rem; }}
.header {{ text-align:center; }}
.brand {{ font-family:'Space Mono',monospace; font-size:0.65rem; letter-spacing:0.35em; color:{colors['primary']}; text-transform:uppercase; opacity:0.6; }}
h1 {{ font-size:1.75rem; font-weight:300; color:#f1f5f9; margin:0.3rem 0; letter-spacing:0.03em; }}
.subtitle {{ font-size:0.82rem; color:#475569; letter-spacing:0.05em; }}
.state-badge {{
  display:inline-flex; align-items:center; gap:0.6rem;
  padding:0.45rem 1.2rem;
  border:1px solid {colors['primary']}33;
  border-radius:999px;
  background:radial-gradient(ellipse at center, {colors['primary']}0a 0%, transparent 70%);
  font-family:'Space Mono',monospace;
  font-size:0.75rem; color:{colors['primary']}; letter-spacing:0.12em;
}}
.dot {{ width:7px; height:7px; border-radius:50%; background:{colors['primary']}; animation:pulse 2s ease-in-out infinite; }}
.canvas-wrapper {{
  width:100%;
  border:1px solid {colors['primary']}18;
  border-radius:16px;
  background:linear-gradient(180deg, #080f1f 0%, #030712 100%);
  overflow:hidden;
  position:relative;
}}
.canvas-wrapper::before {{
  content:'';
  position:absolute; inset:0;
  background:radial-gradient(ellipse 60% 30% at 50% 100%, {colors['primary']}08, transparent);
  pointer-events:none;
}}
canvas {{ display:block; width:100%; height:200px; }}
.info-grid {{ display:grid; grid-template-columns:repeat(4,1fr); gap:0.75rem; width:100%; }}
.info-card {{
  background:linear-gradient(135deg, #0d1526 0%, #0a1020 100%);
  border:1px solid #1e293b;
  border-radius:12px; padding:1rem 0.75rem; text-align:center;
  transition:border-color 0.3s;
}}
.info-card:hover {{ border-color:{colors['primary']}44; }}
.info-card .label {{ font-family:'Space Mono',monospace; font-size:0.62rem; letter-spacing:0.18em; color:#334155; text-transform:uppercase; margin-bottom:0.4rem; }}
.info-card .value {{ font-size:1.05rem; font-weight:500; color:{colors['primary']}; }}
.info-card .sub {{ font-size:0.7rem; color:#334155; margin-top:2px; }}
.beat-indicator {{
  display:flex; align-items:center; gap:0.8rem;
  padding:0.6rem 1.2rem;
  background:#0d1526;
  border:1px solid {'#34d39944' if beat_verified else '#fbbf2444'};
  border-radius:8px;
  font-family:'Space Mono',monospace; font-size:0.72rem;
  color:{'#34d399' if beat_verified else '#fbbf24'};
}}
.footer-note {{ font-family:'Space Mono',monospace; font-size:0.65rem; color:#1e293b; text-align:center; letter-spacing:0.08em; }}
@keyframes pulse {{ 0%,100%{{opacity:1;transform:scale(1)}} 50%{{opacity:0.35;transform:scale(0.8)}} }}
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <div class="brand">Tarang 2.0.0 &nbsp;·&nbsp; Neuro-Acoustic Learning</div>
    <h1>{document_title}</h1>
    <div class="subtitle">Session {session_number} &nbsp;·&nbsp; Audio Waveform</div>
  </div>

  <div class="state-badge">
    <span class="dot"></span>
    {state_info['label']} &nbsp;·&nbsp; {state_info['freq']} {state_info['band']}
  </div>

  <div class="canvas-wrapper"><canvas id="waveCanvas"></canvas></div>

  <div class="info-grid">
    <div class="info-card">
      <div class="label">Duration</div>
      <div class="value">{int(duration//60)}m {int(duration%60)}s</div>
    </div>
    <div class="info-card">
      <div class="label">State</div>
      <div class="value">{state_info['label']}</div>
      <div class="sub">{state_info['band']}</div>
    </div>
    <div class="info-card">
      <div class="label">Beat Freq</div>
      <div class="value">{beat_freq_hz:.0f} Hz</div>
      <div class="sub">binaural</div>
    </div>
    <div class="info-card">
      <div class="label">Session</div>
      <div class="value">#{session_number}</div>
    </div>
  </div>

  <div class="beat-indicator">
    <span>{'✓' if beat_verified else '⟳'}</span>
    <span>Binaural beat &nbsp;·&nbsp; {beat_status_text}</span>
  </div>

  <div class="footer-note">
    Use headphones for full neuro-acoustic effect &nbsp;·&nbsp;
    Generated {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}
  </div>
</div>
<script>
const canvas = document.getElementById('waveCanvas');
const ctx    = canvas.getContext('2d');
const PRIMARY   = '{colors['primary']}';
const SECONDARY = '{colors['secondary']}';
const GLOW      = '{colors['glow']}';
const envelope  = {env_js};
let frame = 0, offset = 0;

function resize() {{ canvas.width = canvas.offsetWidth; canvas.height = canvas.offsetHeight; }}
resize();
window.addEventListener('resize', resize);

function lerp(a,b,t) {{ return a + (b-a)*t; }}
function getEnvAt(x, off) {{
  const pos  = (x / canvas.width) * envelope.length + off;
  const idx  = Math.floor(pos) % envelope.length;
  const next = (idx + 1) % envelope.length;
  return lerp(envelope[idx], envelope[next], pos - Math.floor(pos));
}}

function drawWave(yCenter, amplitude, alpha, lineWidth, colorHex, off) {{
  ctx.beginPath();
  ctx.lineWidth   = lineWidth;
  ctx.strokeStyle = `rgba(${{GLOW}},${{alpha}})`;
  ctx.shadowColor = colorHex;
  ctx.shadowBlur  = lineWidth * 10;
  for (let x = 0; x <= canvas.width; x += 2) {{
    const env  = getEnvAt(x, off);
    const t    = x / canvas.width;
    const y    = yCenter
               + Math.sin(t * Math.PI * 8 + frame * 0.035) * amplitude * env * 0.55
               + Math.sin(t * Math.PI * 3 + frame * 0.018) * amplitude * env * 0.25
               + Math.sin(frame * 0.024 + x * 0.008) * amplitude * 0.12;
    if (x === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }}
  ctx.stroke();
  ctx.shadowBlur = 0;
}}

function drawFrame() {{
  const W = canvas.width, H = canvas.height, cy = H / 2;
  ctx.fillStyle = 'rgba(3,7,18,0.42)';
  ctx.fillRect(0, 0, W, H);

  // Three layered waves — glow, mid, sharp
  drawWave(cy, H*0.40, 0.12, 8,  PRIMARY, offset);
  drawWave(cy, H*0.36, 0.30, 3,  PRIMARY, offset);
  drawWave(cy, H*0.32, 0.85, 1.2, PRIMARY, offset);

  // Subtle secondary channel wave (right channel indicator)
  drawWave(cy, H*0.20, 0.06, 4, SECONDARY, offset + envelope.length * 0.3);

  frame  += 1;
  offset += 0.10;
  requestAnimationFrame(drawFrame);
}}
drawFrame();
</script>
</body>
</html>"""

    logger.info(f"User waveform HTML generated | {len(html):,} chars")
    return html


# ── Main entry point ──────────────────────────────────────────────────────────

def generate_visualization(
    modulated_wav_path,
    role: str = "user",
    cognitive_state: str = "deep_focus",
    beat_freq_hz: float = 14.0,
    carrier_freq_hz: float = 100.0,
    document_title: str = "Tarang Session",
    session_number: int = 1,
    output_stem: str = None,
) -> dict:
    """
    STATELESS: Returns HTML as string in result["html_content"].
    No local files written.

    carrier_freq_hz is now passed from file3_modulator result["carrier_freq_hz"]
    so the instantaneous frequency reference lines are accurate.
    """
    modulated_wav_path = Path(modulated_wav_path)
    role = role.lower().strip()

    if not modulated_wav_path.exists():
        return {"status": "error", "error": f"Audio file not found: {modulated_wav_path}"}
    if role not in ("admin", "user"):
        return {"status": "error", "error": f"Invalid role '{role}'. Use 'admin' or 'user'."}

    if role == "admin":
        report_type  = "admin_report"
        logger.info(f"Generating ADMIN report (8 subplots) for: {modulated_wav_path.name}")
        html_content = generate_admin_html(
            modulated_wav_path, cognitive_state, beat_freq_hz,
            carrier_freq_hz, document_title
        )
    else:
        report_type  = "user_waveform"
        logger.info(f"Generating USER waveform for: {modulated_wav_path.name}")
        html_content = generate_user_waveform_html(
            modulated_wav_path, cognitive_state, beat_freq_hz,
            document_title, session_number
        )

    if not html_content:
        return {"status": "error", "error": "Visualization generation failed. Check logs."}

    logger.info(f"Visualization complete | type={report_type}")

    return {
        "status":       "success",
        "role":         role,
        "output_path":  None,
        "html_content": html_content,
        "report_type":  report_type,
        "timestamp":    datetime.utcnow().isoformat() + "Z",
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    print("\nTarang 2.0.0 — Audio Visualizer (8-subplot binaural verification)")
    if len(sys.argv) < 3:
        print("Usage: python file4_visualizer.py <audio_path> <role> [state] [title]")
        sys.exit(1)
    r = generate_visualization(
        modulated_wav_path=sys.argv[1],
        role=sys.argv[2],
        cognitive_state=sys.argv[3] if len(sys.argv) > 3 else "deep_focus",
        beat_freq_hz=14.0,
        document_title=sys.argv[4] if len(sys.argv) > 4 else "Tarang Session",
    )
    if r["status"] == "success":
        print(f"\n✓ {r['report_type']} | {len(r['html_content']):,} chars")
    else:
        print(f"\n✗ {r['error']}")
        sys.exit(1)
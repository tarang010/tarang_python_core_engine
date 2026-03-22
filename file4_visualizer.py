"""
Tarang 1.0.0.1 — file4_visualizer.py
STATELESS: No local file writes whatsoever.
Returns HTML as string in result["html_content"].
Express stores it in MongoDB Document collection.
No sidecar JSON written.
"""

import os
import json
import logging
import numpy as np
from pathlib import Path
from datetime import datetime

try:
    from scipy.io import wavfile
    from scipy.signal import spectrogram as scipy_spectrogram
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
    format="%(asctime)s [file4_visualizer] %(levelname)s — %(message)s"
)
logger = logging.getLogger("file4_visualizer")

COGNITIVE_STATE_LABELS = {
    "deep_focus":      {"label": "Deep Focus", "freq": "14 Hz", "band": "Beta"},
    "memory":          {"label": "Memory",     "freq": "10 Hz", "band": "Alpha"},
    "calm":            {"label": "Calm",        "freq": "8 Hz",  "band": "Alpha"},
    "deep_relaxation": {"label": "Deep Relax",  "freq": "6 Hz",  "band": "Theta"},
    "sleep":           {"label": "Sleep",       "freq": "4 Hz",  "band": "Theta/Delta"},
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
        f"frames={len(left):,} | duration={len(left)/sample_rate:.2f}s | "
        f"channels={'stereo' if data.ndim==2 else 'mono'}"
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


# ── Admin report ──────────────────────────────────────────────────────────────

def generate_admin_html(
    wav_path: Path,
    cognitive_state: str,
    beat_freq_hz: float,
    document_title: str = "Tarang Session"
) -> str:
    """Generate full Plotly admin report. Returns HTML string."""
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

    logger.info("Computing spectrograms (this may take a moment for long audio)...")
    freqs_l, times_l, pdb_l = compute_spectrogram_data(left,  sample_rate, max_freq_hz=300.0)
    freqs_r, times_r, pdb_r = compute_spectrogram_data(right, sample_rate, max_freq_hz=300.0)

    n_plot   = 6000
    t_full   = np.linspace(0, duration, len(left))
    t_ds     = downsample_for_plot(t_full, n_plot)
    left_ds  = downsample_for_plot(left,   n_plot)
    right_ds = downsample_for_plot(right,  n_plot)

    # Beat envelope verification
    win_frames = 5 * sample_rate
    beat_times, beat_env = [], []
    for start in range(0, len(left) - win_frames, win_frames // 10):
        end = start + win_frames
        rms_l = np.sqrt(np.mean(left[start:end] ** 2))
        rms_r = np.sqrt(np.mean(right[start:end] ** 2))
        beat_times.append((start + win_frames // 2) / sample_rate)
        beat_env.append(abs(rms_l - rms_r))

    # Modulation depth trend
    win_d = 10 * sample_rate
    depth_times, depth_left, depth_right = [], [], []
    for start in range(0, len(left) - win_d, win_d // 2):
        end = start + win_d
        depth_times.append((start + win_d // 2) / sample_rate)
        depth_left.append(float(np.ptp(left[start:end])))
        depth_right.append(float(np.ptp(right[start:end])))

    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=[
            "Left Channel — Frequency Spectrum (0–300 Hz)",
            "Right Channel — Frequency Spectrum (0–300 Hz)",
            "Waveform Overlay — Left vs Right",
            "Beat Envelope Verification",
            "Modulation Depth Trend (Left)",
            "Modulation Depth Trend (Right)",
        ],
        vertical_spacing=0.12, horizontal_spacing=0.08
    )

    fig.add_trace(go.Heatmap(x=times_l, y=freqs_l, z=pdb_l, colorscale="Viridis", name="Left spectrum",
                             zmin=float(np.percentile(pdb_l, 5)), zmax=float(np.percentile(pdb_l, 99))), row=1, col=1)
    fig.add_trace(go.Heatmap(x=times_r, y=freqs_r, z=pdb_r, colorscale="Plasma",  name="Right spectrum",
                             zmin=float(np.percentile(pdb_r, 5)), zmax=float(np.percentile(pdb_r, 99))), row=1, col=2)
    fig.add_trace(go.Scatter(x=t_ds, y=left_ds,  mode="lines", name="Left",  line=dict(color="#00d4ff", width=0.8), opacity=0.85), row=2, col=1)
    fig.add_trace(go.Scatter(x=t_ds, y=right_ds, mode="lines", name="Right", line=dict(color="#ff6b35", width=0.8), opacity=0.85), row=2, col=1)
    fig.add_trace(go.Scatter(x=beat_times, y=beat_env, mode="lines", name="L-R envelope diff",
                             line=dict(color="#a78bfa", width=1.5), fill="tozeroy", fillcolor="rgba(167,139,250,0.15)"), row=2, col=2)
    fig.add_trace(go.Scatter(x=depth_times, y=depth_left,  mode="lines", name="Depth Left",  line=dict(color="#00d4ff", width=1.5)), row=3, col=1)
    fig.add_trace(go.Scatter(x=depth_times, y=depth_right, mode="lines", name="Depth Right", line=dict(color="#ff6b35", width=1.5)), row=3, col=2)

    fig.update_layout(
        title=dict(
            text=(
                f"Tarang 1.0.0.1 — Admin Audio Analysis Report<br>"
                f"<sup>{document_title} | State: {state_info['label']} "
                f"({state_info['freq']} {state_info['band']}) | "
                f"Duration: {duration:.1f}s | "
                f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}</sup>"
            ),
            font=dict(size=16, color="#e2e8f0")
        ),
        paper_bgcolor="#0f172a", plot_bgcolor="#1e293b",
        font=dict(color="#cbd5e1", family="monospace"),
        height=1100, showlegend=True,
        legend=dict(bgcolor="rgba(15,23,42,0.8)", bordercolor="#334155", borderwidth=1, font=dict(color="#cbd5e1")),
    )
    fig.update_xaxes(gridcolor="#1e293b", zerolinecolor="#334155")
    fig.update_yaxes(gridcolor="#1e293b", zerolinecolor="#334155")

    html_str = pio.to_html(fig, full_html=True, include_plotlyjs=True)
    logger.info(f"Admin report HTML generated | {len(html_str):,} chars")
    return html_str


# ── User neon waveform ────────────────────────────────────────────────────────

def generate_user_waveform_html(
    wav_path: Path,
    cognitive_state: str,
    beat_freq_hz: float,
    document_title: str = "Tarang Session",
    session_number: int = 1,
) -> str:
    """Generate animated neon waveform. Returns HTML string."""
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

    n_env    = 200
    mono     = (left + right) / 2
    step     = len(mono) // n_env
    envelope = []
    for i in range(n_env):
        chunk = mono[i * step: (i + 1) * step]
        rms   = float(np.sqrt(np.mean(chunk ** 2))) if len(chunk) > 0 else 0.0
        envelope.append(round(rms, 4))
    max_env  = max(envelope) if max(envelope) > 0 else 1.0
    envelope = [round(v / max_env, 4) for v in envelope]
    env_js   = json.dumps(envelope)

    state_colors = {
        "deep_focus":      {"primary": "#00d4ff", "glow": "0,212,255"},
        "memory":          {"primary": "#a78bfa", "glow": "167,139,250"},
        "calm":            {"primary": "#34d399", "glow": "52,211,153"},
        "deep_relaxation": {"primary": "#fb923c", "glow": "251,146,60"},
        "sleep":           {"primary": "#818cf8", "glow": "129,140,248"},
    }
    colors = state_colors.get(cognitive_state, {"primary": "#00d4ff", "glow": "0,212,255"})

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Tarang — {state_info['label']} Session</title>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ background:#050914; color:#e2e8f0; font-family:'Segoe UI',system-ui,sans-serif;
       display:flex; flex-direction:column; align-items:center; justify-content:center; min-height:100vh; overflow:hidden; }}
.container {{ width:100%; max-width:900px; padding:2rem; display:flex; flex-direction:column; align-items:center; gap:1.5rem; }}
.header {{ text-align:center; }}
.brand {{ font-size:0.75rem; letter-spacing:0.3em; color:{colors['primary']}; text-transform:uppercase; opacity:0.7; }}
h1 {{ font-size:1.8rem; font-weight:300; color:#f1f5f9; margin:0.25rem 0; letter-spacing:0.05em; }}
.subtitle {{ font-size:0.85rem; color:#64748b; }}
.state-badge {{ display:inline-flex; align-items:center; gap:0.5rem; padding:0.4rem 1rem;
               border:1px solid {colors['primary']}44; border-radius:999px;
               background:{colors['primary']}11; font-size:0.8rem; color:{colors['primary']}; letter-spacing:0.1em; }}
.dot {{ width:6px; height:6px; border-radius:50%; background:{colors['primary']}; animation:pulse 2s ease-in-out infinite; }}
.canvas-wrapper {{ width:100%; border:1px solid {colors['primary']}22; border-radius:12px; background:#080f1f; overflow:hidden; }}
canvas {{ display:block; width:100%; height:180px; }}
.info-grid {{ display:grid; grid-template-columns:repeat(3,1fr); gap:1rem; width:100%; }}
.info-card {{ background:#0d1526; border:1px solid #1e293b; border-radius:10px; padding:1rem; text-align:center; }}
.info-card .label {{ font-size:0.7rem; letter-spacing:0.15em; color:#475569; text-transform:uppercase; margin-bottom:0.3rem; }}
.info-card .value {{ font-size:1.1rem; font-weight:500; color:{colors['primary']}; }}
.footer-note {{ font-size:0.72rem; color:#334155; text-align:center; letter-spacing:0.05em; }}
@keyframes pulse {{ 0%,100%{{opacity:1;transform:scale(1)}} 50%{{opacity:0.4;transform:scale(0.85)}} }}
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <div class="brand">Tarang 1.0.0.1 &nbsp;|&nbsp; Neuro-Acoustic Learning</div>
    <h1>{document_title}</h1>
    <div class="subtitle">Session {session_number} &nbsp;&middot;&nbsp; Audio Waveform</div>
  </div>
  <div class="state-badge">
    <span class="dot"></span>
    {state_info['label']} &nbsp;&middot;&nbsp; {state_info['freq']} {state_info['band']}
  </div>
  <div class="canvas-wrapper"><canvas id="waveCanvas"></canvas></div>
  <div class="info-grid">
    <div class="info-card">
      <div class="label">Duration</div>
      <div class="value">{int(duration//60)}m {int(duration%60)}s</div>
    </div>
    <div class="info-card">
      <div class="label">Cognitive State</div>
      <div class="value">{state_info['label']}</div>
    </div>
    <div class="info-card">
      <div class="label">Session</div>
      <div class="value">#{session_number}</div>
    </div>
  </div>
  <div class="footer-note">
    Use headphones for full neuro-acoustic effect &nbsp;&middot;&nbsp;
    Generated {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}
  </div>
</div>
<script>
const canvas = document.getElementById('waveCanvas');
const ctx    = canvas.getContext('2d');
const PRIMARY = '{colors['primary']}';
const GLOW    = '{colors['glow']}';
const envelope = {env_js};
let frame = 0, offset = 0;
function resize() {{ canvas.width = canvas.offsetWidth; canvas.height = canvas.offsetHeight; }}
resize();
window.addEventListener('resize', resize);
function lerp(a,b,t) {{ return a + (b-a)*t; }}
function getEnvAt(x) {{
  const pos  = (x / canvas.width) * envelope.length + offset;
  const idx  = Math.floor(pos) % envelope.length;
  const next = (idx + 1) % envelope.length;
  return lerp(envelope[idx], envelope[next], pos - Math.floor(pos));
}}
function drawWave(yCenter, amplitude, alpha, lineWidth) {{
  ctx.beginPath();
  ctx.lineWidth   = lineWidth;
  ctx.strokeStyle = `rgba(${{GLOW}},${{alpha}})`;
  ctx.shadowColor = PRIMARY;
  ctx.shadowBlur  = lineWidth * 8;
  for (let x = 0; x <= canvas.width; x += 2) {{
    const env  = getEnvAt(x);
    const sine = Math.sin((x / canvas.width) * Math.PI * 6 + frame * 0.04);
    const y    = yCenter + sine * amplitude * env * 0.6
               + Math.sin(frame * 0.02 + x * 0.01) * amplitude * env * 0.3;
    if (x === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }}
  ctx.stroke();
  ctx.shadowBlur = 0;
}}
function drawFrame() {{
  const W = canvas.width, H = canvas.height, cy = H / 2;
  ctx.fillStyle = 'rgba(8,15,31,0.45)';
  ctx.fillRect(0, 0, W, H);
  drawWave(cy, H*0.38, 0.15, 6);
  drawWave(cy, H*0.34, 0.35, 2.5);
  drawWave(cy, H*0.30, 0.90, 1.2);
  drawWave(cy, -H*0.18, 0.08, 3);
  frame  += 1;
  offset += 0.12;
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
    document_title: str = "Tarang Session",
    session_number: int = 1,
    output_stem: str = None,
) -> dict:
    """
    STATELESS: Returns HTML as string in result["html_content"].
    No local files written. Express stores html_content in MongoDB.

    Returns:
        {
            status, role, report_type,
            html_content,   ← stored in MongoDB by Express
            output_path,    ← always None
            timestamp
        }
    """
    modulated_wav_path = Path(modulated_wav_path)
    role = role.lower().strip()

    if not modulated_wav_path.exists():
        return {"status": "error", "error": f"Audio file not found: {modulated_wav_path}"}
    if role not in ("admin", "user"):
        return {"status": "error", "error": f"Invalid role '{role}'. Use 'admin' or 'user'."}

    if role == "admin":
        report_type  = "admin_report"
        logger.info(f"Generating ADMIN report for: {modulated_wav_path.name}")
        html_content = generate_admin_html(modulated_wav_path, cognitive_state, beat_freq_hz, document_title)
    else:
        report_type  = "user_waveform"
        logger.info(f"Generating USER waveform for: {modulated_wav_path.name}")
        html_content = generate_user_waveform_html(modulated_wav_path, cognitive_state, beat_freq_hz, document_title, session_number)

    if not html_content:
        return {"status": "error", "error": "Visualization generation failed. Check logs."}

    logger.info(f"Visualization complete | type={report_type}")

    return {
        "status":       "success",
        "role":         role,
        "output_path":  None,          # no local file — kept for bridge compatibility
        "html_content": html_content,  # returned to bridge → Express stores in MongoDB
        "report_type":  report_type,
        "timestamp":    datetime.utcnow().isoformat() + "Z",
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    print("\nTarang 1.0.0.1 — Audio Visualizer")
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
        print("  (HTML returned as string — no local file written)")
    else:
        print(f"\n✗ {r['error']}")
        sys.exit(1)

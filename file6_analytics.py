"""
Tarang 1.0.0.1 — file6_analytics.py
STATELESS: No local file writes whatsoever.
Accepts session_state dict (from MongoDB via bridge).
Returns analytics dict + HTML string.
Express stores both in MongoDB.
"""

import json
import logging
from datetime import datetime
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [file6_analytics] %(levelname)s — %(message)s"
)
logger = logging.getLogger("file6_analytics")

DIFFICULTY_LABELS = {1: "Easy", 2: "Medium", 3: "Hard"}

COGNITIVE_MODE_LABELS = {
    "deep_focus":      "Deep Focus (14 Hz Beta)",
    "memory":          "Memory (10 Hz Alpha)",
    "calm":            "Calm (8 Hz Alpha)",
    "deep_relaxation": "Deep Relaxation (6 Hz Theta)",
    "sleep":           "Sleep (4 Hz Theta/Delta)",
}

EXCELLENT_THRESHOLD = 0.80
GOOD_THRESHOLD      = 0.60
AVERAGE_THRESHOLD   = 0.40

BRAIN_FACTS = [
    "The human brain has about 86 billion neurons.",
    "Binaural beats were first discovered by Heinrich Wilhelm Dove in 1839.",
    "Beta waves (13–30 Hz) are associated with active thinking and focus.",
    "Alpha waves (8–12 Hz) are linked to relaxed, calm awareness.",
    "Theta waves (4–8 Hz) are associated with deep relaxation and creativity.",
    "The brain can process images seen for as little as 13 milliseconds.",
    "Spaced repetition is one of the most evidence-backed learning techniques.",
    "Sleep consolidates memories — studying before sleep improves retention.",
    "The hippocampus plays a key role in converting short-term to long-term memory.",
    "Active recall is more effective than passive re-reading for retention.",
    "Music at 60 BPM can synchronise brainwaves to Alpha state.",
    "The Pomodoro technique aligns with natural ultradian rhythms of focus.",
    "Interleaving different topics during study improves long-term retention.",
    "The brain uses about 20% of the body's total energy despite being 2% of body weight.",
    "Neuroplasticity allows the brain to rewire itself throughout life.",
    "Flow states are associated with decreased activity in the prefrontal cortex.",
    "Testing yourself on material strengthens memory traces.",
    "The forgetting curve shows 70% of new info is lost within 24 hours without review.",
    "Binaural beats require headphones — speakers cancel the frequency difference.",
    "Delta waves (0.5–4 Hz) dominate during deep dreamless sleep.",
]


# ── Utilities ─────────────────────────────────────────────────────────────────

def now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"

def parse_iso(ts: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).replace(tzinfo=None)
    except Exception:
        return None

def duration_minutes(start_ts: str, end_ts: str) -> Optional[float]:
    s = parse_iso(start_ts)
    e = parse_iso(end_ts)
    if s and e:
        return round((e - s).total_seconds() / 60.0, 2)
    return None

def score_label(pct: float) -> str:
    if pct >= EXCELLENT_THRESHOLD:  return "Excellent"
    elif pct >= GOOD_THRESHOLD:     return "Good"
    elif pct >= AVERAGE_THRESHOLD:  return "Average"
    else:                           return "Needs Improvement"

def improvement_pct(s1: float, s_n: float) -> float:
    if s1 == 0:
        return 0.0
    return round(((s_n - s1) / s1) * 100, 1)


# ── Weak topic extraction ─────────────────────────────────────────────────────

def extract_weak_topics(session_state: dict, all_questions: dict, all_answer_keys: dict) -> list:
    """
    STATELESS: reads from dicts passed in (from MongoDB).

    session_state:   full state dict
    all_questions:   {1: {questions:[...]}, 2: {...}, 3: {...}}
    all_answer_keys: {1: {answers:{...}}, 2: {...}, 3: {...}}

    Returns ranked list of weak concept strings.
    """
    wrong_concepts = {}

    for session in [1, 2, 3]:
        q_data   = all_questions.get(session, {})
        a_data   = all_answer_keys.get(session, {})
        questions    = q_data.get("questions", [])
        answer_key   = a_data.get("answers", {})
        user_answers = session_state["sessions"][str(session)].get("user_answers", {})

        concept_map = {
            q["question_id"]: q.get("source_concept", "").strip()
            for q in questions
        }

        if user_answers:
            # Exact per-question wrong answer detection
            for q_id, key_data in answer_key.items():
                correct  = sorted(key_data["correct_answers"])
                answered = sorted(user_answers.get(q_id, []))
                if answered != correct:
                    concept = concept_map.get(q_id, "")
                    if concept:
                        wrong_concepts[concept] = wrong_concepts.get(concept, 0) + 1
        else:
            # Fallback: score proxy for sessions without stored answers
            score_pct = session_state["sessions"][str(session)].get("score_pct", 0) or 0
            if score_pct < 0.70:
                for q in questions:
                    concept = q.get("source_concept", "").strip()
                    if concept:
                        wrong_concepts[concept] = wrong_concepts.get(concept, 0) + 1

    sorted_weak = sorted(wrong_concepts.items(), key=lambda x: x[1], reverse=True)
    return [c for c, _ in sorted_weak[:10]]


# ── Suggestions ───────────────────────────────────────────────────────────────

def generate_suggestions(avg_score: float, scores: list, weak_topics: list,
                          relistening_recommended: bool, improvement: float) -> list:
    suggestions = []

    if relistening_recommended:
        suggestions.append(
            "Your average score is below 30%. We recommend listening to the audio again "
            "before retaking the sessions. Try 'Memory (10 Hz Alpha)' mode for better retention."
        )

    if avg_score < AVERAGE_THRESHOLD:
        suggestions.append("Focus on the weak topics below. Re-read the source material before your next attempt.")
    elif avg_score < GOOD_THRESHOLD:
        suggestions.append("You have a basic grasp of the content. Revisiting the audio and taking notes should push your score above 60%.")
    elif avg_score < EXCELLENT_THRESHOLD:
        suggestions.append("Good progress! To reach Excellent level, focus on the multiple-correct questions — these test deeper understanding.")
    else:
        suggestions.append("Excellent work! Consider exploring advanced topics related to this subject to continue your learning journey.")

    if improvement > 20:
        suggestions.append(f"Strong improvement of {improvement}% from Session 1 to Session 3 — the spaced repetition approach is working well.")
    elif improvement < 0:
        suggestions.append("Your score declined from Session 1 to Session 3. Focus on the weak topics listed and revisit the material with active recall techniques.")

    if weak_topics:
        suggestions.append(f"Weak areas identified: {', '.join(weak_topics[:3])}. Spend extra time reviewing these concepts specifically.")

    if not suggestions:
        suggestions.append("Keep up the consistent practice. Spaced repetition over multiple sessions is the most effective way to build long-term retention.")

    return suggestions


# ── Core analytics computation ────────────────────────────────────────────────

def compute_analytics(
    doc_id: str,
    role: str,
    session_state: dict,
    all_questions: dict,
    all_answer_keys: dict,
) -> dict:
    """
    STATELESS: All data comes from dicts (no local file reads).

    session_state:   full state dict from MongoDB
    all_questions:   {1: questions_dict, 2: ..., 3: ...}
    all_answer_keys: {1: answers_dict,   2: ..., 3: ...}
    """
    state  = session_state
    s_data = state["sessions"]
    admin  = str(role).strip().lower() == "admin"

    if not state.get("all_sessions_complete"):
        done = sum(1 for s in ["1","2","3"] if s_data[s]["status"] == "completed")
        return {"status": "error", "error": "Analytics available only after all 3 sessions are completed.", "sessions_completed": done}

    # Scores
    scores = [round((s_data[str(i)]["score_pct"] or 0) * 100, 1) for i in [1,2,3]]
    avg_score     = round(sum(scores) / 3, 1)
    best_session  = scores.index(max(scores)) + 1
    worst_session = scores.index(min(scores)) + 1
    s1_to_s3_imp  = improvement_pct(s_data["1"]["score_pct"] or 0, s_data["3"]["score_pct"] or 0)

    # Time spent
    time_spent = {}
    total_time  = 0.0
    for i in [1, 2, 3]:
        started   = s_data[str(i)].get("started_at")
        submitted = s_data[str(i)].get("submitted_at")
        mins      = duration_minutes(started, submitted) if started and submitted else None
        time_spent[f"session_{i}"] = mins
        if mins:
            total_time += mins

    cognitive_states = state.get("cognitive_states", {})

    # Weak topics from actual wrong answers
    weak_topics = extract_weak_topics(state, all_questions, all_answer_keys)

    # Session details
    session_details = {}
    for i in [1, 2, 3]:
        s      = s_data[str(i)]
        detail = {
            "session":        i,
            "difficulty":     DIFFICULTY_LABELS[i],
            "score_pct":      s["score_pct"],
            "score_display":  f"{round((s['score_pct'] or 0) * 100, 1)}%",
            "score_label":    score_label(s["score_pct"] or 0),
            "time_spent_min": time_spent.get(f"session_{i}"),
            "started_at":     s.get("started_at"),
            "submitted_at":   s.get("submitted_at"),
            "override_used":  s.get("override_used", False),
        }
        if admin:
            detail["cognitive_state"] = cognitive_states.get(str(i), "not recorded")
        session_details[f"session_{i}"] = detail

    suggestions = generate_suggestions(
        avg_score=avg_score / 100, scores=scores, weak_topics=weak_topics,
        relistening_recommended=state.get("relistening_recommended", False),
        improvement=s1_to_s3_imp,
    )

    # Learning curve
    if scores[2] > scores[0] and scores[2] > scores[1]:
        curve      = "improving"
        curve_desc = "Your scores improved with each session — the spaced learning approach is working."
    elif scores[2] < scores[0]:
        curve      = "declining"
        curve_desc = "Scores declined by Session 3. Harder questions + possible fatigue — revisit weak topics."
    elif abs(scores[2] - scores[0]) <= 5:
        curve      = "stable"
        curve_desc = "Consistent performance across sessions. Consider deeper engagement with the material."
    else:
        curve      = "variable"
        curve_desc = "Mixed performance across sessions — focus on the weak topics listed."

    analytics = {
        "status":         "success",
        "document_id":    doc_id,
        "document_title": state.get("document_title", ""),
        "role":           role,
        "generated_at":   now_iso(),
        "summary": {
            "average_score_pct":       round(avg_score / 100, 4),
            "average_score_display":   f"{avg_score}%",
            "average_score_label":     score_label(avg_score / 100),
            "best_session":            best_session,
            "worst_session":           worst_session,
            "total_time_spent_min":    round(total_time, 2),
            "improvement_s1_to_s3":    f"{s1_to_s3_imp:+.1f}%",
            "learning_curve":          curve,
            "learning_curve_desc":     curve_desc,
            "relistening_recommended": state.get("relistening_recommended", False),
            "poor_score_warning":      state.get("poor_score_warning", False),
        },
        "session_details":   session_details,
        "score_progression": [{"session": i, "difficulty": DIFFICULTY_LABELS[i], "score": scores[i-1]} for i in [1,2,3]],
        "weak_topics":       weak_topics,
        "suggestions":       suggestions,
        "brain_facts":       BRAIN_FACTS,
    }

    if admin:
        analytics["admin_details"] = {
            "s1_to_s2_hours": state.get("s1_to_s2_hours"),
            "s2_to_s3_hours": state.get("s2_to_s3_hours"),
            "num_questions":  state.get("num_questions"),
            "created_at":     state.get("created_at"),
            "sessions_meta":  state.get("sessions_meta", {}),
        }

    return analytics


# ── HTML report generator ─────────────────────────────────────────────────────

def generate_html_report(analytics: dict) -> str:
    """Generate standalone HTML analytics report. Returns HTML string."""
    doc_title  = analytics.get("document_title", "Tarang Session")
    summary    = analytics["summary"]
    sessions   = analytics["session_details"]
    progression= analytics["score_progression"]
    weak       = analytics["weak_topics"]
    suggestions= analytics["suggestions"]
    role       = analytics.get("role", "user")
    admin      = role == "admin"
    generated  = analytics.get("generated_at", now_iso())

    avg_score    = round(summary["average_score_pct"] * 100, 1)
    curve        = summary["learning_curve"]
    curve_desc   = summary["learning_curve_desc"]
    imp_str      = summary.get("improvement_s1_to_s3", "+0.0%")
    s1_to_s3_imp = float(imp_str.replace("%", "").replace("+", ""))

    def bar(score):
        color = "#34d399" if score >= 80 else "#60a5fa" if score >= 60 else "#fbbf24" if score >= 40 else "#f87171"
        return score, color

    s1 = bar(progression[0]["score"])
    s2 = bar(progression[1]["score"])
    s3 = bar(progression[2]["score"])

    curve_icon  = {"improving":"▲","declining":"▼","stable":"→","variable":"~"}.get(curve,"→")
    curve_color = {"improving":"#34d399","declining":"#f87171","stable":"#60a5fa","variable":"#fbbf24"}.get(curve,"#94a3b8")

    weak_html = "".join(
        f'<span style="display:inline-block;padding:3px 10px;margin:4px;background:#1e3a5f;'
        f'border:1px solid #2563eb44;border-radius:20px;font-size:0.78rem;color:#93c5fd">{w}</span>'
        for w in (weak[:8] if weak else ["No weak topics identified"])
    )
    sugg_html = "".join(
        f'<div style="padding:10px 14px;margin:8px 0;background:#0f2027;border-left:3px solid #6366f1;'
        f'border-radius:0 8px 8px 0;font-size:0.85rem;color:#c7d2fe;line-height:1.6">{s}</div>'
        for s in suggestions
    )

    def session_row(sn, label, data):
        sc    = round((data["score_pct"] or 0) * 100, 1)
        tm    = f"{data['time_spent_min']} min" if data["time_spent_min"] else "—"
        lbl   = data["score_label"]
        color = "#34d399" if sc >= 80 else "#60a5fa" if sc >= 60 else "#fbbf24" if sc >= 40 else "#f87171"
        extra = ""
        if admin and "cognitive_state" in data:
            cs_lbl = COGNITIVE_MODE_LABELS.get(data["cognitive_state"], data["cognitive_state"])
            extra  = f'<td style="color:#94a3b8;font-size:0.78rem">{cs_lbl}</td>'
        return (
            f'<tr style="border-bottom:1px solid #1e293b">'
            f'<td style="padding:10px 14px;color:#e2e8f0">Session {sn} — {label}</td>'
            f'<td style="padding:10px 14px"><span style="color:{color};font-weight:500">{sc}%</span>'
            f'<span style="color:#475569;font-size:0.78rem;margin-left:8px">({lbl})</span></td>'
            f'<td style="padding:10px 14px;color:#94a3b8">{tm}</td>{extra}</tr>'
        )

    rows_html = (session_row(1,"Easy",sessions["session_1"]) +
                 session_row(2,"Medium",sessions["session_2"]) +
                 session_row(3,"Hard",sessions["session_3"]))
    admin_col_hdr = '<th style="padding:10px 14px;color:#64748b;font-weight:400">Cognitive mode</th>' if admin else ""

    admin_section = ""
    if admin and "admin_details" in analytics:
        ad = analytics["admin_details"]
        meta_rows = "".join(
            f'<tr><td style="color:#64748b;padding:6px 12px">{k}</td><td style="color:#94a3b8;padding:6px 12px">{v}</td></tr>'
            for k, v in {"Questions per session": ad.get("num_questions"),
                         "S1→S2 gap (hours)": ad.get("s1_to_s2_hours"),
                         "S2→S3 gap (hours)": ad.get("s2_to_s3_hours"),
                         "Document created": ad.get("created_at","")[:10]}.items()
        )
        admin_section = f'<div style="margin-top:2rem;padding:1.2rem;background:#0d1526;border:1px solid #1e293b;border-radius:10px"><div style="font-size:0.7rem;letter-spacing:0.15em;color:#6366f1;text-transform:uppercase;margin-bottom:0.8rem">Admin details</div><table style="width:100%;border-collapse:collapse;font-size:0.82rem">{meta_rows}</table></div>'

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Tarang Analytics — {doc_title}</title>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ background:#050914; color:#e2e8f0; font-family:'Segoe UI',system-ui,sans-serif; min-height:100vh; padding:2rem 1rem; }}
.container {{ max-width:860px; margin:0 auto; }}
.card {{ background:#0d1526; border:1px solid #1e293b; border-radius:12px; padding:1.5rem; margin-bottom:1.2rem; }}
.section-label {{ font-size:0.7rem; letter-spacing:0.2em; color:#6366f1; text-transform:uppercase; margin-bottom:0.8rem; }}
.bar-track {{ background:#1e293b; border-radius:999px; height:10px; width:100%; margin:6px 0; }}
.bar-fill {{ height:10px; border-radius:999px; }}
table {{ width:100%; border-collapse:collapse; }}
th {{ text-align:left; padding:10px 14px; color:#64748b; font-weight:400; border-bottom:1px solid #1e293b; font-size:0.82rem; }}
</style>
</head>
<body>
<div class="container">
  <div style="text-align:center;margin-bottom:2rem">
    <div style="font-size:0.72rem;letter-spacing:0.3em;color:#6366f1;text-transform:uppercase;margin-bottom:0.3rem">Tarang 1.0.0.1 &nbsp;·&nbsp; Neuro-Acoustic Learning</div>
    <h1 style="font-size:1.6rem;font-weight:300;color:#f1f5f9">{doc_title}</h1>
    <div style="font-size:0.8rem;color:#475569;margin-top:0.3rem">Analytics Report &nbsp;·&nbsp; {role.title()} View &nbsp;·&nbsp; {generated[:10]}</div>
  </div>
  <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;margin-bottom:1.2rem">
    <div class="card" style="text-align:center">
      <div class="section-label">Average score</div>
      <div style="font-size:2rem;font-weight:300;color:#60a5fa">{avg_score}%</div>
      <div style="font-size:0.78rem;color:#475569;margin-top:4px">{summary['average_score_label']}</div>
    </div>
    <div class="card" style="text-align:center">
      <div class="section-label">Learning curve</div>
      <div style="font-size:2rem;color:{curve_color}">{curve_icon}</div>
      <div style="font-size:0.78rem;color:#475569;margin-top:4px;text-transform:capitalize">{curve}</div>
    </div>
    <div class="card" style="text-align:center">
      <div class="section-label">S1 → S3 change</div>
      <div style="font-size:2rem;font-weight:300;color:{'#34d399' if s1_to_s3_imp >= 0 else '#f87171'}">{summary['improvement_s1_to_s3']}</div>
      <div style="font-size:0.78rem;color:#475569;margin-top:4px">improvement</div>
    </div>
  </div>
  <div class="card">
    <div class="section-label">Score progression</div>
    <div style="margin-bottom:0.5rem">
      <div style="display:flex;justify-content:space-between;font-size:0.8rem;color:#64748b;margin-bottom:4px"><span>Session 1 — Easy</span><span style="color:{s1[1]}">{s1[0]}%</span></div>
      <div class="bar-track"><div class="bar-fill" style="width:{s1[0]}%;background:{s1[1]}"></div></div>
    </div>
    <div style="margin-bottom:0.5rem">
      <div style="display:flex;justify-content:space-between;font-size:0.8rem;color:#64748b;margin-bottom:4px"><span>Session 2 — Medium</span><span style="color:{s2[1]}">{s2[0]}%</span></div>
      <div class="bar-track"><div class="bar-fill" style="width:{s2[0]}%;background:{s2[1]}"></div></div>
    </div>
    <div>
      <div style="display:flex;justify-content:space-between;font-size:0.8rem;color:#64748b;margin-bottom:4px"><span>Session 3 — Hard</span><span style="color:{s3[1]}">{s3[0]}%</span></div>
      <div class="bar-track"><div class="bar-fill" style="width:{s3[0]}%;background:{s3[1]}"></div></div>
    </div>
    <div style="margin-top:1rem;font-size:0.82rem;color:#64748b;border-top:1px solid #1e293b;padding-top:0.8rem">{curve_desc}</div>
  </div>
  <div class="card">
    <div class="section-label">Session breakdown</div>
    <table><thead><tr><th>Session</th><th>Score</th><th>Time spent</th>{admin_col_hdr}</tr></thead><tbody>{rows_html}</tbody></table>
  </div>
  <div class="card">
    <div class="section-label">Weak topics identified</div>
    <div style="margin-top:0.4rem">{weak_html}</div>
    <div style="font-size:0.75rem;color:#334155;margin-top:0.8rem">Based on questions answered incorrectly across all sessions.</div>
  </div>
  <div class="card">
    <div class="section-label">Suggested next steps</div>
    {sugg_html}
  </div>
  {admin_section}
  <div style="text-align:center;font-size:0.72rem;color:#1e293b;margin-top:2rem;padding-bottom:1rem">Tarang 1.0.0.1 &nbsp;·&nbsp; Generated {generated[:19].replace('T',' ')} UTC</div>
</div>
</body>
</html>"""

    logger.info(f"Analytics HTML generated | {len(html):,} chars")
    return html


# ── Main entry point ──────────────────────────────────────────────────────────

def generate_analytics(
    doc_id: str,
    role: str = "user",
    session_state: dict = None,
    all_questions: dict = None,
    all_answer_keys: dict = None,
    cognitive_states: dict = None,
) -> dict:
    """
    STATELESS entry point.
    All data passed in as dicts (from MongoDB via bridge).
    Returns analytics dict + HTML string. No files written.

    Args:
        doc_id:          Document ID
        role:            "user" or "admin"
        session_state:   Full session state dict from MongoDB
        all_questions:   {1: questions_dict, 2: ..., 3: ...}
        all_answer_keys: {1: answers_dict, 2: ..., 3: ...}
        cognitive_states: Optional {"1": "deep_focus", "2": "memory", ...}

    Returns:
        {
            status, doc_id, role,
            analytics,      ← full analytics dict (Express stores in MongoDB)
            html_content,   ← HTML string (Express stores in MongoDB)
        }
    """
    if not session_state:
        return {"status": "error", "error": "session_state not provided."}

    logger.info(f"Generating analytics | doc_id={doc_id} | role={role}")

    # Optionally attach cognitive states to state (not persisted here — bridge handles)
    if cognitive_states:
        session_state = json.loads(json.dumps(session_state))
        session_state["cognitive_states"] = cognitive_states

    analytics = compute_analytics(
        doc_id=doc_id, role=role,
        session_state=session_state,
        all_questions=all_questions or {},
        all_answer_keys=all_answer_keys or {},
    )

    if analytics.get("status") == "error":
        return analytics

    html_content = generate_html_report(analytics)
    logger.info(f"Analytics complete | doc_id={doc_id} | weak_topics={len(analytics['weak_topics'])}")

    return {
        "status":       "success",
        "doc_id":       doc_id,
        "role":         role,
        "analytics":    analytics,    # Express stores in MongoDB Document
        "html_content": html_content, # Express stores in MongoDB Document
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    print("\nTarang 1.0.0.1 — Analytics Engine (stateless)")
    print("This file requires session_state to be passed in from bridge.")
    print("Use bridge.py to trigger analytics via the /analytics endpoint.")

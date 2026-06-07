"""
Tarang 2.0.0 — file6_analytics.py
=====================================
STATELESS: No local file writes whatsoever.
Accepts session_state dict (from MongoDB via bridge).
Returns analytics dict + HTML string.
Express stores both in MongoDB.

v2.0.0 — Weighted Retention Score (replaces flat average):
  PROBLEM with flat average:
    - S1 is Easy (right after audio) → naturally high score
    - S2 is Medium (12h later, some forgetting) → naturally lower
    - S3 is Hard (24h later, harder questions) → naturally lowest
    - Result: flat average always penalises users unfairly for
      expected cognitive science patterns. Demotivating.

  NEW: Weighted Retention Score (WRS)
    - Accounts for difficulty scaling (Easy/Medium/Hard)
    - Accounts for time-based forgetting (spaced repetition pattern)
    - Rewards improvement trajectories
    - Compares the user to their own baseline, not an absolute scale
    - A user scoring 80/60/50 can still have an EXCELLENT WRS if
      their retention across time is strong relative to difficulty increase

  Formula:
    WRS = (s1_adj * w1 + s2_adj * w2 + s3_adj * w3) / total_weight

    Where:
      s1_adj = s1_raw (baseline, no adjustment)
      s2_adj = s2_raw / DIFFICULTY_MULTIPLIER[2]  (normalise for medium difficulty)
      s3_adj = s3_raw / DIFFICULTY_MULTIPLIER[3]  (normalise for hard difficulty)

      w1 = 0.20  (Easy baseline — low weight, expected to be high)
      w2 = 0.35  (Medium after gap — moderate weight)
      w3 = 0.45  (Hard after long gap — highest weight, best predictor of retention)

    Retention Decay Bonus:
      If s2 >= s1 * 0.85 (user retained 85%+ from S1 to S2) → +5% bonus
      If s3 >= s2 * 0.80 (user retained 80%+ from S2 to S3) → +5% bonus

    Improvement Bonus:
      If s3_adj > s1_adj → user improved through difficulty → +0–10% bonus

    WRS is capped at 100%.

  Result labels use WRS, not flat average:
    Excellent:  WRS >= 75%  (strong retention despite difficulty)
    Good:       WRS >= 55%
    Average:    WRS >= 40%
    Developing: WRS >= 25%
    Needs Work: WRS < 25%
"""

import json
import logging
from datetime import datetime
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [file6_analytics v2.0.0] %(levelname)s — %(message)s"
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

# ── Weighted Retention Score parameters ──────────────────────────────────────

WRS_WEIGHTS = {1: 0.20, 2: 0.35, 3: 0.45}

# Difficulty normalisation: Hard is ~1.5x harder than Easy
# So a 60% on Hard ≈ 90% on Easy in terms of knowledge demonstrated
DIFFICULTY_MULTIPLIER = {1: 1.0, 2: 0.85, 3: 0.70}

RETENTION_DECAY_BONUS = 0.05   # +5% if user retained well S1→S2
RETENTION_DECAY_BONUS_2 = 0.05 # +5% if user retained well S2→S3
MAX_IMPROVEMENT_BONUS = 0.10   # up to +10% for improvement trajectory

# WRS thresholds (different from old flat-average thresholds)
WRS_EXCELLENT  = 0.75
WRS_GOOD       = 0.55
WRS_AVERAGE    = 0.40
WRS_DEVELOPING = 0.25

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

def wrs_label(wrs: float) -> str:
    if wrs >= WRS_EXCELLENT:   return "Excellent Retention"
    elif wrs >= WRS_GOOD:      return "Good Retention"
    elif wrs >= WRS_AVERAGE:   return "Average Retention"
    elif wrs >= WRS_DEVELOPING:return "Developing"
    else:                      return "Needs Improvement"

def improvement_pct(s1: float, s_n: float) -> float:
    if s1 == 0:
        return 0.0
    return round(((s_n - s1) / s1) * 100, 1)


# ── Weighted Retention Score computation ──────────────────────────────────────

def compute_wrs(s1_pct: float, s2_pct: float, s3_pct: float) -> dict:
    """
    Compute the Weighted Retention Score (WRS) from the three session scores.

    Returns a dict with the WRS, component scores, bonuses, and explanation.

    Design rationale:
      - We don't punish users for expected difficulty-based score drops.
      - We reward stable or improving trajectories.
      - The score should tell the user "how well you retained knowledge over time"
        not "how well you did on increasingly hard tests".
    """
    # Difficulty-normalised scores
    s1_adj = s1_pct / DIFFICULTY_MULTIPLIER[1]   # = s1_pct (no change)
    s2_adj = s2_pct / DIFFICULTY_MULTIPLIER[2]   # ÷ 0.85
    s3_adj = s3_pct / DIFFICULTY_MULTIPLIER[3]   # ÷ 0.70

    # Cap normalised scores at 1.0
    s1_adj = min(1.0, s1_adj)
    s2_adj = min(1.0, s2_adj)
    s3_adj = min(1.0, s3_adj)

    # Weighted base score
    base_wrs = (
        s1_adj * WRS_WEIGHTS[1] +
        s2_adj * WRS_WEIGHTS[2] +
        s3_adj * WRS_WEIGHTS[3]
    )

    bonuses = {}

    # Retention decay bonus: did the user retain well between sessions?
    # S1→S2: expected natural drop is ~15-20% for medium difficulty
    decay_s1_s2 = s2_pct / s1_pct if s1_pct > 0 else 1.0
    decay_s2_s3 = s3_pct / s2_pct if s2_pct > 0 else 1.0

    if decay_s1_s2 >= 0.85:
        bonuses["retention_s1_s2"] = RETENTION_DECAY_BONUS
    if decay_s2_s3 >= 0.80:
        bonuses["retention_s2_s3"] = RETENTION_DECAY_BONUS_2

    # Improvement bonus: did difficulty-normalised score improve S1→S3?
    if s3_adj > s1_adj:
        improvement_ratio = min(1.0, (s3_adj - s1_adj) / max(s1_adj, 0.01))
        bonuses["improvement_trajectory"] = round(improvement_ratio * MAX_IMPROVEMENT_BONUS, 4)

    total_bonus = sum(bonuses.values())
    final_wrs   = min(1.0, base_wrs + total_bonus)

    # Explanation text for user
    explanation_parts = []
    if decay_s1_s2 >= 0.90:
        explanation_parts.append("You retained knowledge very well between Session 1 and 2.")
    elif decay_s1_s2 >= 0.75:
        explanation_parts.append("Some expected forgetting between sessions — this is normal.")
    else:
        explanation_parts.append("Significant forgetting detected between Session 1 and 2.")

    if s3_adj >= s1_adj:
        explanation_parts.append("Your performance actually improved as difficulty increased — strong retention.")
    elif s3_pct >= 0.50:
        explanation_parts.append(f"Session 3 (Hard) score of {round(s3_pct*100,1)}% is solid given the difficulty.")

    if total_bonus > 0:
        explanation_parts.append(
            f"Bonuses applied for {', '.join(bonuses.keys()).replace('_', ' ')} "
            f"({round(total_bonus*100,1)}% total)."
        )

    logger.info(
        f"WRS | s1={round(s1_pct*100,1)}% s2={round(s2_pct*100,1)}% s3={round(s3_pct*100,1)}% | "
        f"adj={round(s1_adj*100,1)}%/{round(s2_adj*100,1)}%/{round(s3_adj*100,1)}% | "
        f"base={round(base_wrs*100,1)}% bonus={round(total_bonus*100,1)}% | "
        f"final_wrs={round(final_wrs*100,1)}%"
    )

    return {
        "wrs":                   round(final_wrs, 4),
        "wrs_display":           f"{round(final_wrs * 100, 1)}%",
        "wrs_label":             wrs_label(final_wrs),
        "base_wrs":              round(base_wrs, 4),
        "bonuses":               bonuses,
        "total_bonus":           round(total_bonus, 4),
        "normalised_scores": {
            "s1_adj": round(s1_adj, 4),
            "s2_adj": round(s2_adj, 4),
            "s3_adj": round(s3_adj, 4),
        },
        "retention_decay": {
            "s1_to_s2": round(decay_s1_s2, 3),
            "s2_to_s3": round(decay_s2_s3, 3),
        },
        "explanation": " ".join(explanation_parts),
    }


# ── Weak topic extraction ─────────────────────────────────────────────────────

def extract_weak_topics(session_state: dict, all_questions: dict, all_answer_keys: dict) -> list:
    wrong_concepts = {}

    for session in [1, 2, 3]:
        q_data       = all_questions.get(session, {})
        a_data       = all_answer_keys.get(session, {})
        questions    = q_data.get("questions", [])
        answer_key   = a_data.get("answers", {})
        user_answers = session_state["sessions"][str(session)].get("user_answers", {})

        concept_map = {q["question_id"]: q.get("source_concept", "").strip() for q in questions}

        if user_answers:
            for q_id, key_data in answer_key.items():
                correct  = sorted(key_data["correct_answers"])
                answered = sorted(user_answers.get(q_id, []))
                if answered != correct:
                    concept = concept_map.get(q_id, "")
                    if concept:
                        wrong_concepts[concept] = wrong_concepts.get(concept, 0) + 1
        else:
            score_pct = session_state["sessions"][str(session)].get("score_pct", 0) or 0
            if score_pct < 0.70:
                for q in questions:
                    concept = q.get("source_concept", "").strip()
                    if concept:
                        wrong_concepts[concept] = wrong_concepts.get(concept, 0) + 1

    sorted_weak = sorted(wrong_concepts.items(), key=lambda x: x[1], reverse=True)
    return [c for c, _ in sorted_weak[:10]]


# ── Suggestions (WRS-aware) ───────────────────────────────────────────────────

def generate_suggestions(wrs_data: dict, scores: list, weak_topics: list,
                          relistening_recommended: bool) -> list:
    wrs   = wrs_data["wrs"]
    label = wrs_data["wrs_label"]
    decay = wrs_data["retention_decay"]
    suggestions = []

    if relistening_recommended:
        suggestions.append(
            "One or more sessions scored below 30%. We recommend listening again "
            "before retaking. Try 'Memory (10 Hz Alpha)' mode for better consolidation."
        )

    # WRS-based primary suggestion
    if wrs >= WRS_EXCELLENT:
        suggestions.append(
            f"Outstanding retention score of {wrs_data['wrs_display']}! "
            f"Your knowledge held up well across all three difficulty levels and time gaps. "
            f"You're ready to move on to more advanced material."
        )
    elif wrs >= WRS_GOOD:
        suggestions.append(
            f"Good retention score of {wrs_data['wrs_display']}. "
            f"Your knowledge is solid. To push further, focus on the weak topics below "
            f"and consider a fourth voluntary re-listen using Memory mode."
        )
    elif wrs >= WRS_AVERAGE:
        suggestions.append(
            f"Average retention ({wrs_data['wrs_display']}). "
            f"The content is partially consolidated. Re-listening before Session 2 or 3 "
            f"in Memory (10 Hz) mode can significantly improve your retention curve."
        )
    else:
        suggestions.append(
            f"Your retention score ({wrs_data['wrs_display']}) indicates the material "
            f"needs more reinforcement. We strongly recommend re-listening — try "
            f"Deep Focus mode for fresh absorption, then Memory mode for the review session."
        )

    # Decay-specific suggestions
    if decay["s1_to_s2"] < 0.70:
        suggestions.append(
            f"You lost {round((1 - decay['s1_to_s2'])*100, 0):.0f}% of Session 1 knowledge by Session 2. "
            f"Shortening the S1→S2 gap to 6–8 hours (instead of 12) may help preserve more."
        )
    if decay["s2_to_s3"] < 0.70:
        suggestions.append(
            f"Significant forgetting occurred between Session 2 and 3. "
            f"Try adding a brief 5-minute active recall exercise before Session 3."
        )

    # Bonus explanation
    if wrs_data["total_bonus"] > 0:
        bonus_items = [k.replace("_", " ") for k in wrs_data["bonuses"]]
        suggestions.append(
            f"Your WRS received a +{round(wrs_data['total_bonus']*100, 1)}% bonus "
            f"for: {', '.join(bonus_items)}. These bonuses reflect genuine retention strengths."
        )

    # Weak topics
    if weak_topics:
        suggestions.append(
            f"Focus review on these topics: {', '.join(weak_topics[:4])}. "
            f"Even 10 minutes of targeted re-reading can raise your next WRS significantly."
        )

    return suggestions


# ── Core analytics computation ────────────────────────────────────────────────

def compute_analytics(
    doc_id: str,
    role: str,
    session_state: dict,
    all_questions: dict,
    all_answer_keys: dict,
) -> dict:
    state  = session_state
    s_data = state["sessions"]
    admin  = str(role).strip().lower() == "admin"

    if not state.get("all_sessions_complete"):
        done = sum(1 for s in ["1","2","3"] if s_data[s]["status"] == "completed")
        return {"status": "error", "error": "Analytics available only after all 3 sessions.", "sessions_completed": done}

    # Raw scores (0–1 floats)
    s1_pct = float(s_data["1"]["score_pct"] or 0)
    s2_pct = float(s_data["2"]["score_pct"] or 0)
    s3_pct = float(s_data["3"]["score_pct"] or 0)
    scores_pct = [s1_pct, s2_pct, s3_pct]
    scores_100 = [round(p * 100, 1) for p in scores_pct]

    # Flat average (kept for reference/admin)
    flat_avg = round(sum(scores_pct) / 3, 4)

    # Weighted Retention Score (NEW — primary metric)
    wrs_data = compute_wrs(s1_pct, s2_pct, s3_pct)

    best_session  = scores_100.index(max(scores_100)) + 1
    worst_session = scores_100.index(min(scores_100)) + 1
    s1_to_s3_imp  = improvement_pct(s1_pct, s3_pct)

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

    # Weak topics
    weak_topics = extract_weak_topics(state, all_questions, all_answer_keys)
    cognitive_states = state.get("cognitive_states", {})

    # Session details
    session_details = {}
    for i in [1, 2, 3]:
        s      = s_data[str(i)]
        sp     = float(s["score_pct"] or 0)
        detail = {
            "session":         i,
            "difficulty":      DIFFICULTY_LABELS[i],
            "score_pct":       sp,
            "score_display":   f"{round(sp * 100, 1)}%",
            "score_label":     _simple_label(sp),
            "normalised_score": wrs_data["normalised_scores"][f"s{i}_adj"],
            "time_spent_min":  time_spent.get(f"session_{i}"),
            "started_at":      s.get("started_at"),
            "submitted_at":    s.get("submitted_at"),
            "override_used":   s.get("override_used", False),
        }
        if admin:
            detail["cognitive_state"] = cognitive_states.get(str(i), "not recorded")
        session_details[f"session_{i}"] = detail

    suggestions = generate_suggestions(
        wrs_data=wrs_data, scores=scores_pct,
        weak_topics=weak_topics,
        relistening_recommended=state.get("relistening_recommended", False),
    )

    # Learning curve (based on normalised scores, not raw)
    ns = wrs_data["normalised_scores"]
    n_scores = [ns["s1_adj"], ns["s2_adj"], ns["s3_adj"]]
    if n_scores[2] > n_scores[0] and n_scores[2] > n_scores[1]:
        curve = "improving"
        curve_desc = "Excellent — your normalised performance improved with each session despite increasing difficulty."
    elif n_scores[2] < n_scores[0] * 0.75:
        curve = "declining"
        curve_desc = "Your normalised performance dropped noticeably by Session 3. Focus on the weak topics."
    elif abs(n_scores[2] - n_scores[0]) <= 0.05:
        curve = "stable"
        curve_desc = "Consistent normalised performance — you're retaining knowledge evenly across difficulty levels."
    else:
        curve = "variable"
        curve_desc = "Mixed pattern — some difficulty levels landed better than others."

    analytics = {
        "status":         "success",
        "document_id":    doc_id,
        "document_title": state.get("document_title", ""),
        "role":           role,
        "generated_at":   now_iso(),
        "scoring_method": "weighted_retention_score",
        "summary": {
            # Primary metric
            "wrs":                   wrs_data["wrs"],
            "wrs_display":           wrs_data["wrs_display"],
            "wrs_label":             wrs_data["wrs_label"],
            "wrs_explanation":       wrs_data["explanation"],
            # Secondary / reference
            "flat_average_pct":      flat_avg,
            "flat_average_display":  f"{round(flat_avg * 100, 1)}%",
            "best_session":          best_session,
            "worst_session":         worst_session,
            "total_time_spent_min":  round(total_time, 2),
            "improvement_s1_to_s3":  f"{s1_to_s3_imp:+.1f}%",
            "learning_curve":        curve,
            "learning_curve_desc":   curve_desc,
            "relistening_recommended": state.get("relistening_recommended", False),
            "poor_score_warning":    state.get("poor_score_warning", False),
            "retention_decay":       wrs_data["retention_decay"],
            "bonuses_earned":        wrs_data["bonuses"],
            "total_bonus_pct":       wrs_data["total_bonus"],
        },
        "session_details":   session_details,
        "score_progression": [
            {"session": i, "difficulty": DIFFICULTY_LABELS[i],
             "raw_score": scores_100[i-1],
             "normalised_score": round(n_scores[i-1]*100, 1)}
            for i in [1, 2, 3]
        ],
        "weak_topics":  weak_topics,
        "suggestions":  suggestions,
        "brain_facts":  BRAIN_FACTS,
        "wrs_detail":   wrs_data,
    }

    if admin:
        analytics["admin_details"] = {
            "s1_to_s2_hours":     state.get("s1_to_s2_hours"),
            "s2_to_s3_hours":     state.get("s2_to_s3_hours"),
            "num_questions":      state.get("num_questions"),
            "llm_engine":         state.get("llm_engine", "unknown"),
            "created_at":         state.get("created_at"),
            "sessions_meta":      state.get("sessions_meta", {}),
        }

    return analytics


def _simple_label(pct: float) -> str:
    if pct >= 0.80: return "Strong"
    elif pct >= 0.60: return "Good"
    elif pct >= 0.40: return "Average"
    else: return "Needs Improvement"


# ── HTML report (WRS-focused design) ─────────────────────────────────────────

def generate_html_report(analytics: dict) -> str:
    doc_title   = analytics.get("document_title", "Tarang Session")
    summary     = analytics["summary"]
    sessions    = analytics["session_details"]
    progression = analytics["score_progression"]
    weak        = analytics["weak_topics"]
    suggestions = analytics["suggestions"]
    role        = analytics.get("role", "user")
    admin       = role == "admin"
    generated   = analytics.get("generated_at", now_iso())
    wrs_data    = analytics.get("wrs_detail", {})

    wrs_pct   = round(summary["wrs"] * 100, 1)
    wrs_label = summary["wrs_label"]
    flat_avg  = round(summary["flat_average_pct"] * 100, 1)

    # WRS colour
    if wrs_pct >= 75:
        wrs_color = "#34d399"
    elif wrs_pct >= 55:
        wrs_color = "#60a5fa"
    elif wrs_pct >= 40:
        wrs_color = "#fbbf24"
    else:
        wrs_color = "#f87171"

    def bar_color(score):
        if score >= 80: return "#34d399"
        elif score >= 60: return "#60a5fa"
        elif score >= 40: return "#fbbf24"
        return "#f87171"

    curve       = summary["learning_curve"]
    curve_icon  = {"improving":"▲","declining":"▼","stable":"→","variable":"~"}.get(curve,"→")
    curve_color = {"improving":"#34d399","declining":"#f87171","stable":"#60a5fa","variable":"#fbbf24"}.get(curve,"#94a3b8")

    # Retention decay display
    decay    = wrs_data.get("retention_decay", {})
    d12      = round(decay.get("s1_to_s2", 1.0) * 100, 1)
    d23      = round(decay.get("s2_to_s3", 1.0) * 100, 1)
    d12_col  = "#34d399" if d12 >= 85 else "#fbbf24" if d12 >= 70 else "#f87171"
    d23_col  = "#34d399" if d23 >= 80 else "#fbbf24" if d23 >= 65 else "#f87171"

    # Bonuses
    bonuses_earned = summary.get("bonuses_earned", {})
    bonus_html     = ""
    if bonuses_earned:
        bonus_items = []
        for k, v in bonuses_earned.items():
            label = k.replace("_", " ").title()
            bonus_items.append(
                f'<span style="padding:3px 10px;margin:3px;background:#14532d44;'
                f'border:1px solid #34d39944;border-radius:20px;font-size:0.75rem;color:#34d399">'
                f'+{round(v*100,1)}% {label}</span>'
            )
        bonus_html = f'<div style="margin-top:0.6rem">{"".join(bonus_items)}</div>'

    # Score progression bars
    prog_bars = ""
    for p in progression:
        raw_c = bar_color(p["raw_score"])
        nor_c = bar_color(p["normalised_score"])
        prog_bars += f"""
        <div style="margin-bottom:1.2rem">
          <div style="display:flex;justify-content:space-between;font-size:0.8rem;color:#64748b;margin-bottom:5px">
            <span>Session {p['session']} — {p['difficulty']}</span>
            <span>
              Raw: <span style="color:{raw_c}">{p['raw_score']}%</span>
              &nbsp;|&nbsp;
              Difficulty-adjusted: <span style="color:{nor_c}">{p['normalised_score']}%</span>
            </span>
          </div>
          <div style="background:#1e293b;border-radius:999px;height:8px;margin-bottom:3px">
            <div style="width:{p['raw_score']}%;background:{raw_c};height:8px;border-radius:999px;opacity:0.5"></div>
          </div>
          <div style="background:#1e293b;border-radius:999px;height:8px">
            <div style="width:{p['normalised_score']}%;background:{nor_c};height:8px;border-radius:999px"></div>
          </div>
          <div style="font-size:0.68rem;color:#334155;margin-top:2px">Top bar = raw score &nbsp;·&nbsp; Bottom bar = difficulty-adjusted score</div>
        </div>"""

    # Session table rows
    def session_row(sn, data):
        sc    = round(data["score_pct"] * 100, 1)
        na    = round(data["normalised_score"] * 100, 1)
        tm    = f"{data['time_spent_min']} min" if data["time_spent_min"] else "—"
        col   = bar_color(sc)
        nacol = bar_color(na)
        extra = ""
        if admin and "cognitive_state" in data:
            cs_lbl = COGNITIVE_MODE_LABELS.get(data["cognitive_state"], data["cognitive_state"])
            extra  = f'<td style="color:#94a3b8;font-size:0.78rem">{cs_lbl}</td>'
        return (
            f'<tr style="border-bottom:1px solid #1e293b">'
            f'<td style="padding:10px 14px;color:#e2e8f0">Session {sn} — {data["difficulty"]}</td>'
            f'<td style="padding:10px 14px"><span style="color:{col};font-weight:500">{sc}%</span></td>'
            f'<td style="padding:10px 14px"><span style="color:{nacol};font-weight:500">{na}%</span>'
            f'<span style="color:#334155;font-size:0.72rem;margin-left:6px">(adj.)</span></td>'
            f'<td style="padding:10px 14px;color:#64748b">{tm}</td>{extra}</tr>'
        )

    rows_html = (
        session_row(1, sessions["session_1"]) +
        session_row(2, sessions["session_2"]) +
        session_row(3, sessions["session_3"])
    )
    admin_col_hdr = '<th style="padding:10px 14px;color:#64748b;font-weight:400">Cognitive mode</th>' if admin else ""

    weak_html = "".join(
        f'<span style="display:inline-block;padding:3px 10px;margin:4px;background:#1e3a5f;'
        f'border:1px solid #2563eb44;border-radius:20px;font-size:0.78rem;color:#93c5fd">{w}</span>'
        for w in (weak[:8] if weak else ["No weak topics identified"])
    )
    sugg_html = "".join(
        f'<div style="padding:10px 14px;margin:8px 0;background:#0f2027;'
        f'border-left:3px solid #6366f1;border-radius:0 8px 8px 0;'
        f'font-size:0.85rem;color:#c7d2fe;line-height:1.6">{s}</div>'
        for s in suggestions
    )

    admin_section = ""
    if admin and "admin_details" in analytics:
        ad = analytics["admin_details"]
        meta_rows = "".join(
            f'<tr><td style="color:#64748b;padding:6px 12px">{k}</td>'
            f'<td style="color:#94a3b8;padding:6px 12px">{v}</td></tr>'
            for k, v in {
                "Questions per session": ad.get("num_questions"),
                "S1→S2 gap (hours)":     ad.get("s1_to_s2_hours"),
                "S2→S3 gap (hours)":     ad.get("s2_to_s3_hours"),
                "LLM Engine":            ad.get("llm_engine", "—"),
                "Created":               ad.get("created_at","")[:10],
            }.items()
        )
        admin_section = (
            f'<div style="margin-top:2rem;padding:1.2rem;background:#0d1526;'
            f'border:1px solid #1e293b;border-radius:10px">'
            f'<div style="font-size:0.7rem;letter-spacing:0.15em;color:#6366f1;'
            f'text-transform:uppercase;margin-bottom:0.8rem">Admin details</div>'
            f'<table style="width:100%;border-collapse:collapse;font-size:0.82rem">'
            f'{meta_rows}</table></div>'
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Tarang Analytics — {doc_title}</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;500;600&display=swap');
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ background:#050914; color:#e2e8f0; font-family:'Inter',system-ui,sans-serif; min-height:100vh; padding:2rem 1rem; }}
.container {{ max-width:880px; margin:0 auto; }}
.card {{ background:#0d1526; border:1px solid #1e293b; border-radius:12px; padding:1.5rem; margin-bottom:1.2rem; }}
.section-label {{ font-family:'Space Mono',monospace; font-size:0.65rem; letter-spacing:0.2em; color:#6366f1; text-transform:uppercase; margin-bottom:0.8rem; }}
.bar-track {{ background:#1e293b; border-radius:999px; height:10px; width:100%; margin:6px 0; }}
.bar-fill {{ height:10px; border-radius:999px; }}
table {{ width:100%; border-collapse:collapse; }}
th {{ text-align:left; padding:10px 14px; color:#64748b; font-weight:400; border-bottom:1px solid #1e293b; font-size:0.82rem; }}
.wrs-badge {{
  display:inline-flex; align-items:center; gap:8px;
  padding:6px 16px; border-radius:999px;
  background:{wrs_color}15; border:1px solid {wrs_color}44;
  font-family:'Space Mono',monospace; font-size:0.78rem; color:{wrs_color};
}}
</style>
</head>
<body>
<div class="container">
  <div style="text-align:center;margin-bottom:2rem">
    <div style="font-family:'Space Mono',monospace;font-size:0.65rem;letter-spacing:0.3em;color:#6366f1;text-transform:uppercase;margin-bottom:0.3rem">Tarang 2.0.0 &nbsp;·&nbsp; Neuro-Acoustic Learning</div>
    <h1 style="font-size:1.6rem;font-weight:300;color:#f1f5f9">{doc_title}</h1>
    <div style="font-size:0.8rem;color:#475569;margin-top:0.3rem">Analytics Report &nbsp;·&nbsp; {role.title()} View &nbsp;·&nbsp; {generated[:10]}</div>
  </div>

  <!-- WRS Primary Metric -->
  <div class="card" style="border-color:{wrs_color}33">
    <div class="section-label">Weighted Retention Score (WRS)</div>
    <div style="display:flex;align-items:center;gap:1.5rem;flex-wrap:wrap">
      <div style="text-align:center">
        <div style="font-size:3.5rem;font-weight:200;color:{wrs_color};font-family:'Space Mono',monospace;line-height:1">{wrs_pct}%</div>
        <div style="font-size:0.85rem;color:{wrs_color};margin-top:4px">{wrs_label}</div>
      </div>
      <div style="flex:1;min-width:200px">
        <div style="font-size:0.82rem;color:#94a3b8;line-height:1.7">{summary['wrs_explanation']}</div>
        {bonus_html}
        <div style="margin-top:0.8rem;font-size:0.75rem;color:#475569">
          Flat average (reference only): {flat_avg}% &nbsp;·&nbsp;
          WRS accounts for difficulty scaling and retention over time.
        </div>
      </div>
    </div>
    <div style="margin-top:1rem">
      <div class="bar-track"><div class="bar-fill" style="width:{wrs_pct}%;background:{wrs_color}"></div></div>
    </div>
  </div>

  <!-- Summary grid -->
  <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;margin-bottom:1.2rem">
    <div class="card" style="text-align:center">
      <div class="section-label">Learning curve</div>
      <div style="font-size:2.2rem;color:{curve_color}">{curve_icon}</div>
      <div style="font-size:0.78rem;color:#475569;margin-top:4px;text-transform:capitalize">{curve}</div>
    </div>
    <div class="card" style="text-align:center">
      <div class="section-label">S1→S2 retention</div>
      <div style="font-size:2.2rem;font-weight:300;color:{d12_col}">{d12}%</div>
      <div style="font-size:0.75rem;color:#475569;margin-top:4px">knowledge carried over</div>
    </div>
    <div class="card" style="text-align:center">
      <div class="section-label">S2→S3 retention</div>
      <div style="font-size:2.2rem;font-weight:300;color:{d23_col}">{d23}%</div>
      <div style="font-size:0.75rem;color:#475569;margin-top:4px">knowledge carried over</div>
    </div>
  </div>

  <!-- Score progression -->
  <div class="card">
    <div class="section-label">Score Progression</div>
    {prog_bars}
    <div style="font-size:0.78rem;color:#334155;border-top:1px solid #1e293b;padding-top:0.8rem;margin-top:0.5rem">{summary['learning_curve_desc']}</div>
  </div>

  <!-- Session breakdown table -->
  <div class="card">
    <div class="section-label">Session Breakdown</div>
    <table>
      <thead>
        <tr>
          <th>Session</th>
          <th>Raw Score</th>
          <th>Adjusted Score</th>
          <th>Time spent</th>
          {admin_col_hdr}
        </tr>
      </thead>
      <tbody>{rows_html}</tbody>
    </table>
    <div style="font-size:0.72rem;color:#334155;margin-top:0.8rem">
      Adjusted score = raw score normalised for difficulty (Hard answers weighted higher than Easy).
    </div>
  </div>

  <!-- Weak topics -->
  <div class="card">
    <div class="section-label">Weak Topics</div>
    <div style="margin-top:0.4rem">{weak_html}</div>
    <div style="font-size:0.72rem;color:#334155;margin-top:0.8rem">Based on incorrectly answered questions across all sessions.</div>
  </div>

  <!-- Suggestions -->
  <div class="card">
    <div class="section-label">Personalised Suggestions</div>
    {sugg_html}
  </div>

  {admin_section}

  <div style="text-align:center;font-size:0.68rem;font-family:'Space Mono',monospace;color:#1e293b;margin-top:2rem;padding-bottom:1rem">
    Tarang 2.0.0 &nbsp;·&nbsp; WRS Analytics &nbsp;·&nbsp; {generated[:19].replace('T',' ')} UTC
  </div>
</div>
</body>
</html>"""

    logger.info(f"Analytics HTML generated | WRS={wrs_pct}% | {len(html):,} chars")
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
    if not session_state:
        return {"status": "error", "error": "session_state not provided."}

    logger.info(f"Generating WRS analytics | doc_id={doc_id} | role={role}")

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
    logger.info(
        f"Analytics complete | WRS={analytics['summary']['wrs_display']} | "
        f"label={analytics['summary']['wrs_label']} | "
        f"weak_topics={len(analytics['weak_topics'])}"
    )

    return {
        "status":       "success",
        "doc_id":       doc_id,
        "role":         role,
        "analytics":    analytics,
        "html_content": html_content,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    print("\nTarang 2.0.0 — Analytics Engine (Weighted Retention Score)")
    print("This file requires session_state to be passed in from bridge.")
    print("Use bridge.py to trigger analytics via the /analytics endpoint.")
    # Demo WRS calculation
    print("\nDemo WRS scenarios:")
    scenarios = [
        ("High achiever",         0.90, 0.75, 0.65),
        ("Expected drop pattern", 0.80, 0.60, 0.50),
        ("Consistent performer",  0.70, 0.70, 0.70),
        ("Improving learner",     0.60, 0.70, 0.80),
        ("Struggling learner",    0.50, 0.35, 0.25),
    ]
    for name, s1, s2, s3 in scenarios:
        r = compute_wrs(s1, s2, s3)
        print(f"  {name:<25} S1={round(s1*100)}% S2={round(s2*100)}% S3={round(s3*100)}%  →  WRS={r['wrs_display']} ({r['wrs_label']})")
"""
Tarang 1.0.0.1 — file5_mcq.py
STATELESS: No local JSON files written at all.
initialise_document() returns ALL data as dicts.
Session management functions accept session_state as dict (from MongoDB via bridge)
and return updated_state for Express to save back to MongoDB.
"""

import os
import re
import json
import math
import time
import logging
import hashlib
from datetime import datetime, timedelta
from typing import Optional
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [file5_mcq] %(levelname)s — %(message)s"
)
logger = logging.getLogger("file5_mcq")

DEFAULT_NUM_QUESTIONS    = int(os.getenv("TARANG_MCQ_COUNT", "10"))
POOR_SCORE_THRESHOLD     = float(os.getenv("TARANG_POOR_SCORE", "0.30"))
SESSION_1_WINDOW_MINUTES = int(os.getenv("TARANG_S1_WINDOW_MIN", "10"))
DEFAULT_S1_TO_S2_HOURS   = float(os.getenv("TARANG_S1_S2_HOURS", "12.0"))
DEFAULT_S2_TO_S3_HOURS   = float(os.getenv("TARANG_S2_S3_HOURS", "24.0"))
MIN_S1_TO_S2_HOURS       = 4.0
MIN_S2_TO_S3_HOURS       = 6.0
OPENAI_API_KEY           = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL             = "gpt-4o-mini"

COGNITIVE_MODES = ["deep_focus", "memory", "calm", "deep_relaxation", "sleep"]
COGNITIVE_MODE_LABELS = {
    "deep_focus":      "Deep Focus (14 Hz Beta)",
    "memory":          "Memory (10 Hz Alpha)",
    "calm":            "Calm (8 Hz Alpha)",
    "deep_relaxation": "Deep Relaxation (6 Hz Theta)",
    "sleep":           "Sleep (4 Hz Theta/Delta)",
}
DIFFICULTY_CONFIG = {
    1: {"label": "Easy",   "description": "Basic recall and definitions",    "multiple_ratio": 0.10},
    2: {"label": "Medium", "description": "Application and relationships",   "multiple_ratio": 0.30},
    3: {"label": "Hard",   "description": "Deep analysis and multi-concept", "multiple_ratio": 0.50},
}


# ── OpenAI MCQ generation ─────────────────────────────────────────────────────

def _build_prompt(text: str, difficulty: int, num_questions: int) -> str:
    cfg      = DIFFICULTY_CONFIG[difficulty]
    label    = cfg["label"]
    desc     = cfg["description"]
    multi_n  = max(1, math.ceil(num_questions * cfg["multiple_ratio"]))
    single_n = num_questions - multi_n
    return f"""You are an expert educational assessment designer. Read the following document and generate exactly {num_questions} multiple choice questions.

DOCUMENT:
\"\"\"
{text[:4000]}
\"\"\"

REQUIREMENTS:
- Difficulty: {label} ({desc})
- Generate exactly {single_n} single-correct questions and {multi_n} multiple-correct questions
- Each question must be directly answerable from the document text
- Questions must test understanding, not just copy sentences word for word
- All 4 answer options must be short phrases (2-8 words each), NOT full sentences
- Correct answers must be factually accurate based on the document
- Wrong options (distractors) must be plausible but clearly incorrect
- Do NOT use code snippets, programming syntax, or raw data tables as answer options
- Do NOT ask about page numbers, formatting, or document structure

Return a JSON object with a "questions" key containing the array:
{{
  "questions": [
    {{
      "question_id": "q001",
      "question": "Question text here?",
      "instruction": "(Select ONE correct answer)",
      "answer_type": "single_correct",
      "difficulty": "{label}",
      "options": {{"A": "Short phrase", "B": "Short phrase", "C": "Short phrase", "D": "Short phrase"}},
      "source_concept": "main concept tested",
      "correct_answers": ["B"]
    }},
    {{
      "question_id": "q002",
      "question": "Multi-concept question?",
      "instruction": "(Select ALL correct answers)",
      "answer_type": "multiple_correct",
      "difficulty": "{label}",
      "options": {{"A": "Short phrase", "B": "Short phrase", "C": "Short phrase", "D": "Short phrase"}},
      "source_concept": "main concept tested",
      "correct_answers": ["A", "C"]
    }}
  ]
}}

Generate exactly {num_questions} questions following the format above."""


def _call_openai(prompt: str) -> Optional[list]:
    if not OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY not set.")
        return None
    try:
        from openai import OpenAI
    except ImportError:
        logger.error("openai not installed. Run: pip install openai")
        return None
    try:
        client   = OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert educational assessment designer. Respond with valid JSON only. No markdown, no explanation."},
                {"role": "user",   "content": prompt}
            ],
            temperature=0.3,
            max_tokens=4096,
            response_format={"type": "json_object"},
        )
        raw    = response.choices[0].message.content.strip()
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            questions = parsed
        elif isinstance(parsed, dict):
            questions = (
                parsed.get("questions") or parsed.get("mcqs") or
                next((v for v in parsed.values() if isinstance(v, list)), None)
            )
        else:
            return None
        if not questions:
            return None
        logger.info(f"OpenAI returned {len(questions)} questions | tokens: {response.usage.prompt_tokens}in {response.usage.completion_tokens}out")
        return questions
    except json.JSONDecodeError as e:
        logger.error(f"OpenAI JSON parse failed: {e}")
        return None
    except Exception as e:
        logger.error(f"OpenAI API call failed: {type(e).__name__}: {e}")
        return None


def _validate_questions(questions: list, session_num: int, num_questions: int) -> list:
    valid = []
    label = DIFFICULTY_CONFIG[session_num]["label"]
    for i, q in enumerate(questions):
        try:
            if not all(k in q for k in ["question", "options", "correct_answers"]):
                continue
            opts = q.get("options", {})
            if not all(k in opts for k in ["A", "B", "C", "D"]):
                continue
            correct = [c for c in q.get("correct_answers", []) if c in opts]
            if not correct:
                continue
            if any(re.search(r'[(){};]|INT\s|VARCHAR|SELECT\s|FROM\s', str(v)) for v in opts.values()):
                continue
            answer_type = "multiple_correct" if len(correct) > 1 else "single_correct"
            valid.append({
                "question_id":     f"q{(i+1):03d}",
                "question":        str(q.get("question", "")).strip(),
                "instruction":     "(Select ALL correct answers)" if answer_type == "multiple_correct" else "(Select ONE correct answer)",
                "answer_type":     answer_type,
                "difficulty":      label,
                "options":         {"A": str(opts["A"]).strip(), "B": str(opts["B"]).strip(),
                                    "C": str(opts["C"]).strip(), "D": str(opts["D"]).strip()},
                "source_concept":  str(q.get("source_concept", "")).strip(),
                "correct_answers": correct,
            })
        except Exception as e:
            logger.warning(f"Skipping invalid question {i}: {e}")
    for i, q in enumerate(valid):
        q["question_id"] = f"q{(i+1):03d}"
    return valid[:num_questions]


def generate_questions_with_ai(text: str, session_num: int, num_questions: int, max_retries: int = 2) -> list:
    for attempt in range(max_retries + 1):
        if attempt > 0:
            wait = 10 * attempt
            logger.info(f"Waiting {wait}s before retry {attempt+1}...")
            time.sleep(wait)
        logger.info(f"MCQ generation | session={session_num} | attempt={attempt+1}")
        raw_qs    = _call_openai(_build_prompt(text, session_num, num_questions))
        if not raw_qs:
            logger.warning(f"No questions on attempt {attempt+1}")
            continue
        validated = _validate_questions(raw_qs, session_num, num_questions)
        if len(validated) >= max(5, num_questions // 2):
            logger.info(f"Session {session_num}: {len(validated)} valid questions")
            return validated
        logger.warning(f"Only {len(validated)} valid questions on attempt {attempt+1}")
    logger.error(f"Failed to generate questions for session {session_num}")
    return []


# ── Helpers ───────────────────────────────────────────────────────────────────

def now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"

def parse_iso(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).replace(tzinfo=None)

def minutes_since(ts: str) -> float:
    return (datetime.utcnow() - parse_iso(ts)).total_seconds() / 60.0

def hours_since(ts: str) -> float:
    return (datetime.utcnow() - parse_iso(ts)).total_seconds() / 3600.0

def is_admin(role: str) -> bool:
    return str(role).strip().lower() == "admin"


# ── Main initialise function ──────────────────────────────────────────────────

def initialise_document(
    text: str = None,
    source_txt_path=None,
    num_questions: int = DEFAULT_NUM_QUESTIONS,
    custom_s1_to_s2_hours: float = DEFAULT_S1_TO_S2_HOURS,
    custom_s2_to_s3_hours: float = DEFAULT_S2_TO_S3_HOURS,
    document_title: str = "Tarang Document",
) -> dict:
    """
    STATELESS: No local files written.
    Generates all 3 MCQ sessions and returns everything as dicts.
    Bridge passes this data to Express which stores in MongoDB:
      - session_state       → Session collection
      - session_N_questions → Session.questions[N]
      - session_N_answers   → Session.answerKey[N]
    """
    if text is None and source_txt_path is not None:
        from pathlib import Path as _Path
        p = _Path(source_txt_path)
        if not p.exists():
            return {"status": "error", "error": f"File not found: {p}"}
        text = p.read_text(encoding="utf-8")
    if not text or not text.strip():
        return {"status": "error", "error": "Input text is empty."}
    if not OPENAI_API_KEY:
        return {"status": "error", "error": "OPENAI_API_KEY not set."}

    doc_id   = hashlib.md5((text[:500] + str(int(time.time()*1000))).encode()).hexdigest()[:12]
    s1_to_s2 = max(MIN_S1_TO_S2_HOURS, float(custom_s1_to_s2_hours))
    s2_to_s3 = max(MIN_S2_TO_S3_HOURS, float(custom_s2_to_s3_hours))
    logger.info(f"Initialising | doc_id={doc_id} | title={document_title}")

    sessions_meta = {}
    session_data  = {}   # session_num → {questions, public_qs, answer_key}

    # Run all 3 OpenAI calls concurrently in a thread pool.
    # Sequential: 3 × ~25s = ~75s. Parallel: ~25s (limited by slowest call).
    def _generate_session(session_num):
        return session_num, generate_questions_with_ai(
            text=text, session_num=session_num, num_questions=num_questions
        )

    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(_generate_session, n): n for n in [1, 2, 3]}
        results = {}
        for future in concurrent.futures.as_completed(futures):
            session_num, qs = future.result()
            results[session_num] = qs

    for session_num in [1, 2, 3]:
        qs = results[session_num]
        if not qs:
            return {"status": "error", "error": f"Failed to generate Session {session_num} questions. Check OPENAI_API_KEY."}

        public_qs  = [{k: v for k, v in q.items() if k != "correct_answers"} for q in qs]
        answer_key = {
            q["question_id"]: {"correct_answers": q["correct_answers"], "answer_type": q["answer_type"]}
            for q in qs
        }
        session_data[session_num] = {"questions": qs, "public_qs": public_qs, "answer_key": answer_key}

        sc = sum(1 for q in qs if q["answer_type"] == "single_correct")
        mc = sum(1 for q in qs if q["answer_type"] == "multiple_correct")
        sessions_meta[f"session_{session_num}"] = {
            "difficulty": DIFFICULTY_CONFIG[session_num]["label"],
            "total_questions": len(qs), "single_correct": sc, "multiple_correct": mc,
        }
        logger.info(f"Session {session_num} ({DIFFICULTY_CONFIG[session_num]['label']}) — {len(qs)} questions ({sc} single, {mc} multiple)")

    # Session state — Express stores this in MongoDB
    session_state = {
        "document_id":           doc_id,
        "document_title":        document_title,
        "num_questions":         num_questions,
        "s1_to_s2_hours":        s1_to_s2,
        "s2_to_s3_hours":        s2_to_s3,
        "current_session":       0,
        "audio_completed_at":    None,
        "sessions": {
            "1": {"status": "pending", "started_at": None, "submitted_at": None, "score_pct": None, "override_used": False, "user_answers": {}},
            "2": {"status": "locked",  "started_at": None, "submitted_at": None, "score_pct": None, "override_used": False, "user_answers": {}},
            "3": {"status": "locked",  "started_at": None, "submitted_at": None, "score_pct": None, "override_used": False, "user_answers": {}},
        },
        "all_sessions_complete":   False,
        "answers_unlocked":        False,
        "poor_score_warning":      False,
        "relistening_recommended": False,
        "sessions_meta":           sessions_meta,
        "created_at":              now_iso(),
    }

    logger.info(f"Document initialised — doc_id: {doc_id}")

    return {
        "status":             "success",
        "document_id":        doc_id,
        "document_title":     document_title,
        "sessions_generated": 3,
        "sessions_meta":      sessions_meta,
        "s1_to_s2_hours":     s1_to_s2,
        "s2_to_s3_hours":     s2_to_s3,
        # ── All data returned — bridge passes to Express → stored in MongoDB ──
        "session_state": session_state,
        "session_1_questions": {
            "document_id": doc_id, "session": 1,
            "difficulty":  DIFFICULTY_CONFIG[1]["label"],
            "description": DIFFICULTY_CONFIG[1]["description"],
            "questions":   session_data[1]["public_qs"],
            "generated_at": now_iso(),
        },
        "session_1_answers": {"document_id": doc_id, "session": 1, "answers": session_data[1]["answer_key"], "generated_at": now_iso()},
        "session_2_questions": {
            "document_id": doc_id, "session": 2,
            "difficulty":  DIFFICULTY_CONFIG[2]["label"],
            "description": DIFFICULTY_CONFIG[2]["description"],
            "questions":   session_data[2]["public_qs"],
            "generated_at": now_iso(),
        },
        "session_2_answers": {"document_id": doc_id, "session": 2, "answers": session_data[2]["answer_key"], "generated_at": now_iso()},
        "session_3_questions": {
            "document_id": doc_id, "session": 3,
            "difficulty":  DIFFICULTY_CONFIG[3]["label"],
            "description": DIFFICULTY_CONFIG[3]["description"],
            "questions":   session_data[3]["public_qs"],
            "generated_at": now_iso(),
        },
        "session_3_answers": {"document_id": doc_id, "session": 3, "answers": session_data[3]["answer_key"], "generated_at": now_iso()},
        "message": "Document ready. Call audio_completed() when user finishes listening.",
    }


# ── Session management functions ──────────────────────────────────────────────
# All functions accept session_state dict (from MongoDB via bridge)
# and return updated_state for Express to save back to MongoDB.

def audio_completed(doc_id: str, role: str = "user", session_state: dict = None) -> dict:
    if not session_state:
        return {"status": "error", "error": "session_state not provided."}
    state = json.loads(json.dumps(session_state))  # deep copy
    state["audio_completed_at"]      = now_iso()
    state["current_session"]         = 1
    state["sessions"]["1"]["status"] = "available"
    state["role"]                    = role
    if is_admin(role):
        state["sessions"]["1"]["override_used"] = True
    window = 0 if is_admin(role) else SESSION_1_WINDOW_MINUTES
    return {
        "status":         "success",
        "document_id":    doc_id,
        "role":           role,
        "window_minutes": window,
        "deadline":       (datetime.utcnow() + timedelta(minutes=window)).isoformat() + "Z",
        "updated_state":  state,  # Express saves this back to MongoDB
        "message":        "Session 1 available immediately (admin)." if is_admin(role)
                          else f"Session 1 available. {window} minutes to begin.",
    }


def override_window(doc_id: str, session_state: dict = None) -> dict:
    if not session_state:
        return {"status": "error", "error": "session_state not provided."}
    state = json.loads(json.dumps(session_state))
    state["sessions"]["1"]["override_used"] = True
    return {"status": "success", "updated_state": state, "message": "Override accepted. You may now take Session 1."}


def get_session_status(doc_id: str, role: str = "user", session_state: dict = None) -> dict:
    if not session_state:
        return {"status": "error", "error": "session_state not provided."}
    state  = json.loads(json.dumps(session_state))
    s_data = state["sessions"]
    admin  = is_admin(role)

    audio_done        = state.get("audio_completed_at")
    s1_window_expired = False
    s1_minutes_left   = None
    if audio_done and s_data["1"]["status"] == "available":
        mins = minutes_since(audio_done)
        if mins > SESSION_1_WINDOW_MINUTES:
            s1_window_expired = True
            s1_minutes_left   = 0
        else:
            s1_minutes_left = round(SESSION_1_WINDOW_MINUTES - mins, 1)

    state_changed = False
    if s_data["2"]["status"] == "locked":
        s1_sub = s_data["1"].get("submitted_at")
        if s1_sub and (admin or hours_since(s1_sub) >= state["s1_to_s2_hours"]):
            state["sessions"]["2"]["status"] = "available"
            state_changed = True
    if s_data["3"]["status"] == "locked":
        s2_sub = s_data["2"].get("submitted_at")
        if s2_sub and (admin or hours_since(s2_sub) >= state["s2_to_s3_hours"]):
            state["sessions"]["3"]["status"] = "available"
            state_changed = True

    sessions_out = {}
    for snum in ["1", "2", "3"]:
        s      = state["sessions"][snum]
        sn_int = int(snum)
        entry  = {
            "session":       sn_int,
            "difficulty":    DIFFICULTY_CONFIG[sn_int]["label"],
            "description":   DIFFICULTY_CONFIG[sn_int]["description"],
            "status":        s["status"],
            "override_used": s["override_used"],
            "started_at":    s["started_at"],
            "submitted_at":  s["submitted_at"],
            "score_pct":     s["score_pct"] if state["all_sessions_complete"] else None,
        }
        if not admin:
            if snum == "2" and s["status"] == "locked":
                s1_sub = s_data["1"].get("submitted_at")
                if s1_sub:
                    entry["hours_until_available"] = round(max(0, state["s1_to_s2_hours"] - hours_since(s1_sub)), 2)
            if snum == "3" and s["status"] == "locked":
                s2_sub = s_data["2"].get("submitted_at")
                if s2_sub:
                    entry["hours_until_available"] = round(max(0, state["s2_to_s3_hours"] - hours_since(s2_sub)), 2)
        sessions_out[snum] = entry

    messages = []
    if s1_window_expired and not s_data["1"]["override_used"] and s_data["1"]["status"] not in ("in_progress", "completed"):
        messages.append("Session 1 window expired. Click 'I have listened carefully' to proceed.")
    if state["relistening_recommended"]:
        messages.append("Score below 30% detected. Consider re-listening before continuing.")

    return {
        "status":                  "success",
        "document_id":             doc_id,
        "document_title":          state.get("document_title", ""),
        "current_session":         state["current_session"],
        "all_sessions_complete":   state["all_sessions_complete"],
        "answers_unlocked":        state["answers_unlocked"],
        "relistening_recommended": state["relistening_recommended"],
        "poor_score_warning":      state["poor_score_warning"],
        "sessions":                sessions_out,
        "messages":                messages,
        "s1_window_expired":       s1_window_expired,
        "s1_minutes_left":         s1_minutes_left,
        "updated_state":           state if state_changed else None,
        "relisten_options": {
            "available": True,
            "modes": [{"key": k, "label": COGNITIVE_MODE_LABELS[k]} for k in COGNITIVE_MODES],
        },
    }


def get_questions(doc_id: str, session: int, role: str = "user",
                  session_state: dict = None, questions_data: dict = None) -> dict:
    """
    session_state:  full state dict from MongoDB
    questions_data: the questions dict for this session from MongoDB
    """
    if not session_state:
        return {"status": "error", "error": "session_state not provided.", "can_override": False}
    if not questions_data:
        return {"status": "error", "error": "questions_data not provided.", "can_override": False}

    state  = json.loads(json.dumps(session_state))
    s_key  = str(session)
    s_data = state["sessions"].get(s_key)
    if not s_data:
        return {"status": "error", "error": f"Invalid session: {session}"}
    admin = is_admin(role)

    if session == 1:
        if not state.get("audio_completed_at"):
            return {"status": "error", "error": "Audio not marked complete.", "can_override": False}
        if not admin:
            mins = minutes_since(state["audio_completed_at"])
            if mins > SESSION_1_WINDOW_MINUTES and not s_data["override_used"]:
                return {"status": "error", "reason": "window_expired", "can_override": True,
                        "error": "Session 1 window expired. Click 'I have listened carefully' to proceed.",
                        "minutes_elapsed": round(mins, 1)}
    elif session == 2:
        if not state["sessions"]["1"].get("submitted_at"):
            return {"status": "error", "error": "Session 1 not completed.", "can_override": False}
        if not admin:
            gap  = state["s1_to_s2_hours"]
            wait = hours_since(state["sessions"]["1"]["submitted_at"])
            if wait < gap:
                return {"status": "error", "reason": "too_early", "can_override": False,
                        "error": f"Session 2 available in {round(gap-wait,1)} hours.", "hours_remaining": round(gap-wait,2)}
    elif session == 3:
        if not state["sessions"]["2"].get("submitted_at"):
            return {"status": "error", "error": "Session 2 not completed.", "can_override": False}
        if not admin:
            gap  = state["s2_to_s3_hours"]
            wait = hours_since(state["sessions"]["2"]["submitted_at"])
            if wait < gap:
                return {"status": "error", "reason": "too_early", "can_override": False,
                        "error": f"Session 3 available in {round(gap-wait,1)} hours.", "hours_remaining": round(gap-wait,2)}

    state_changed = False
    if not s_data["started_at"]:
        state["sessions"][s_key]["started_at"] = now_iso()
        state["sessions"][s_key]["status"]     = "in_progress"
        state_changed = True

    qs = questions_data.get("questions", [])
    return {
        "status":        "success",
        "document_id":   doc_id,
        "session":       session,
        "difficulty":    DIFFICULTY_CONFIG[session]["label"],
        "description":   DIFFICULTY_CONFIG[session]["description"],
        "instruction":   f"Session {session} of 3 — {DIFFICULTY_CONFIG[session]['label']} difficulty. {DIFFICULTY_CONFIG[session]['description']}.",
        "questions":     qs,
        "total":         len(qs),
        "started_at":    state["sessions"][s_key]["started_at"],
        "updated_state": state if state_changed else None,
        "relisten_options": {"modes": [{"key": k, "label": COGNITIVE_MODE_LABELS[k]} for k in COGNITIVE_MODES]},
    }


def submit_test(doc_id: str, session: int, user_answers: dict, role: str = "user",
                session_state: dict = None, answer_key_data: dict = None) -> dict:
    """
    session_state:   full state dict from MongoDB
    answer_key_data: answer key dict for this session from MongoDB
    Returns updated_state for Express to save back to MongoDB.
    """
    if not session_state:
        return {"status": "error", "error": "session_state not provided."}
    if not answer_key_data:
        return {"status": "error", "error": "answer_key_data not provided."}

    state  = json.loads(json.dumps(session_state))
    s_key  = str(session)
    s_data = state["sessions"].get(s_key)
    if not s_data:
        return {"status": "error", "error": f"Invalid session: {session}"}
    if s_data["status"] == "completed":
        return {"status": "error", "error": f"Session {session} already submitted."}

    answer_key    = answer_key_data.get("answers", answer_key_data)
    correct_count = 0
    total         = len(answer_key)
    for q_id, key_data in answer_key.items():
        if sorted(user_answers.get(q_id, [])) == sorted(key_data["correct_answers"]):
            correct_count += 1

    score_pct = round(correct_count / total, 4) if total > 0 else 0.0
    logger.info(f"Session {session} submitted | doc={doc_id} | score={round(score_pct*100,1)}% ({correct_count}/{total})")

    state["sessions"][s_key].update({
        "status":       "completed",
        "submitted_at": now_iso(),
        "score_pct":    score_pct,
        "user_answers": user_answers,  # stored for weak topic analysis in file6
    })

    all_done = all(state["sessions"][str(s)]["status"] == "completed" for s in [1,2,3])
    state["all_sessions_complete"] = all_done
    if all_done:
        state["answers_unlocked"] = True

    poor = [s for s in ["1","2","3"]
            if state["sessions"][s]["score_pct"] is not None
            and state["sessions"][s]["score_pct"] < POOR_SCORE_THRESHOLD]
    if poor:
        state["poor_score_warning"]      = True
        state["relistening_recommended"] = True

    response = {
        "status":                  "success",
        "document_id":             doc_id,
        "session":                 session,
        "correct_count":           correct_count,
        "total_questions":         total,
        "scores_visible":          all_done,
        "all_sessions_done":       all_done,
        "answers_unlocked":        state["answers_unlocked"],
        "relistening_recommended": state["relistening_recommended"],
        "poor_sessions":           poor,
        "submitted_at":            state["sessions"][s_key]["submitted_at"],
        "updated_state":           state,  # Express saves this back to MongoDB
    }

    if all_done:
        avg = sum(state["sessions"][str(s)]["score_pct"] or 0 for s in [1,2,3]) / 3
        response.update({
            "score_pct":              score_pct,
            "score_display":          f"{round(score_pct*100,1)}%",
            "all_scores": {
                f"session_{s}": {
                    "score_pct":     state["sessions"][str(s)]["score_pct"],
                    "score_display": f"{round(state['sessions'][str(s)]['score_pct']*100,1)}%",
                    "difficulty":    DIFFICULTY_CONFIG[s]["label"],
                } for s in [1,2,3]
            },
            "average_score_pct":     round(avg, 4),
            "average_score_display": f"{round(avg*100,1)}%",
            "message": ("All 3 sessions complete! Answers are now unlocked." if not poor
                        else f"All done. Score below 30% in session(s) {', '.join(poor)}. Consider re-listening."),
        })
    else:
        response["score_pct"]     = None
        response["score_display"] = "Results visible after Session 3"
        if session < 3:
            next_s  = session + 1
            gap_key = "s1_to_s2_hours" if session == 1 else "s2_to_s3_hours"
            response["message"] = (
                f"Session {session} submitted. Session {next_s} "
                f"({DIFFICULTY_CONFIG[next_s]['label']}) unlocks in {state[gap_key]} hours."
            )
    return response


def get_final_results(doc_id: str, role: str = "user",
                      session_state: dict = None, all_answer_keys: dict = None) -> dict:
    if not session_state:
        return {"status": "error", "error": "session_state not provided."}
    state = session_state
    if not state.get("all_sessions_complete"):
        done = sum(1 for s in ["1","2","3"] if state["sessions"][s]["status"] == "completed")
        return {"status": "error", "error": "All 3 sessions must be complete.", "sessions_completed": done}

    avg = sum(state["sessions"][str(s)]["score_pct"] or 0 for s in [1,2,3]) / 3
    return {
        "status":                "success",
        "document_id":           doc_id,
        "document_title":        state.get("document_title", ""),
        "all_sessions_complete": True,
        "answers_unlocked":      True,
        "all_scores": {
            f"session_{s}": {
                "score_pct":     state["sessions"][str(s)]["score_pct"],
                "score_display": f"{round(state['sessions'][str(s)]['score_pct']*100,1)}%",
                "difficulty":    DIFFICULTY_CONFIG[s]["label"],
                "submitted_at":  state["sessions"][str(s)]["submitted_at"],
            } for s in [1,2,3]
        },
        "average_score_pct":       round(avg, 4),
        "average_score_display":   f"{round(avg*100,1)}%",
        "relistening_recommended": state["relistening_recommended"],
        "poor_score_warning":      state["poor_score_warning"],
        "correct_answers":         all_answer_keys or {},
    }

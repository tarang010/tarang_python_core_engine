"""
Tarang 1.0.0.1 — Python Bridge
bridge.py — FastAPI Microservice

Runs on: http://localhost:5001
Auto docs: http://localhost:5001/docs  (Swagger UI)

Responsibilities:
  - Expose all 6 core engine files as REST endpoints
  - Accept JSON + multipart file uploads from Express backend
  - Call the appropriate core engine function
  - Return structured JSON responses
  - Provide a health check endpoint for Express to ping on startup

All endpoints return:
  { status: "success" | "error", data: {...}, error?: "..." }

Run with:
  uvicorn bridge:app --host 0.0.0.0 --port 5001 --reload

Fixes applied (v1.0.0.2):
  1. locals() used instead of dir() in finally blocks
  2. /pipeline/mcq reads from initialise_document() return value — no disk reads
  3. AnalyticsRequest includes session_state, all_questions, all_answer_keys
  4. MCQ session models include session_state, questions_data, answer_key_data
  5. All endpoints pass session_state dicts to file5_mcq.py functions
  6. Verbose console logging on every step for Render log visibility
  7. CORS updated to allow Render backend + Firebase origins
"""

import os
import sys
import json
import shutil
import logging
import tempfile
import base64
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel

# ── Add project root to path ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Core engine imports ───────────────────────────────────────────────────────
print("==> [bridge] Importing core engine modules...")
from file1_extractor import extract
from file2_tts       import generate_tts, get_voices
from file3_modulator import modulate_audio
from file4_visualizer import generate_visualization
from file5_mcq       import (
    initialise_document,
    audio_completed,
    get_session_status,
    get_questions,
    override_window,
    submit_test,
    get_final_results,
)
from file6_analytics import generate_analytics
from file7_captions  import generate_captions
print("==> [bridge] All core engine modules imported OK")

# ── Logging — verbose for Render console visibility ───────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [bridge] %(levelname)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("bridge")
logger.setLevel(logging.INFO)

# Also set root logger so all module logs appear in Render console
logging.getLogger().setLevel(logging.INFO)
for mod in ["file1_extractor","file2_tts","file3_modulator",
            "file4_visualizer","file5_mcq","file6_analytics","file7_captions"]:
    logging.getLogger(mod).setLevel(logging.INFO)

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Tarang Core Engine Bridge",
    description=(
        "FastAPI microservice exposing Tarang's Python core engine "
        "to the Express.js backend. Runs on localhost:5001."
    ),
    version="1.0.0.2",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS — allow all origins (Express + Firebase) ────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic request models ───────────────────────────────────────────────────

class TTSRequest(BaseModel):
    extracted_txt_path: str
    engine:             Optional[str] = "edge"
    output_filename:    Optional[str] = None
    voice_id:           Optional[str] = None


class ModulateRequest(BaseModel):
    tts_wav_path:     str
    cognitive_state:  Optional[str]   = "deep_focus"
    output_filename:  Optional[str]   = None
    custom_beat_freq: Optional[float] = None
    custom_depth:     Optional[float] = None


class VisualizeRequest(BaseModel):
    modulated_wav_path: str
    role:               Optional[str]   = "user"
    cognitive_state:    Optional[str]   = "deep_focus"
    beat_freq_hz:       Optional[float] = 14.0
    document_title:     Optional[str]   = "Tarang Session"
    session_number:     Optional[int]   = 1
    output_stem:        Optional[str]   = None


class MCQInitRequest(BaseModel):
    extracted_txt_path:     str
    document_title:         Optional[str]   = "Tarang Document"
    num_questions:          Optional[int]   = 10
    custom_s1_to_s2_hours:  Optional[float] = 12.0
    custom_s2_to_s3_hours:  Optional[float] = 24.0


# ── MCQ session models — ALL include session_state ────────────────────────────

class AudioCompletedRequest(BaseModel):
    doc_id:        str
    role:          Optional[str]  = "user"
    session_state: Optional[dict] = None   # full state dict from MongoDB


class GetQuestionsRequest(BaseModel):
    doc_id:         str
    session:        int
    role:           Optional[str]  = "user"
    session_state:  Optional[dict] = None  # full state dict from MongoDB
    questions_data: Optional[dict] = None  # questions dict for this session


class OverrideWindowRequest(BaseModel):
    doc_id:        str
    session_state: Optional[dict] = None  # full state dict from MongoDB


class SubmitTestRequest(BaseModel):
    doc_id:          str
    session:         int
    user_answers:    dict             # { "q001": ["A"], "q002": ["B","C"] }
    role:            Optional[str]  = "user"
    session_state:   Optional[dict] = None  # full state dict from MongoDB
    answer_key_data: Optional[dict] = None  # answer key for this session


class MCQStatusRequest(BaseModel):
    doc_id:        str
    role:          Optional[str]  = "user"
    session_state: Optional[dict] = None  # full state dict from MongoDB


class MCQResultsRequest(BaseModel):
    doc_id:          str
    role:            Optional[str]  = "user"
    session_state:   Optional[dict] = None
    all_answer_keys: Optional[dict] = None


# ── Analytics request — includes all session data ────────────────────────────

class AnalyticsRequest(BaseModel):
    doc_id:          str
    role:            Optional[str]  = "user"
    session_state:   Optional[dict] = None  # full state dict from MongoDB
    all_questions:   Optional[dict] = None  # {1: questions_dict, 2: ..., 3: ...}
    all_answer_keys: Optional[dict] = None  # {1: answers_dict, 2: ..., 3: ...}
    cognitive_states:Optional[dict] = None


class CaptionsRequest(BaseModel):
    text:         str
    duration_sec: float


# ── Pipeline models ───────────────────────────────────────────────────────────

class PipelineMCQRequest(BaseModel):
    extracted_text: str
    document_title: str = "Tarang Document"
    doc_id:         str = ""


# ── Response helpers ──────────────────────────────────────────────────────────

def ok(data: dict) -> JSONResponse:
    return JSONResponse(content={"status": "success", "data": data})


def err(message: str, code: int = 400) -> JSONResponse:
    logger.error(f"Returning error response | code={code} | message={message}")
    return JSONResponse(
        status_code=code,
        content={"status": "error", "error": message}
    )


# ── System endpoints ──────────────────────────────────────────────────────────

@app.get("/", tags=["System"])
async def root():
    logger.info("GET / — root endpoint hit")
    return {
        "service": "Tarang Core Engine Bridge",
        "version": "1.0.0.2",
        "docs":    "/docs",
        "status":  "running",
    }


@app.get("/health", tags=["System"])
async def health():
    """Express backend pings this on startup to confirm bridge is alive."""
    logger.info("GET /health — health check OK")
    return {
        "status":  "ok",
        "service": "Tarang Python Bridge",
        "version": "1.0.0.2",
    }


@app.get("/voices", tags=["Core Engine"])
async def list_voices():
    """Returns available TTS voices. For edge-tts engine: curated English list."""
    logger.info("GET /voices — listing voices")
    result = get_voices()
    if result["status"] != "success":
        return err(result.get("error", "Failed to list voices"))
    logger.info(f"GET /voices — returning {result['count']} voices")
    return ok({"voices": result["voices"], "count": result["count"]})


# ── FILE 1 — Document extraction ──────────────────────────────────────────────

@app.post("/extract", tags=["Core Engine"])
async def extract_document(file: UploadFile = File(...)):
    """Upload a document. Extracts clean text. Returns text, word count."""
    logger.info(f"POST /extract | filename={file.filename}")
    suffix    = Path(file.filename).suffix.lower()
    tmp_path  = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            shutil.copyfileobj(file.file, tmp)
        logger.info(f"  Saved upload to temp: {tmp_path}")
        result = extract(filepath=tmp_path, save_output=False)
        if result["status"] != "success":
            return err(result.get("error", "Extraction failed"))
        logger.info(f"  Extraction OK | words={result['word_count']}")
        return ok({
            "output_path": None,
            "text":        result["text"],
            "word_count":  result["word_count"],
            "char_count":  result["char_count"],
            "format":      result["format"],
            "metadata":    result["metadata"],
            "timestamp":   result["timestamp"],
        })
    finally:
        if tmp_path:
            try:
                if os.path.exists(tmp_path): os.unlink(tmp_path)
            except Exception as e:
                logger.warning(f"  Temp cleanup failed: {e}")


# ── FILE 2 — TTS ──────────────────────────────────────────────────────────────

@app.post("/tts", tags=["Core Engine"])
async def generate_tts_audio(req: TTSRequest):
    """Generate TTS audio from extracted text file path."""
    logger.info(f"POST /tts | path={req.extracted_txt_path} | engine={req.engine}")
    if not Path(req.extracted_txt_path).exists():
        return err(f"Text file not found: {req.extracted_txt_path}")
    result = generate_tts(
        source_txt_path=req.extracted_txt_path,
        engine=req.engine,
        output_filename=req.output_filename,
        voice_id=req.voice_id,
    )
    if result["status"] != "success":
        return err(result.get("error", "TTS generation failed"))
    logger.info(f"  TTS OK | engine={result['engine_used']} | duration={result['duration_sec']}s")
    return ok({
        "output_path":  result["output_path"],
        "engine_used":  result["engine_used"],
        "duration_sec": result["duration_sec"],
        "word_count":   result["word_count"],
        "chunks_total": result["chunks_total"],
        "timestamp":    result["timestamp"],
    })


# ── FILE 3 — Modulation ───────────────────────────────────────────────────────

@app.post("/modulate", tags=["Core Engine"])
async def modulate(req: ModulateRequest):
    """Apply binaural AM modulation to TTS WAV file."""
    logger.info(f"POST /modulate | path={req.tts_wav_path} | state={req.cognitive_state}")
    if not Path(req.tts_wav_path).exists():
        return err(f"WAV file not found: {req.tts_wav_path}")
    result = modulate_audio(
        input_wav_path=req.tts_wav_path,
        cognitive_state=req.cognitive_state,
        output_filename=req.output_filename,
        custom_beat_freq=req.custom_beat_freq,
        custom_depth=req.custom_depth,
    )
    if result["status"] != "success":
        return err(result.get("error", "Modulation failed"))
    logger.info(f"  Modulate OK | beat={result['beat_freq_hz']}Hz | duration={result['duration_sec']}s")
    return ok({
        "output_path":     result["output_path"],
        "cognitive_state": result["cognitive_state"],
        "beat_freq_hz":    result["beat_freq_hz"],
        "carrier_freq_hz": result["carrier_freq_hz"],
        "depth":           result["depth"],
        "duration_sec":    result["duration_sec"],
        "sample_rate":     result["sample_rate"],
        "timestamp":       result["timestamp"],
    })


# ── FILE 4 — Visualization ────────────────────────────────────────────────────

@app.post("/visualize", tags=["Core Engine"])
async def visualize(req: VisualizeRequest):
    """Generate audio visualization HTML."""
    logger.info(f"POST /visualize | role={req.role} | state={req.cognitive_state}")
    if not Path(req.modulated_wav_path).exists():
        return err(f"Audio file not found: {req.modulated_wav_path}")
    result = generate_visualization(
        modulated_wav_path=req.modulated_wav_path,
        role=req.role,
        cognitive_state=req.cognitive_state,
        beat_freq_hz=req.beat_freq_hz,
        document_title=req.document_title,
        session_number=req.session_number,
        output_stem=req.output_stem,
    )
    if result["status"] != "success":
        return err(result.get("error", "Visualization failed"))
    logger.info(f"  Visualize OK | type={result['report_type']}")
    return ok({
        "output_path":  result["output_path"],
        "html_content": result.get("html_content", ""),
        "report_type":  result["report_type"],
        "role":         result["role"],
        "timestamp":    result["timestamp"],
    })


# ── FILE 5 — MCQ session management ──────────────────────────────────────────

@app.post("/mcq/init", tags=["MCQ"])
async def mcq_init(req: MCQInitRequest):
    """
    Initialise a document for MCQ. Generates all 3 question sets.
    Returns session_state + all questions/answers as dicts.
    """
    logger.info(f"POST /mcq/init | path={req.extracted_txt_path} | title={req.document_title}")
    if not Path(req.extracted_txt_path).exists():
        return err(f"Text file not found: {req.extracted_txt_path}")
    result = initialise_document(
        source_txt_path=req.extracted_txt_path,
        document_title=req.document_title,
        num_questions=req.num_questions,
        custom_s1_to_s2_hours=req.custom_s1_to_s2_hours,
        custom_s2_to_s3_hours=req.custom_s2_to_s3_hours,
    )
    if result["status"] != "success":
        return err(result.get("error", "MCQ initialisation failed"))
    logger.info(f"  MCQ init OK | doc_id={result['document_id']}")
    return ok(result)


@app.post("/mcq/audio-completed", tags=["MCQ"])
async def mcq_audio_completed(req: AudioCompletedRequest):
    """
    Mark audio as completed. Starts the Session 1 window.
    Requires session_state from MongoDB.
    """
    logger.info(f"POST /mcq/audio-completed | doc_id={req.doc_id} | role={req.role}")
    if not req.session_state:
        logger.error("  Missing session_state in request body")
        return err("session_state is required. Pass it from MongoDB.", 400)
    result = audio_completed(
        doc_id=req.doc_id,
        role=req.role,
        session_state=req.session_state,
    )
    if result["status"] != "success":
        return err(result.get("error", "Failed to mark audio complete"))
    logger.info(f"  Audio completed OK | doc_id={req.doc_id} | role={req.role}")
    return ok(result)


@app.post("/mcq/status", tags=["MCQ"])
async def mcq_status_post(req: MCQStatusRequest):
    """
    Get full session status. Requires session_state from MongoDB.
    POST version — accepts session_state in body.
    """
    logger.info(f"POST /mcq/status | doc_id={req.doc_id} | role={req.role}")
    if not req.session_state:
        logger.error("  Missing session_state in request body")
        return err("session_state is required.", 400)
    result = get_session_status(
        doc_id=req.doc_id,
        role=req.role,
        session_state=req.session_state,
    )
    if result["status"] != "success":
        return err(result.get("error", "Failed to get status"), 404)
    logger.info(f"  MCQ status OK | doc_id={req.doc_id} | all_complete={result.get('all_sessions_complete')}")
    return ok(result)


@app.post("/mcq/questions", tags=["MCQ"])
async def mcq_questions(req: GetQuestionsRequest):
    """
    Get questions for a session. Enforces timing rules.
    Requires session_state + questions_data from MongoDB.
    """
    logger.info(f"POST /mcq/questions | doc_id={req.doc_id} | session={req.session} | role={req.role}")
    if not req.session_state:
        return err("session_state is required.", 400)
    if not req.questions_data:
        return err("questions_data is required.", 400)
    result = get_questions(
        doc_id=req.doc_id,
        session=req.session,
        role=req.role,
        session_state=req.session_state,
        questions_data=req.questions_data,
    )
    if result["status"] != "success":
        logger.warning(f"  MCQ questions blocked | reason={result.get('reason')} | doc_id={req.doc_id}")
        return JSONResponse(
            status_code=403,
            content={"status": "error", **{k: v for k, v in result.items() if k != "status"}}
        )
    logger.info(f"  MCQ questions OK | session={req.session} | total={result.get('total')}")
    return ok(result)


@app.post("/mcq/override", tags=["MCQ"])
async def mcq_override(req: OverrideWindowRequest):
    """
    Override the Session 1 time window.
    Requires session_state from MongoDB.
    """
    logger.info(f"POST /mcq/override | doc_id={req.doc_id}")
    if not req.session_state:
        return err("session_state is required.", 400)
    result = override_window(doc_id=req.doc_id, session_state=req.session_state)
    if result["status"] != "success":
        return err(result.get("error", "Override failed"))
    logger.info(f"  MCQ override OK | doc_id={req.doc_id}")
    return ok(result)


@app.post("/mcq/submit", tags=["MCQ"])
async def mcq_submit(req: SubmitTestRequest):
    """
    Submit answers for a session. Returns score + updated state.
    Requires session_state + answer_key_data from MongoDB.
    """
    logger.info(f"POST /mcq/submit | doc_id={req.doc_id} | session={req.session} | role={req.role}")
    if not req.session_state:
        return err("session_state is required.", 400)
    if not req.answer_key_data:
        return err("answer_key_data is required.", 400)
    result = submit_test(
        doc_id=req.doc_id,
        session=req.session,
        user_answers=req.user_answers,
        role=req.role,
        session_state=req.session_state,
        answer_key_data=req.answer_key_data,
    )
    if result["status"] != "success":
        return err(result.get("error", "Submission failed"))
    logger.info(f"  MCQ submit OK | session={req.session} | correct={result.get('correct_count')}/{result.get('total_questions')}")
    return ok(result)


@app.post("/mcq/results", tags=["MCQ"])
async def mcq_results_post(req: MCQResultsRequest):
    """
    Get final results after all 3 sessions complete.
    Requires session_state from MongoDB.
    """
    logger.info(f"POST /mcq/results | doc_id={req.doc_id} | role={req.role}")
    if not req.session_state:
        return err("session_state is required.", 400)
    result = get_final_results(
        doc_id=req.doc_id,
        role=req.role,
        session_state=req.session_state,
        all_answer_keys=req.all_answer_keys or {},
    )
    if result["status"] != "success":
        return err(result.get("error", "Results not available yet"), 403)
    logger.info(f"  MCQ results OK | doc_id={req.doc_id} | avg={result.get('average_score_display')}")
    return ok(result)


# ── FILE 6 — Analytics ────────────────────────────────────────────────────────

@app.post("/analytics", tags=["Analytics"])
async def analytics(req: AnalyticsRequest):
    """
    Generate analytics for a completed document.
    All 3 sessions must be complete.
    Requires session_state, all_questions, all_answer_keys from MongoDB.
    """
    logger.info(f"POST /analytics | doc_id={req.doc_id} | role={req.role}")
    if not req.session_state:
        return err("session_state is required.", 400)

    # Convert string keys to int keys for all_questions / all_answer_keys
    # (MongoDB returns string keys, file6_analytics expects int keys)
    def normalize_keys(d):
        if not d:
            return {}
        result = {}
        for k, v in d.items():
            try:
                result[int(k)] = v
            except (ValueError, TypeError):
                result[k] = v
        return result

    all_q  = normalize_keys(req.all_questions)
    all_ak = normalize_keys(req.all_answer_keys)

    logger.info(f"  Analytics | sessions_in_questions={list(all_q.keys())} | sessions_in_answers={list(all_ak.keys())}")

    result = generate_analytics(
        doc_id=req.doc_id,
        role=req.role,
        session_state=req.session_state,
        all_questions=all_q,
        all_answer_keys=all_ak,
        cognitive_states=req.cognitive_states,
    )
    if result["status"] != "success":
        return err(result.get("error", "Analytics generation failed"))

    logger.info(f"  Analytics OK | doc_id={req.doc_id} | avg={result['analytics']['summary']['average_score_display']}")
    return ok({
        "analytics":    result["analytics"],
        "html_content": result["html_content"],
    })


# ── FILE 7 — Captions ─────────────────────────────────────────────────────────

@app.post("/captions", tags=["Captions"])
async def captions(req: CaptionsRequest):
    """
    Generate time-aligned captions from text + audio duration.
    Pure Python proportional timing — no AI needed.
    """
    logger.info(f"POST /captions | duration={req.duration_sec}s | words={len(req.text.split())}")
    if not req.text or not req.text.strip():
        return err("No text provided.")
    if req.duration_sec <= 0:
        return err("Invalid audio duration.")
    result = generate_captions(text=req.text, duration_sec=req.duration_sec)
    if result["status"] != "success":
        return err(result.get("error", "Caption generation failed"))
    logger.info(f"  Captions OK | segments={result['total_segments']} | method={result['method']}")
    return ok({
        "captions":       result["captions"],
        "total_segments": result["total_segments"],
        "duration_sec":   result["duration_sec"],
        "method":         result["method"],
    })


# ── Static file serving ───────────────────────────────────────────────────────

@app.get("/files/{folder}/{filename}", tags=["Files"])
async def serve_file(folder: str, filename: str):
    """Serve generated files to Express backend."""
    logger.info(f"GET /files/{folder}/{filename}")
    allowed = {"audio_cache", "reports", "analytics", "extracted", "mcq"}
    if folder not in allowed:
        raise HTTPException(status_code=403, detail=f"Folder '{folder}' not accessible.")
    file_path = PROJECT_ROOT / "storage" / folder / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")
    return FileResponse(
        path=str(file_path),
        filename=filename,
        media_type="application/octet-stream"
    )


# ── Pipeline: /pipeline/full (legacy) ────────────────────────────────────────

@app.post("/pipeline/full", tags=["Pipeline"])
async def full_pipeline(
    file:            UploadFile = File(...),
    cognitive_state: str        = Form("deep_focus"),
    document_title:  str        = Form("Tarang Document"),
    tts_engine:      str        = Form("edge"),
    voice_id:        str        = Form(""),
    role:            str        = Form("user"),
):
    """
    Legacy single-call pipeline: Extract → TTS → Modulate → Visualize → MCQ.
    Kept for backward compatibility. New code should use /pipeline/audio + /pipeline/mcq.
    """
    logger.info(f"POST /pipeline/full | file={file.filename} | state={cognitive_state} | engine={tts_engine} | role={role}")

    tmp_path = txt_path = wav_path = None
    suffix   = os.path.splitext(file.filename)[1]
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            shutil.copyfileobj(file.file, tmp)

        logger.info(f"  Step 1: Extracting text from {file.filename}")
        r1 = extract(filepath=tmp_path, save_output=False)
        if r1["status"] != "success":
            return err(f"Extraction failed: {r1.get('error')}")
        extracted_text = r1.get("text", "")
        logger.info(f"  Step 1 OK | words={r1['word_count']}")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w", encoding="utf-8") as tf:
            tf.write(extracted_text)
            txt_path = tf.name

        logger.info(f"  Step 2: TTS | engine={tts_engine}")
        r2 = generate_tts(source_txt_path=txt_path, engine=tts_engine, voice_id=voice_id or None)
        if r2["status"] != "success":
            return err(f"TTS failed: {r2.get('error')}")
        wav_path = r2["output_path"]
        logger.info(f"  Step 2 OK | duration={r2['duration_sec']}s")

        logger.info(f"  Step 3: Modulate | state={cognitive_state}")
        r3 = modulate_audio(input_wav_path=wav_path, cognitive_state=cognitive_state)
        if r3["status"] != "success":
            return err(f"Modulation failed: {r3.get('error')}")
        mod_path = r3["output_path"]
        logger.info(f"  Step 3 OK | beat={r3['beat_freq_hz']}Hz")

        logger.info(f"  Step 4: Visualize | role={role}")
        r4 = generate_visualization(
            modulated_wav_path=mod_path,
            role=role, cognitive_state=cognitive_state,
            beat_freq_hz=r3["beat_freq_hz"], document_title=document_title,
        )
        if r4["status"] != "success":
            return err(f"Visualization failed: {r4.get('error')}")
        logger.info(f"  Step 4 OK | viz={r4['report_type']}")

        logger.info(f"  Step 5: MCQ init | title={document_title}")
        r5 = initialise_document(text=extracted_text, document_title=document_title)
        if r5["status"] != "success":
            return err(f"MCQ init failed: {r5.get('error')}")
        doc_id = r5["document_id"]
        logger.info(f"  Step 5 OK | doc_id={doc_id}")

        logger.info(f"  /pipeline/full COMPLETE | doc_id={doc_id}")
        return ok({
            "doc_id":             doc_id,
            "document_title":     document_title,
            "extracted_text":     extracted_text,
            "cognitive_state":    cognitive_state,
            "beat_freq_hz":       r3["beat_freq_hz"],
            "duration_sec":       r3["duration_sec"],
            "word_count":         r1["word_count"],
            "sessions_generated": r5["sessions_generated"],
            "sessions_meta":      r5["sessions_meta"],
            "session_state":      r5["session_state"],
            "session_1_questions":r5["session_1_questions"],
            "session_1_answers":  r5["session_1_answers"],
            "session_2_questions":r5["session_2_questions"],
            "session_2_answers":  r5["session_2_answers"],
            "session_3_questions":r5["session_3_questions"],
            "session_3_answers":  r5["session_3_answers"],
        })
    finally:
        for p in [
            tmp_path if 'tmp_path' in locals() else None,
            txt_path if 'txt_path' in locals() else None,
        ]:
            try:
                if p and os.path.exists(p): os.unlink(p)
            except Exception as e:
                logger.warning(f"  Temp cleanup failed: {e}")


# ── Pipeline: /pipeline/audio (Phase 1) ──────────────────────────────────────

@app.post("/pipeline/audio", tags=["Pipeline"])
async def pipeline_audio(
    file:            UploadFile = File(...),
    cognitive_state: str        = Form("deep_focus"),
    document_title:  str        = Form("Tarang Document"),
    tts_engine:      str        = Form("edge"),
    voice_id:        str        = Form(""),
    role:            str        = Form("user"),
):
    """
    Phase 1 pipeline: Extract → TTS → Modulate → Captions.
    Returns MP3 as base64, extracted text, captions, duration.
    MCQ is NOT generated here — call /pipeline/mcq separately when user hits Play.
    """
    logger.info(
        f"POST /pipeline/audio | file={file.filename} | state={cognitive_state} "
        f"| engine={tts_engine} | role={role}"
    )

    tmp_path = txt_path = wav_path = mod_path = None
    suffix   = os.path.splitext(file.filename)[1]

    try:
        # ── Save upload ───────────────────────────────────────────────────
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            shutil.copyfileobj(file.file, tmp)
        logger.info(f"  Upload saved | tmp={tmp_path} | size={os.path.getsize(tmp_path)} bytes")

        # ── Step 1: Extract ───────────────────────────────────────────────
        logger.info(f"  Step 1: Extracting text...")
        r1 = extract(filepath=tmp_path, save_output=False)
        if r1["status"] != "success":
            logger.error(f"  Step 1 FAILED: {r1.get('error')}")
            return err(f"Extraction failed: {r1.get('error')}")
        extracted_text = r1.get("text") or ""
        word_count     = r1.get("word_count", 0)
        logger.info(f"  Step 1 OK | words={word_count} | chars={r1.get('char_count',0)}")

        if not extracted_text.strip():
            return err("Document extraction produced empty text. Check if document is image-based.")

        # ── Step 2: Write text to temp file for TTS ───────────────────────
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w", encoding="utf-8") as tf:
            tf.write(extracted_text)
            txt_path = tf.name
        logger.info(f"  Text written to: {txt_path}")

        # ── Step 3: TTS ───────────────────────────────────────────────────
        logger.info(f"  Step 2: TTS | engine={tts_engine} | voice={voice_id or 'default'}")
        r2 = generate_tts(
            source_txt_path=txt_path,
            engine=tts_engine,
            voice_id=voice_id or None,
        )
        if r2["status"] != "success":
            logger.error(f"  Step 2 FAILED: {r2.get('error')}")
            return err(f"TTS failed: {r2.get('error')}")
        wav_path = r2["output_path"]
        logger.info(f"  Step 2 OK | engine={r2['engine_used']} | duration={r2['duration_sec']}s | chunks={r2['chunks_total']}")

        # ── Step 4: Modulate ──────────────────────────────────────────────
        logger.info(f"  Step 3: Modulate | state={cognitive_state}")
        r3 = modulate_audio(input_wav_path=wav_path, cognitive_state=cognitive_state)
        if r3["status"] != "success":
            logger.error(f"  Step 3 FAILED: {r3.get('error')}")
            return err(f"Modulation failed: {r3.get('error')}")
        mod_path     = r3["output_path"]
        beat_freq_hz = r3["beat_freq_hz"]
        duration_sec = r3["duration_sec"]
        logger.info(f"  Step 3 OK | beat={beat_freq_hz}Hz | duration={duration_sec}s | output={mod_path}")

        # ── Step 5: Read MP3 as base64 ────────────────────────────────────
        logger.info(f"  Step 4: Encoding MP3 as base64...")
        if not os.path.exists(mod_path):
            return err(f"Modulated audio file not found at: {mod_path}")
        mp3_size = os.path.getsize(mod_path)
        with open(mod_path, "rb") as mf:
            mp3_b64 = base64.b64encode(mf.read()).decode("utf-8")
        logger.info(f"  Step 4 OK | mp3_size={mp3_size} bytes | b64_len={len(mp3_b64)} chars")

        # ── Step 6: Generate captions ─────────────────────────────────────
        logger.info(f"  Step 5: Generating captions | duration={duration_sec}s")
        cap_result = generate_captions(text=extracted_text, duration_sec=duration_sec)
        if cap_result.get("status") == "success":
            captions = cap_result.get("captions", [])
            logger.info(f"  Step 5 OK | segments={len(captions)} | method={cap_result.get('method')}")
        else:
            captions = []
            logger.warning(f"  Step 5 WARNING: Caption generation failed: {cap_result.get('error')} — continuing without captions")

        logger.info(
            f"  /pipeline/audio COMPLETE | words={word_count} | duration={duration_sec}s "
            f"| captions={len(captions)} | b64_kb={len(mp3_b64)//1024}KB"
        )

        return ok({
            "mp3_b64":         mp3_b64,
            "extracted_text":  extracted_text,
            "document_title":  document_title,
            "word_count":      word_count,
            "duration_sec":    duration_sec,
            "beat_freq_hz":    beat_freq_hz,
            "cognitive_state": cognitive_state,
            "captions":        captions,
        })

    except Exception as e:
        logger.error(f"  /pipeline/audio EXCEPTION: {type(e).__name__}: {e}", exc_info=True)
        return err(f"Pipeline audio failed: {str(e)}", 500)

    finally:
        # Clean up all temp files using locals() — safe and reliable
        for var_name, p in [("tmp_path", tmp_path), ("txt_path", txt_path)]:
            try:
                if p and os.path.exists(p):
                    os.unlink(p)
                    logger.info(f"  Cleaned up {var_name}: {p}")
            except Exception as e:
                logger.warning(f"  Cleanup failed for {var_name}: {e}")
        # mod_path (MP3) — keep if base64 encoding failed, otherwise clean
        if mod_path:
            try:
                if os.path.exists(mod_path):
                    os.unlink(mod_path)
                    logger.info(f"  Cleaned up mod_path: {mod_path}")
            except Exception as e:
                logger.warning(f"  Cleanup failed for mod_path: {e}")


# ── Pipeline: /pipeline/mcq (Phase 2) ────────────────────────────────────────

@app.post("/pipeline/mcq", tags=["Pipeline"])
async def pipeline_mcq(req: PipelineMCQRequest):
    """
    Phase 2 pipeline: Generate MCQ questions for all 3 sessions.
    Called in background when user clicks Play on the audio player.
    Takes extracted_text from MongoDB — no file reads needed.
    Returns all question/answer data as dicts for Express to store in MongoDB.
    """
    logger.info(
        f"POST /pipeline/mcq | doc_id={req.doc_id} "
        f"| words={len(req.extracted_text.split())} | title={req.document_title}"
    )

    txt_path = None
    try:
        # Write extracted text to temp file (initialise_document reads from file path)
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".txt", mode="w", encoding="utf-8"
        ) as tf:
            tf.write(req.extracted_text)
            txt_path = tf.name
        logger.info(f"  Text written to temp: {txt_path} | size={os.path.getsize(txt_path)} bytes")

        logger.info(f"  Calling initialise_document | title={req.document_title}")
        r5 = initialise_document(
            source_txt_path=txt_path,
            document_title=req.document_title,
        )

        if r5["status"] != "success":
            logger.error(f"  MCQ init FAILED: {r5.get('error')}")
            return err(f"MCQ init failed: {r5.get('error')}")

        doc_id = r5["document_id"]
        logger.info(
            f"  MCQ init OK | doc_id={doc_id} | sessions={r5['sessions_generated']} "
            f"| s1_qs={len(r5['session_1_questions'].get('questions',[]))} "
            f"| s2_qs={len(r5['session_2_questions'].get('questions',[]))} "
            f"| s3_qs={len(r5['session_3_questions'].get('questions',[]))}"
        )

        # Return everything directly from initialise_document() return value
        # NO disk reads — file5_mcq.py is fully stateless
        result = {
            "doc_id":             doc_id,
            "sessions_generated": r5["sessions_generated"],
            "session_state":      r5["session_state"],        # full state for MongoDB
            "sessions_meta":      r5["sessions_meta"],

            # Questions (without correct_answers — safe for frontend)
            "session_1_questions": r5["session_1_questions"],
            "session_2_questions": r5["session_2_questions"],
            "session_3_questions": r5["session_3_questions"],

            # Answer keys (with correct_answers — stored in MongoDB only)
            "session_1_answers":  r5["session_1_answers"],
            "session_2_answers":  r5["session_2_answers"],
            "session_3_answers":  r5["session_3_answers"],
        }

        logger.info(f"  /pipeline/mcq COMPLETE | doc_id={doc_id}")
        return ok(result)

    except Exception as e:
        logger.error(f"  /pipeline/mcq EXCEPTION: {type(e).__name__}: {e}", exc_info=True)
        return err(f"MCQ pipeline failed: {str(e)}", 500)

    finally:
        if txt_path:
            try:
                if os.path.exists(txt_path):
                    os.unlink(txt_path)
                    logger.info(f"  Cleaned up txt_path: {txt_path}")
            except Exception as e:
                logger.warning(f"  Cleanup failed: {e}")


# ── Global exception handler ──────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(
        f"UNHANDLED EXCEPTION | {request.method} {request.url.path} | "
        f"{type(exc).__name__}: {exc}",
        exc_info=True
    )
    return JSONResponse(
        status_code=500,
        content={"status": "error", "error": f"Internal server error: {str(exc)}"}
    )


# ── Startup / shutdown events ─────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    logger.info("=" * 60)
    logger.info("Tarang 1.0.0.2 — Python Bridge STARTING")
    logger.info(f"  PROJECT_ROOT : {PROJECT_ROOT}")
    logger.info(f"  Python       : {sys.version.split()[0]}")
    logger.info(f"  Endpoints    : /pipeline/audio, /pipeline/mcq, /health, /docs")
    logger.info("=" * 60)


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Tarang Python Bridge — shutting down")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    print("\nTarang 1.0.0.2 — Python Bridge")
    print("================================")
    print("Starting FastAPI on http://localhost:5001")
    print("Swagger UI:  http://localhost:5001/docs")
    print()
    uvicorn.run(
        "bridge:app",
        host="0.0.0.0",
        port=5001,
        reload=True,
        log_level="info",
    )

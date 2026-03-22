"""
Tarang 1.0.0.1 — bridge.py — FastAPI Microservice
STATELESS VERSION: No local storage/audio_cache, storage/mcq,
storage/analytics, storage/reports, or storage/extracted directories used.

All data flows:
  - Extracted text    → returned to Express → MongoDB Document.extractedText
  - TTS WAV           → temp dir only → deleted after modulation
  - Modulated MP3     → temp dir → uploaded to Cloudinary by Express → temp deleted
  - Viz HTML          → returned as string → Express → MongoDB Document
  - MCQ data          → returned as dicts → Express → MongoDB Session
  - Captions          → returned as list → Express → MongoDB Document
  - Analytics         → returned as dict + HTML string → Express → MongoDB

Only temp files exist during a pipeline run. TemporaryDirectory is deleted
automatically when the pipeline endpoint returns.

Run with:
  uvicorn bridge:app --host 0.0.0.0 --port 5001 --reload
"""

import os
import sys
import json
import shutil
import logging
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [bridge] %(levelname)s — %(message)s"
)
logger = logging.getLogger("bridge")

# ── One temp dir for uploaded PDFs during pipeline — cleaned per request ──────
UPLOAD_DIR = Path(tempfile.gettempdir()) / "tarang_uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="Tarang Core Engine Bridge",
    description="Stateless FastAPI microservice — no local storage. All data returned to Express/MongoDB.",
    version="1.0.0.1",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production to your Render/Firebase URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic models ───────────────────────────────────────────────────────────

class TTSRequest(BaseModel):
    text: str
    engine: Optional[str] = "pyttsx3"
    voice_id: Optional[str] = None

class ModulateRequest(BaseModel):
    tts_wav_path: str
    cognitive_state: Optional[str] = "deep_focus"
    custom_beat_freq: Optional[float] = None
    custom_depth: Optional[float] = None

class VisualizeRequest(BaseModel):
    modulated_mp3_path: str
    role: Optional[str] = "user"
    cognitive_state: Optional[str] = "deep_focus"
    beat_freq_hz: Optional[float] = 14.0
    document_title: Optional[str] = "Tarang Session"
    session_number: Optional[int] = 1

class MCQInitRequest(BaseModel):
    text: str
    document_title: Optional[str] = "Tarang Document"
    num_questions: Optional[int] = 10
    custom_s1_to_s2_hours: Optional[float] = 12.0
    custom_s2_to_s3_hours: Optional[float] = 24.0

class AudioCompletedRequest(BaseModel):
    doc_id: str
    role: Optional[str] = "user"
    session_state: dict

class GetQuestionsRequest(BaseModel):
    doc_id: str
    session: int
    role: Optional[str] = "user"
    session_state: dict
    questions_data: dict

class OverrideWindowRequest(BaseModel):
    doc_id: str
    session_state: dict

class SubmitTestRequest(BaseModel):
    doc_id: str
    session: int
    user_answers: dict
    role: Optional[str] = "user"
    session_state: dict
    answer_key_data: dict

class FinalResultsRequest(BaseModel):
    doc_id: str
    role: Optional[str] = "user"
    session_state: dict
    all_answer_keys: Optional[dict] = None

class AnalyticsRequest(BaseModel):
    doc_id: str
    role: Optional[str] = "user"
    session_state: dict
    all_questions: dict        # {1: questions_dict, 2: ..., 3: ...}
    all_answer_keys: dict      # {1: answers_dict, 2: ..., 3: ...}
    cognitive_states: Optional[dict] = None

class CaptionsRequest(BaseModel):
    text: str
    duration_sec: float

class SessionStatusRequest(BaseModel):
    doc_id: str
    role: Optional[str] = "user"
    session_state: dict


# ── Response helpers ──────────────────────────────────────────────────────────

def ok(data: dict) -> JSONResponse:
    return JSONResponse(content={"status": "success", "data": data})

def err(message: str, code: int = 400) -> JSONResponse:
    return JSONResponse(status_code=code, content={"status": "error", "error": message})


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
async def health():
    return {"status": "ok", "service": "Tarang Python Bridge", "version": "1.0.0.1", "storage": "stateless"}

@app.get("/", tags=["System"])
async def root():
    return {"service": "Tarang Core Engine Bridge", "version": "1.0.0.1", "docs": "/docs"}

@app.get("/voices", tags=["Core Engine"])
async def list_voices():
    result = get_voices()
    if result["status"] != "success":
        return err(result.get("error", "Failed to list voices"))
    return ok({"voices": result["voices"], "count": result["count"]})


# ── FILE 1 — Extract ──────────────────────────────────────────────────────────

@app.post("/extract", tags=["Core Engine"])
async def extract_document(file: UploadFile = File(...)):
    """
    Upload document → extract text → return text string.
    No local file saved. Express stores text in MongoDB Document.extractedText.
    Uploaded file saved to system temp dir, deleted after extraction.
    """
    logger.info(f"Extract request | filename={file.filename}")

    # Save upload to temp location
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=Path(file.filename).suffix,
        dir=tempfile.gettempdir(),
        prefix="tarang_upload_"
    ) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)

    try:
        result = extract(filepath=str(tmp_path), save_output=False)
    finally:
        tmp_path.unlink(missing_ok=True)  # delete upload immediately

    if result["status"] != "success":
        return err(result.get("error", "Extraction failed"))

    logger.info(f"Extraction OK | words={result['word_count']}")
    return ok({
        "text":       result["text"],       # Express stores in MongoDB
        "word_count": result["word_count"],
        "char_count": result["char_count"],
        "format":     result["format"],
        "metadata":   result["metadata"],
        "timestamp":  result["timestamp"],
    })


# ── FILE 5 — MCQ (stateless — session_state passed in from Express/MongoDB) ───

@app.post("/mcq/init", tags=["MCQ"])
async def mcq_init(req: MCQInitRequest):
    """
    Generate all 3 MCQ sessions from text.
    Returns all data as dicts — Express stores in MongoDB.
    No local files written.
    """
    logger.info(f"MCQ init | title={req.document_title}")
    result = initialise_document(
        text=req.text,
        document_title=req.document_title,
        num_questions=req.num_questions,
        custom_s1_to_s2_hours=req.custom_s1_to_s2_hours,
        custom_s2_to_s3_hours=req.custom_s2_to_s3_hours,
    )
    if result["status"] != "success":
        return err(result.get("error", "MCQ init failed"))
    logger.info(f"MCQ init OK | doc_id={result['document_id']}")
    return ok(result)


@app.post("/mcq/audio-completed", tags=["MCQ"])
async def mcq_audio_completed(req: AudioCompletedRequest):
    """session_state from MongoDB passed in. Returns updated_state for Express to save."""
    result = audio_completed(doc_id=req.doc_id, role=req.role, session_state=req.session_state)
    if result["status"] != "success":
        return err(result.get("error", "Failed"))
    return ok(result)


@app.post("/mcq/status", tags=["MCQ"])
async def mcq_status(req: SessionStatusRequest):
    """session_state from MongoDB passed in. Returns status + updated_state if changed."""
    result = get_session_status(doc_id=req.doc_id, role=req.role, session_state=req.session_state)
    if result["status"] != "success":
        return err(result.get("error", "Failed"), code=404)
    return ok(result)


@app.post("/mcq/questions", tags=["MCQ"])
async def mcq_questions(req: GetQuestionsRequest):
    """session_state + questions_data from MongoDB passed in."""
    result = get_questions(
        doc_id=req.doc_id, session=req.session, role=req.role,
        session_state=req.session_state, questions_data=req.questions_data,
    )
    if result["status"] != "success":
        return JSONResponse(
            status_code=403,
            content={"status": "error", **{k: v for k, v in result.items() if k != "status"}}
        )
    return ok(result)


@app.post("/mcq/override", tags=["MCQ"])
async def mcq_override(req: OverrideWindowRequest):
    """session_state from MongoDB passed in. Returns updated_state."""
    result = override_window(doc_id=req.doc_id, session_state=req.session_state)
    if result["status"] != "success":
        return err(result.get("error", "Override failed"))
    return ok(result)


@app.post("/mcq/submit", tags=["MCQ"])
async def mcq_submit(req: SubmitTestRequest):
    """session_state + answer_key_data from MongoDB passed in. Returns updated_state."""
    result = submit_test(
        doc_id=req.doc_id, session=req.session,
        user_answers=req.user_answers, role=req.role,
        session_state=req.session_state, answer_key_data=req.answer_key_data,
    )
    if result["status"] != "success":
        return err(result.get("error", "Submission failed"))
    return ok(result)


@app.post("/mcq/results", tags=["MCQ"])
async def mcq_results(req: FinalResultsRequest):
    """session_state from MongoDB passed in."""
    result = get_final_results(
        doc_id=req.doc_id, role=req.role,
        session_state=req.session_state,
        all_answer_keys=req.all_answer_keys,
    )
    if result["status"] != "success":
        return err(result.get("error", "Results not available"), code=403)
    return ok(result)


# ── FILE 6 — Analytics ────────────────────────────────────────────────────────

@app.post("/analytics", tags=["Analytics"])
async def analytics(req: AnalyticsRequest):
    """
    All data passed in from MongoDB via Express.
    Returns analytics dict + html_content string.
    Express stores both in MongoDB.
    """
    logger.info(f"Analytics request | doc_id={req.doc_id} | role={req.role}")

    # Convert string keys to int keys for all_questions / all_answer_keys
    all_questions   = {int(k): v for k, v in req.all_questions.items()}
    all_answer_keys = {int(k): v for k, v in req.all_answer_keys.items()}

    result = generate_analytics(
        doc_id=req.doc_id, role=req.role,
        session_state=req.session_state,
        all_questions=all_questions,
        all_answer_keys=all_answer_keys,
        cognitive_states=req.cognitive_states,
    )
    if result["status"] != "success":
        return err(result.get("error", "Analytics failed"))

    logger.info(f"Analytics OK | doc_id={req.doc_id}")
    return ok({
        "analytics":    result["analytics"],
        "html_content": result["html_content"],  # Express stores in MongoDB
    })


# ── FILE 7 — Captions ─────────────────────────────────────────────────────────

@app.post("/captions", tags=["Captions"])
async def captions(req: CaptionsRequest):
    """Pure compute — no files. Returns caption list for Express to store in MongoDB."""
    logger.info(f"Captions request | duration={req.duration_sec}s | words={len(req.text.split())}")
    if not req.text or not req.text.strip():
        return err("No text provided.")
    if req.duration_sec <= 0:
        return err("Invalid audio duration.")

    result = generate_captions(text=req.text, duration_sec=req.duration_sec)
    if result["status"] != "success":
        return err(result.get("error", "Caption generation failed"))

    logger.info(f"Captions OK | segments={result['total_segments']} | wps={result.get('wps')}")
    return ok({
        "captions":       result["captions"],
        "total_segments": result["total_segments"],
        "duration_sec":   result["duration_sec"],
        "wps":            result.get("wps"),
        "method":         result["method"],
    })


# ── PHASE 1: Audio pipeline (fast — user sees player immediately) ─────────────
# Extract → TTS → Modulate → Captions → return MP3 + captions to Express.
# Express uploads MP3 to Cloudinary and shows the player.
# MCQ generation does NOT happen here.

@app.post("/pipeline/audio", tags=["Pipeline"])
async def pipeline_audio(
    file:            UploadFile = File(...),
    cognitive_state: str  = Form("deep_focus"),
    document_title:  str  = Form("Tarang Document"),
    tts_engine:      str  = Form("pyttsx3"),
    voice_id:        str  = Form(""),
    role:            str  = Form("user"),
):
    """
    PHASE 1 — fast path. Returns audio + captions only.
    Target: user sees the player within ~20s regardless of document length.

    Express must after receiving this response:
      1. Upload mp3_b64 to Cloudinary → store audioCloudUrl in MongoDB
      2. Store extractedText, captions in MongoDB Document
      3. Set pipelineStatus = "audio_ready"
      4. Redirect frontend to /listen/:docId immediately

    MCQ generation is triggered separately via POST /pipeline/mcq
    when the user clicks Play (fired by Express after frontend notifies it).
    """
    logger.info(f"Phase 1 audio pipeline | file={file.filename} | state={cognitive_state}")

    with tempfile.TemporaryDirectory(prefix="tarang_audio_") as tmp:
        tmp_dir = Path(tmp)

        # ── Save upload ───────────────────────────────────────────────────
        upload_path = tmp_dir / file.filename
        try:
            with open(upload_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
        except Exception as e:
            return err(f"Upload failed: {e}")

        # ── Extract ───────────────────────────────────────────────────────
        r1 = extract(filepath=str(upload_path), save_output=False)
        if r1["status"] != "success":
            return err(f"Extraction failed: {r1.get('error')}")
        extracted_text = r1["text"]
        upload_path.unlink(missing_ok=True)
        logger.info(f"Step 1 extract OK | words={r1['word_count']}")

        # ── TTS ───────────────────────────────────────────────────────────
        r2 = generate_tts(
            text=extracted_text,
            engine=tts_engine,
            voice_id=voice_id or None,
            output_dir=tmp_dir,
        )
        if r2["status"] != "success":
            return err(f"TTS failed: {r2.get('error')}")
        logger.info(f"Step 2 TTS OK | duration={r2['duration_sec']}s")

        # ── Modulate ──────────────────────────────────────────────────────
        r3 = modulate_audio(
            input_wav_path=r2["output_path"],
            cognitive_state=cognitive_state,
            output_dir=tmp_dir,
        )
        if r3["status"] != "success":
            return err(f"Modulation failed: {r3.get('error')}")
        mp3_path = r3["output_path"]
        logger.info(f"Step 3 modulate OK | beat={r3['beat_freq_hz']}Hz | duration={r3['duration_sec']}s")

        # ── Captions — pure compute, instant ─────────────────────────────
        r_caps = generate_captions(text=extracted_text, duration_sec=r3["duration_sec"])
        captions_data = r_caps.get("captions", []) if r_caps.get("status") == "success" else []
        logger.info(f"Step 4 captions OK | segments={len(captions_data)} | wps={r_caps.get('wps','?')}")

        # ── Read MP3 → base64 for Express to upload to Cloudinary ─────────
        import base64
        try:
            mp3_bytes = Path(mp3_path).read_bytes()
            mp3_b64   = base64.b64encode(mp3_bytes).decode("utf-8")
        except Exception as e:
            return err(f"Failed to read MP3: {e}")

        logger.info(f"Phase 1 complete | mp3_size={len(mp3_bytes)//1024}KB | captions={len(captions_data)}")

        return ok({
            "phase":          "audio",
            "document_title": document_title,
            "extracted_text": extracted_text,
            "word_count":     r1["word_count"],
            "mp3_b64":        mp3_b64,
            "mp3_filename":   Path(mp3_path).name,
            "cognitive_state":cognitive_state,
            "beat_freq_hz":   r3["beat_freq_hz"],
            "duration_sec":   r3["duration_sec"],
            "captions":       captions_data,
            "captions_wps":   r_caps.get("wps"),
            # doc_id not yet assigned — Express generates it or waits for MCQ phase
            "message": "Audio ready. Show player. Trigger /pipeline/mcq when user clicks play.",
        })


# ── PHASE 2: MCQ generation (triggered when user clicks Play) ─────────────────
# Express calls this endpoint in the background when the user clicks Play.
# Returns all MCQ data for Express to store in MongoDB.
# Frontend is already showing the player — user doesn't wait for this.

class MCQPipelineRequest(BaseModel):
    extracted_text:  str
    document_title:  str  = "Tarang Document"
    doc_id:          str  = ""   # MongoDB docId to associate results with

@app.post("/pipeline/mcq", tags=["Pipeline"])
async def pipeline_mcq(req: MCQPipelineRequest):
    """
    PHASE 2 — background MCQ generation.
    Called by Express when user clicks Play (fire-and-store pattern).
    Express does NOT wait for this before showing the player.
    Express calls this, stores the result in MongoDB, then the
    frontend polls /api/sessions/:docId/status to know when MCQs are ready.
    """
    logger.info(f"Phase 2 MCQ pipeline | doc_id={req.doc_id} | title={req.document_title}")

    if not req.extracted_text.strip():
        return err("No extracted text provided.")

    r5 = initialise_document(
        text=req.extracted_text,
        document_title=req.document_title,
    )
    if r5["status"] != "success":
        return err(f"MCQ init failed: {r5.get('error')}")

    logger.info(f"Phase 2 MCQ complete | doc_id={r5['document_id']}")

    return ok({
        "phase":               "mcq",
        "doc_id":              req.doc_id or r5["document_id"],
        "sessions_generated":  r5["sessions_generated"],
        "sessions_meta":       r5["sessions_meta"],
        "session_state":       r5["session_state"],
        "session_1_questions": r5["session_1_questions"],
        "session_1_answers":   r5["session_1_answers"],
        "session_2_questions": r5["session_2_questions"],
        "session_2_answers":   r5["session_2_answers"],
        "session_3_questions": r5["session_3_questions"],
        "session_3_answers":   r5["session_3_answers"],
        "s1_to_s2_hours":      r5["s1_to_s2_hours"],
        "s2_to_s3_hours":      r5["s2_to_s3_hours"],
        "message":             "MCQ sessions generated. Store in MongoDB Session collection.",
    })


# ── Legacy full pipeline (kept for compatibility) ────────────────────────────
# Calls Phase 1 + Phase 2 sequentially. Use for admin testing only.
# Users should use /pipeline/audio + /pipeline/mcq separately.

@app.post("/pipeline/full", tags=["Pipeline"])
async def full_pipeline(
    file:            UploadFile = File(...),
    cognitive_state: str  = Form("deep_focus"),
    document_title:  str  = Form("Tarang Document"),
    tts_engine:      str  = Form("pyttsx3"),
    voice_id:        str  = Form(""),
    role:            str  = Form("user"),
):
    """Legacy endpoint — calls audio + MCQ sequentially. Slow. Use /pipeline/audio instead."""
    logger.info(f"Full pipeline (legacy) | file={file.filename}")

    with tempfile.TemporaryDirectory(prefix="tarang_pipeline_") as tmp:
        tmp_dir = Path(tmp)

        upload_path = tmp_dir / file.filename
        try:
            with open(upload_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
        except Exception as e:
            return err(f"Upload failed: {e}")

        r1 = extract(filepath=str(upload_path), save_output=False)
        if r1["status"] != "success":
            return err(f"Extraction failed: {r1.get('error')}")
        extracted_text = r1["text"]
        upload_path.unlink(missing_ok=True)

        r2 = generate_tts(text=extracted_text, engine=tts_engine,
                          voice_id=voice_id or None, output_dir=tmp_dir)
        if r2["status"] != "success":
            return err(f"TTS failed: {r2.get('error')}")

        r3 = modulate_audio(input_wav_path=r2["output_path"],
                            cognitive_state=cognitive_state, output_dir=tmp_dir)
        if r3["status"] != "success":
            return err(f"Modulation failed: {r3.get('error')}")
        mp3_path = r3["output_path"]

        r_caps = generate_captions(text=extracted_text, duration_sec=r3["duration_sec"])
        captions_data = r_caps.get("captions", []) if r_caps.get("status") == "success" else []

        r5 = initialise_document(text=extracted_text, document_title=document_title)
        if r5["status"] != "success":
            return err(f"MCQ init failed: {r5.get('error')}")

        import base64
        mp3_bytes = Path(mp3_path).read_bytes()
        mp3_b64   = base64.b64encode(mp3_bytes).decode("utf-8")

        logger.info(f"Full pipeline complete | doc_id={r5['document_id']} | mp3={len(mp3_bytes)//1024}KB")

        return ok({
            "doc_id":              r5["document_id"],
            "document_title":      document_title,
            "extracted_text":      extracted_text,
            "word_count":          r1["word_count"],
            "mp3_b64":             mp3_b64,
            "mp3_filename":        Path(mp3_path).name,
            "cognitive_state":     cognitive_state,
            "beat_freq_hz":        r3["beat_freq_hz"],
            "duration_sec":        r3["duration_sec"],
            "visualization_html":  None,
            "visualization_type":  None,
            "captions":            captions_data,
            "captions_wps":        r_caps.get("wps"),
            "sessions_generated":  r5["sessions_generated"],
            "sessions_meta":       r5["sessions_meta"],
            "session_state":       r5["session_state"],
            "session_1_questions": r5["session_1_questions"],
            "session_1_answers":   r5["session_1_answers"],
            "session_2_questions": r5["session_2_questions"],
            "session_2_answers":   r5["session_2_answers"],
            "session_3_questions": r5["session_3_questions"],
            "session_3_answers":   r5["session_3_answers"],
            "s1_to_s2_hours":      r5["s1_to_s2_hours"],
            "s2_to_s3_hours":      r5["s2_to_s3_hours"],
            "message":             "Pipeline complete (legacy mode).",
        })


# ── Exception handler ─────────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"status": "error", "error": f"Internal server error: {str(exc)}"}
    )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    print("\nTarang 1.0.0.1 — Python Bridge (STATELESS)")
    print("============================================")
    print("No local storage used. All data returned to Express/MongoDB.")
    print("Starting FastAPI on http://localhost:5001")
    print("Swagger UI: http://localhost:5001/docs")
    print()
    uvicorn.run("bridge:app", host="0.0.0.0", port=5001, reload=True, log_level="info")

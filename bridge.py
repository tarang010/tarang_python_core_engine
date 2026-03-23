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
  - Handle file storage paths consistently
  - Provide a health check endpoint for Express to ping on startup

All endpoints return:
  { status: "success" | "error", data: {...}, error?: "..." }

Run with:
  uvicorn bridge:app --host 0.0.0.0 --port 5001 --reload

Install dependencies:
  pip install fastapi uvicorn python-multipart aiofiles
"""

import os
import sys
import json
import shutil
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel

# ── Add project root to path so core engine files are importable ──────────────
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Core engine imports ───────────────────────────────────────────────────────
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

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [bridge] %(levelname)s — %(message)s"
)
logger = logging.getLogger("bridge")

# ── Storage dirs ──────────────────────────────────────────────────────────────
UPLOAD_DIR = PROJECT_ROOT / "storage" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Tarang Core Engine Bridge",
    description=(
        "FastAPI microservice exposing Tarang's Python core engine "
        "to the Express.js backend. Runs on localhost:5001."
    ),
    version="1.0.0.1",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS — allow Express backend on localhost:3000 / :5000 ───────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5000",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic request models ───────────────────────────────────────────────────

class TTSRequest(BaseModel):
    extracted_txt_path: str
    engine: Optional[str] = "pyttsx3"
    output_filename: Optional[str] = None
    voice_id: Optional[str] = None


class ModulateRequest(BaseModel):
    tts_wav_path: str
    cognitive_state: Optional[str] = "deep_focus"
    output_filename: Optional[str] = None
    custom_beat_freq: Optional[float] = None
    custom_depth: Optional[float] = None


class VisualizeRequest(BaseModel):
    modulated_wav_path: str
    role: Optional[str] = "user"
    cognitive_state: Optional[str] = "deep_focus"
    beat_freq_hz: Optional[float] = 14.0
    document_title: Optional[str] = "Tarang Session"
    session_number: Optional[int] = 1
    output_stem: Optional[str] = None


class MCQInitRequest(BaseModel):
    extracted_txt_path: str
    document_title: Optional[str] = "Tarang Document"
    num_questions: Optional[int] = 10
    custom_s1_to_s2_hours: Optional[float] = 12.0
    custom_s2_to_s3_hours: Optional[float] = 24.0


class AudioCompletedRequest(BaseModel):
    doc_id: str
    role: Optional[str] = "user"


class GetQuestionsRequest(BaseModel):
    doc_id: str
    session: int
    role: Optional[str] = "user"


class OverrideWindowRequest(BaseModel):
    doc_id: str


class SubmitTestRequest(BaseModel):
    doc_id: str
    session: int
    user_answers: dict   # { "q001": ["A"], "q002": ["B", "C"] }
    role: Optional[str] = "user"


class AnalyticsRequest(BaseModel):
    doc_id: str
    role: Optional[str] = "user"
    cognitive_states: Optional[dict] = None


class CaptionsRequest(BaseModel):
    text:         str
    duration_sec: float


# ── Response wrapper ──────────────────────────────────────────────────────────

def ok(data: dict) -> JSONResponse:
    return JSONResponse(content={"status": "success", "data": data})


def err(message: str, code: int = 400) -> JSONResponse:
    return JSONResponse(
        status_code=code,
        content={"status": "error", "error": message}
    )


# ── Health check ──────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
async def health():
    """Express backend pings this on startup to confirm bridge is alive."""
    return {"status": "ok", "service": "Tarang Python Bridge", "version": "1.0.0.1"}


@app.get("/voices", tags=["Core Engine"])
async def list_voices():
    """
    Returns all available pyttsx3 voices on this machine.
    Called by the frontend upload page to populate the voice selector.
    Only relevant when engine = pyttsx3.
    """
    result = get_voices()
    if result["status"] != "success":
        return err(result.get("error", "Failed to list voices"))
    return ok({"voices": result["voices"], "count": result["count"]})


@app.get("/", tags=["System"])
async def root():
    return {
        "service": "Tarang Core Engine Bridge",
        "version": "1.0.0.1",
        "docs":    "http://localhost:5001/docs",
    }


# ── FILE 1 — Document extraction ──────────────────────────────────────────────

@app.post("/extract", tags=["Core Engine"])
async def extract_document(file: UploadFile = File(...)):
    """
    Upload a document (PDF, DOCX, TXT, MD).
    Extracts clean text and saves to storage/extracted/.
    Returns extracted text path, word count, char count.
    """
    logger.info(f"Extract request | filename={file.filename}")

    # Save upload to temp storage
    suffix   = Path(file.filename).suffix.lower()
    save_path = UPLOAD_DIR / file.filename
    try:
        with open(save_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as e:
        return err(f"Failed to save uploaded file: {e}")

    # Call file1
    result = extract(filepath=str(save_path), save_output=True)

    if result["status"] != "success":
        return err(result.get("error", "Extraction failed"))

    logger.info(f"Extraction OK | words={result['word_count']} | path={result['output_path']}")
    return ok({
        "output_path":  result["output_path"],
        "word_count":   result["word_count"],
        "char_count":   result["char_count"],
        "format":       result["format"],
        "metadata":     result["metadata"],
        "timestamp":    result["timestamp"],
    })


# ── FILE 2 — TTS generation ───────────────────────────────────────────────────

@app.post("/tts", tags=["Core Engine"])
async def generate_tts_audio(req: TTSRequest):
    """
    Generate TTS audio from an extracted .txt file.
    Returns path to raw WAV file, duration, engine used.
    """
    logger.info(f"TTS request | path={req.extracted_txt_path} | engine={req.engine}")

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

    logger.info(f"TTS OK | engine={result['engine_used']} | duration={result['duration_sec']}s")
    return ok({
        "output_path":  result["output_path"],
        "engine_used":  result["engine_used"],
        "duration_sec": result["duration_sec"],
        "word_count":   result["word_count"],
        "chunks_total": result["chunks_total"],
        "timestamp":    result["timestamp"],
    })


# ── FILE 3 — Binaural modulation ──────────────────────────────────────────────

@app.post("/modulate", tags=["Core Engine"])
async def modulate(req: ModulateRequest):
    """
    Apply binaural AM modulation to a raw TTS WAV file.
    Returns path to modulated stereo WAV.
    """
    logger.info(f"Modulate request | path={req.tts_wav_path} | state={req.cognitive_state}")

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

    logger.info(f"Modulate OK | beat={result['beat_freq_hz']}Hz | duration={result['duration_sec']}s")
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
    """
    Generate audio visualization HTML.
    Admin role → full frequency analysis report.
    User role  → animated neon waveform.
    Returns path to generated HTML file.
    """
    logger.info(f"Visualize request | role={req.role} | state={req.cognitive_state}")

    if not Path(req.modulated_wav_path).exists():
        return err(f"WAV file not found: {req.modulated_wav_path}")

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

    logger.info(f"Visualize OK | type={result['report_type']} | path={result['output_path']}")
    return ok({
        "output_path": result["output_path"],
        "report_type": result["report_type"],
        "role":        result["role"],
        "timestamp":   result["timestamp"],
    })


# ── FILE 5 — MCQ session management ──────────────────────────────────────────

@app.post("/mcq/init", tags=["MCQ"])
async def mcq_init(req: MCQInitRequest):
    """
    Initialise a document for MCQ sessions.
    Generates all 3 question sets (Easy / Medium / Hard).
    Returns document_id for all subsequent MCQ calls.
    """
    logger.info(f"MCQ init | path={req.extracted_txt_path} | title={req.document_title}")

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

    logger.info(f"MCQ init OK | doc_id={result['document_id']}")
    return ok(result)


@app.post("/mcq/audio-completed", tags=["MCQ"])
async def mcq_audio_completed(req: AudioCompletedRequest):
    """
    Mark audio as completed for a document.
    Starts the 10-minute Session 1 window (bypassed for admin role).
    """
    logger.info(f"MCQ audio completed | doc_id={req.doc_id} | role={req.role}")
    result = audio_completed(doc_id=req.doc_id, role=req.role)
    if result["status"] != "success":
        return err(result.get("error", "Failed to mark audio complete"))
    return ok(result)


@app.get("/mcq/status/{doc_id}", tags=["MCQ"])
async def mcq_status(doc_id: str, role: str = "user"):
    """
    Get full session status for a document.
    Returns which sessions are available, locked, completed,
    time remaining until next session unlocks, messages, etc.
    """
    logger.info(f"MCQ status | doc_id={doc_id} | role={role}")
    result = get_session_status(doc_id=doc_id, role=role)
    if result["status"] != "success":
        return err(result.get("error", "Failed to get status"), code=404)
    return ok(result)


@app.post("/mcq/questions", tags=["MCQ"])
async def mcq_questions(req: GetQuestionsRequest):
    """
    Get questions for a specific session.
    Enforces timing rules for normal users.
    Admin role bypasses time gaps.
    Returns error with can_override=True if Session 1 window expired.
    """
    logger.info(f"MCQ questions | doc_id={req.doc_id} | session={req.session} | role={req.role}")
    result = get_questions(doc_id=req.doc_id, session=req.session, role=req.role)
    if result["status"] != "success":
        # Pass through timing errors with full detail for frontend to handle
        return JSONResponse(
            status_code=403,
            content={"status": "error", **{k: v for k, v in result.items() if k != "status"}}
        )
    return ok(result)


@app.post("/mcq/override", tags=["MCQ"])
async def mcq_override(req: OverrideWindowRequest):
    """
    Called when user clicks 'I have listened carefully and I am ready.'
    Overrides the 10-minute Session 1 window.
    """
    logger.info(f"MCQ override | doc_id={req.doc_id}")
    result = override_window(doc_id=req.doc_id)
    if result["status"] != "success":
        return err(result.get("error", "Override failed"))
    return ok(result)


@app.post("/mcq/submit", tags=["MCQ"])
async def mcq_submit(req: SubmitTestRequest):
    """
    Submit answers for a session.
    Scores the test, updates session state.
    Scores are hidden until all 3 sessions complete.
    Returns re-listen recommendation if score < 30%.

    user_answers format: { "q001": ["A"], "q002": ["B", "C"] }
    """
    logger.info(f"MCQ submit | doc_id={req.doc_id} | session={req.session} | role={req.role}")
    result = submit_test(
        doc_id=req.doc_id,
        session=req.session,
        user_answers=req.user_answers,
        role=req.role,
    )
    if result["status"] != "success":
        return err(result.get("error", "Submission failed"))
    return ok(result)


@app.get("/mcq/results/{doc_id}", tags=["MCQ"])
async def mcq_results(doc_id: str, role: str = "user"):
    """
    Get final results including all scores and unlocked correct answers.
    Only available after all 3 sessions are complete.
    """
    logger.info(f"MCQ results | doc_id={doc_id} | role={role}")
    result = get_final_results(doc_id=doc_id, role=role)
    if result["status"] != "success":
        return err(result.get("error", "Results not available yet"), code=403)
    return ok(result)


# ── FILE 6 — Analytics ────────────────────────────────────────────────────────

@app.post("/captions", tags=["Captions"])
async def captions(req: CaptionsRequest):
    """
    Generate time-aligned captions from extracted text + audio duration.
    Pure Python proportional timing — no AI, no extra dependencies.
    Uses text already stored in MongoDB + duration from pipeline output.
    Results cached in MongoDB after first call.
    """
    logger.info(f"Captions request | duration={req.duration_sec}s | words={len(req.text.split())}")

    if not req.text or not req.text.strip():
        return err("No text provided.")

    if req.duration_sec <= 0:
        return err("Invalid audio duration.")

    result = generate_captions(text=req.text, duration_sec=req.duration_sec)

    if result["status"] != "success":
        return err(result.get("error", "Caption generation failed"))

    logger.info(
        f"Captions OK | segments={result['total_segments']} | "
        f"duration={result['duration_sec']}s | method={result['method']}"
    )
    return ok({
        "captions":       result["captions"],
        "total_segments": result["total_segments"],
        "duration_sec":   result["duration_sec"],
        "method":         result["method"],
    })


@app.post("/analytics", tags=["Analytics"])
async def analytics(req: AnalyticsRequest):
    """
    Generate analytics for a completed document.
    All 3 sessions must be complete.
    Returns JSON analytics + paths to JSON and HTML report files.
    Admin role gets additional detail (question meta, timing config).
    """
    logger.info(f"Analytics request | doc_id={req.doc_id} | role={req.role}")
    result = generate_analytics(
        doc_id=req.doc_id,
        role=req.role,
        cognitive_states=req.cognitive_states,
    )
    if result["status"] != "success":
        return err(result.get("error", "Analytics generation failed"))

    logger.info(f"Analytics OK | doc_id={req.doc_id}")
    return ok({
        "analytics": result["analytics"],
        "json_path": result["json_path"],
        "html_path": result["html_path"],
    })


# ── Static file serving ───────────────────────────────────────────────────────

@app.get("/files/{folder}/{filename}", tags=["Files"])
async def serve_file(folder: str, filename: str):
    """
    Serve generated files (audio, reports, visualizations) to Express.
    Express then streams them to the React frontend.

    Accessible folders: audio_cache, reports, analytics, extracted
    """
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


# ── Full pipeline endpoint ────────────────────────────────────────────────────

@app.post("/pipeline/full", tags=["Pipeline"])
async def full_pipeline(
    file: UploadFile = File(...),
    cognitive_state: str = Form("deep_focus"),
    document_title:  str = Form("Tarang Document"),
    tts_engine:      str = Form("pyttsx3"),
    voice_id:        str = Form(""),
    role:            str = Form("user"),
):
    """
    Single endpoint that runs the complete pipeline:
      Upload doc → Extract → TTS → Modulate → Visualize → MCQ Init

    Returns all output paths in one response.
    Use this for the main document upload flow in the React frontend.
    """
    logger.info(
        f"Full pipeline | file={file.filename} | state={cognitive_state} "
        f"| title={document_title} | role={role}"
    )

    # ── Step 1: Save upload ───────────────────────────────────────────────
    save_path = UPLOAD_DIR / file.filename
    try:
        with open(save_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as e:
        return err(f"Upload failed: {e}")

    # ── Step 2: Extract text ──────────────────────────────────────────────
    r1 = extract(filepath=str(save_path), save_output=True)
    if r1["status"] != "success":
        return err(f"Extraction failed: {r1.get('error')}")
    txt_path = r1["output_path"]
    logger.info(f"Pipeline step 1 OK | words={r1['word_count']}")

    # ── Step 3: TTS ───────────────────────────────────────────────────────
    r2 = generate_tts(source_txt_path=txt_path, engine=tts_engine, voice_id=voice_id or None)
    if r2["status"] != "success":
        return err(f"TTS failed: {r2.get('error')}")
    wav_path = r2["output_path"]
    logger.info(f"Pipeline step 2 OK | duration={r2['duration_sec']}s")

    # ── Step 4: Modulate ──────────────────────────────────────────────────
    r3 = modulate_audio(input_wav_path=wav_path, cognitive_state=cognitive_state)
    if r3["status"] != "success":
        return err(f"Modulation failed: {r3.get('error')}")
    mod_path = r3["output_path"]
    logger.info(f"Pipeline step 3 OK | beat={r3['beat_freq_hz']}Hz")

    # ── Step 5: Visualize ─────────────────────────────────────────────────
    r4 = generate_visualization(
        modulated_wav_path=mod_path,
        role=role,
        cognitive_state=cognitive_state,
        beat_freq_hz=r3["beat_freq_hz"],
        document_title=document_title,
    )
    if r4["status"] != "success":
        return err(f"Visualization failed: {r4.get('error')}")
    logger.info(f"Pipeline step 4 OK | viz={r4['report_type']}")

    # ── Step 6: MCQ init ──────────────────────────────────────────────────
    r5 = initialise_document(
        source_txt_path=txt_path,
        document_title=document_title,
    )
    if r5["status"] != "success":
        return err(f"MCQ init failed: {r5.get('error')}")
    doc_id = r5["document_id"]
    logger.info(f"Pipeline step 5 OK | doc_id={doc_id}")

    logger.info(f"Full pipeline complete | doc_id={doc_id}")

    return ok({
        "doc_id":           doc_id,
        "document_title":   document_title,
        "extracted_path":   txt_path,
        "tts_wav_path":     wav_path,
        "modulated_path":   mod_path,
        "visualization_path": r4["output_path"],
        "visualization_type": r4["report_type"],
        "cognitive_state":  cognitive_state,
        "beat_freq_hz":     r3["beat_freq_hz"],
        "duration_sec":     r3["duration_sec"],
        "word_count":       r1["word_count"],
        "sessions_generated": r5["sessions_generated"],
        "sessions_meta":    r5["sessions_meta"],
        "message": (
            "Pipeline complete. Call /mcq/audio-completed when the "
            "user finishes listening to start the session flow."
        ),
    })


# ── /pipeline/audio — Phase 1: Extract + TTS + Modulate + Captions ────────────
# Returns mp3_b64, extracted_text, captions, beat_freq_hz, duration_sec
# Does NOT run MCQ. MCQ fires separately via /pipeline/mcq when user hits Play.

class PipelineAudioRequest(BaseModel):
    cognitive_state: str  = "deep_focus"
    document_title:  str  = "Tarang Document"
    tts_engine:      str  = "edge"
    voice_id:        str  = ""
    role:            str  = "user"

@app.post("/pipeline/audio", tags=["Pipeline"])
async def pipeline_audio(
    file: UploadFile = File(...),
    cognitive_state: str = Form("deep_focus"),
    document_title:  str = Form("Tarang Document"),
    tts_engine:      str = Form("edge"),
    voice_id:        str = Form(""),
    role:            str = Form("user"),
):
    """
    Phase 1 pipeline: Extract → TTS → Modulate → Captions.
    Returns mp3 as base64, extracted text, and captions.
    MCQ is NOT generated here — call /pipeline/mcq separately.
    """
    import base64, tempfile, os
    logger.info(
        f"Phase 1 pipeline | file={file.filename} | state={cognitive_state} "
        f"| engine={tts_engine} | role={role}"
    )

    # Save upload to temp file
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = tmp.name
        shutil.copyfileobj(file.file, tmp)

    try:
        # Step 1: Extract
        r1 = extract(filepath=tmp_path, save_output=False)
        if r1["status"] != "success":
            return err(f"Extraction failed: {r1.get('error')}")
        extracted_text = r1.get("text") or r1.get("extracted_text") or ""
        word_count     = r1.get("word_count", 0)
        logger.info(f"Phase1 step 1 OK | words={word_count}")

        # Step 2: TTS — write text to temp file for TTS
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w", encoding="utf-8") as tf:
            tf.write(extracted_text)
            txt_path = tf.name

        r2 = generate_tts(source_txt_path=txt_path, engine=tts_engine, voice_id=voice_id or None)
        if r2["status"] != "success":
            return err(f"TTS failed: {r2.get('error')}")
        wav_path = r2["output_path"]
        logger.info(f"Phase1 step 2 OK | duration={r2['duration_sec']}s")

        # Step 3: Modulate
        r3 = modulate_audio(input_wav_path=wav_path, cognitive_state=cognitive_state)
        if r3["status"] != "success":
            return err(f"Modulation failed: {r3.get('error')}")
        mod_path     = r3["output_path"]
        beat_freq_hz = r3["beat_freq_hz"]
        duration_sec = r3["duration_sec"]
        logger.info(f"Phase1 step 3 OK | beat={beat_freq_hz}Hz | duration={duration_sec}s")

        # Step 4: Read MP3 and encode as base64
        with open(mod_path, "rb") as mf:
            mp3_b64 = base64.b64encode(mf.read()).decode("utf-8")

        # Step 5: Captions (proportional timing — no AI needed)
        from file7_captions import generate_captions
        cap_result = generate_captions(text=extracted_text, duration_sec=duration_sec)
        captions   = cap_result.get("captions", []) if cap_result.get("status") == "success" else []
        logger.info(f"Phase1 step 4 OK | captions={len(captions)}")

        logger.info(f"Phase 1 complete | words={word_count} | duration={duration_sec}s")

        return ok({
            "mp3_b64":        mp3_b64,
            "extracted_text": extracted_text,
            "document_title": document_title,
            "word_count":     word_count,
            "duration_sec":   duration_sec,
            "beat_freq_hz":   beat_freq_hz,
            "cognitive_state":cognitive_state,
            "captions":       captions,
        })

    finally:
        # Clean up temp files
        for p in [tmp_path, txt_path if "txt_path" in dir() else None, wav_path if "wav_path" in dir() else None]:
            try:
                if p and os.path.exists(p): os.unlink(p)
            except: pass


# ── /pipeline/mcq — Phase 2: MCQ Generation ──────────────────────────────────
# Called in background when user clicks Play.
# Takes extracted_text from MongoDB, generates 3 sessions of MCQs.

class PipelineMCQRequest(BaseModel):
    extracted_text: str
    document_title: str = "Tarang Document"
    doc_id:         str = ""

@app.post("/pipeline/mcq", tags=["Pipeline"])
async def pipeline_mcq(req: PipelineMCQRequest):
    """
    Phase 2 pipeline: Generate MCQ questions for all 3 sessions.
    Called in background when user clicks Play on the audio player.
    """
    import tempfile, os
    logger.info(f"Phase 2 MCQ | doc_id={req.doc_id} | words={len(req.extracted_text.split())}")

    # Write extracted text to temp file for MCQ
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w", encoding="utf-8") as tf:
        tf.write(req.extracted_text)
        txt_path = tf.name

    try:
        r5 = initialise_document(
            source_txt_path=txt_path,
            document_title=req.document_title,
        )
        if r5["status"] != "success":
            return err(f"MCQ init failed: {r5.get('error')}")

        doc_id = r5["document_id"]
        logger.info(f"Phase 2 MCQ complete | doc_id={doc_id} | sessions={r5['sessions_generated']}")

        # Read generated question/answer files
        import json
        from pathlib import Path

        mcq_dir = Path("storage/mcq") if Path("storage/mcq").exists() else Path(tempfile.gettempdir())
        result  = {
            "doc_id":            doc_id,
            "sessions_generated":r5["sessions_generated"],
            "session_state":     r5.get("sessions_meta", {}),
        }

        for n in [1, 2, 3]:
            q_path = Path(f"storage/mcq/{doc_id}_session{n}_questions.json")
            a_path = Path(f"storage/mcq/{doc_id}_session{n}_answers.json")
            result[f"session_{n}_questions"] = json.loads(q_path.read_text()) if q_path.exists() else {"questions": []}
            result[f"session_{n}_answers"]   = json.loads(a_path.read_text()) if a_path.exists() else {"answers": {}}

        return ok(result)

    finally:
        try:
            if os.path.exists(txt_path): os.unlink(txt_path)
        except: pass



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
    print("\nTarang 1.0.0.1 — Python Bridge")
    print("================================")
    print("Starting FastAPI on http://localhost:5001")
    print("Swagger UI:  http://localhost:5001/docs")
    print("ReDoc:       http://localhost:5001/redoc")
    print()
    uvicorn.run(
        "bridge:app",
        host="0.0.0.0",
        port=5001,
        reload=True,
        log_level="info",
    )

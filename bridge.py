"""
Tarang 2.0.1 — bridge.py — FastAPI Microservice (Production Edition)
=====================================================================
Runs on: http://localhost:9801 (or whatever port assigned by load_balancer)
Load balancer entry: http://localhost:5000

v2.0.1 Bug Fix vs v2.0.0:
  CRITICAL FIX — /extract endpoint now returns BOTH text fields:
    • data.text     = optimize_for_presentation(clean_text)  — TTS-ready prose
    • data.raw_text = sanitize_text(clean_text)              — continuous string
                                                               used by documentController
                                                               for correct word-splitting

  ROOT CAUSE of 19-parts bug:
    /extract was only returning data.text (paragraph-split by optimize_for_presentation).
    documentController.js used raw_text || text fallback, but raw_text was undefined
    so it fell back to data.text → splitTextIntoParts() saw ~36 \n\n-separated chunks
    and made 19 "parts" instead of 2.

    Fix: expose raw_text from file1_extractor's return dict in the /extract response.
    documentController.js already reads extractRes.data.raw_text correctly — no changes
    needed there.

  Also fixed /pipeline/audio:
    Previously the pipeline endpoint re-used result["text"] for both the extracted_text
    payload AND any internal splitting. Now it correctly separates:
      • extracted_text  → result["text"]     (presentation-optimized, for audio/captions)
      • raw_text        → result["raw_text"] (continuous, exposed for future splitting use)

All other v2.0.0 features unchanged.

Environment variables (all optional):
  TARANG_CPU_WORKERS  = 4     # worker processes per bridge instance
  TARANG_MAX_QUEUE    = 50    # max concurrent requests before 503
"""

import os
import sys
import json
import uuid
import shutil
import logging
import asyncio
import tempfile
import base64
from pathlib import Path
from typing import Optional

import httpx
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from pydantic import BaseModel

# ── Add project root to path ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Worker pool (multiprocessing) ─────────────────────────────────────────────
from worker_pool import run_in_process, get_pool_stats, shutdown_pool

# ── Core engine imports (used for non-CPU-heavy calls only) ───────────────────
print("==> [bridge v2.1.0] Importing core engine modules...")
from file1_extractor import extract, sanitize_text, optimize_for_presentation
from file2_tts       import generate_tts, get_voices
from file3_modulator import modulate_audio, get_quiz_questions, suggest_cognitive_state
from file4_visualizer import generate_visualization
from file5_mcq       import (
    initialise_document, audio_completed, get_session_status,
    get_questions, override_window, submit_test, get_final_results,
)
from file6_analytics import generate_analytics
from file7_captions  import generate_captions
print("==> [bridge v2.1.0] All modules imported OK")


# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [bridge v2.1.0] %(levelname)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("bridge")

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Tarang Core Engine Bridge",
    description="FastAPI microservice — Tarang 2.1.0 (Production, multiprocessing, SSE streaming).",
    version="2.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request ID middleware ─────────────────────────────────────────────────────

@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Attach a unique X-Request-ID to every response for distributed tracing."""
    req_id = request.headers.get("X-Request-ID", str(uuid.uuid4())[:8])
    response = await call_next(request)
    response.headers["X-Request-ID"] = req_id
    return response


# ── Response helpers ──────────────────────────────────────────────────────────

def ok(data: dict) -> JSONResponse:
    return JSONResponse(content={"status": "success", "data": data})

def err(message: str, code: int = 400) -> JSONResponse:
    logger.error(f"Error | {code} | {message}")
    return JSONResponse(status_code=code, content={"status": "error", "error": message})

def queue_full_error() -> JSONResponse:
    """Returned when all worker slots are occupied."""
    stats = get_pool_stats()
    return JSONResponse(
        status_code=503,
        headers={"Retry-After": "10"},
        content={
            "status": "error",
            "error": "Server is at capacity. Please retry in a few seconds.",
            "queue_depth": stats["queue_depth"],
            "max_queue":   stats["max_queue"],
        },
    )


# ── Pydantic request models ───────────────────────────────────────────────────

class TTSRequest(BaseModel):
    extracted_txt_path:  str
    engine:              Optional[str]  = "edge"
    output_filename:     Optional[str]  = None
    voice_id:            Optional[str]  = None
    file1_metadata:      Optional[dict] = None

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
    carrier_freq_hz:    Optional[float] = 100.0
    document_title:     Optional[str]   = "Tarang Session"
    session_number:     Optional[int]   = 1
    output_stem:        Optional[str]   = None

class MCQInitRequest(BaseModel):
    extracted_txt_path:     str
    document_title:         Optional[str]   = "Tarang Document"
    num_questions:          Optional[int]   = 10
    custom_s1_to_s2_hours:  Optional[float] = 12.0
    custom_s2_to_s3_hours:  Optional[float] = 24.0

class SuggestModeRequest(BaseModel):
    answers: dict

class AudioCompletedRequest(BaseModel):
    doc_id:        str
    role:          Optional[str]  = "user"
    session_state: Optional[dict] = None

class GetQuestionsRequest(BaseModel):
    doc_id:         str
    session:        int
    role:           Optional[str]  = "user"
    session_state:  Optional[dict] = None
    questions_data: Optional[dict] = None

class OverrideWindowRequest(BaseModel):
    doc_id:        str
    session_state: Optional[dict] = None

class SubmitTestRequest(BaseModel):
    doc_id:          str
    session:         int
    user_answers:    dict
    role:            Optional[str]  = "user"
    session_state:   Optional[dict] = None
    answer_key_data: Optional[dict] = None

class MCQStatusRequest(BaseModel):
    doc_id:        str
    role:          Optional[str]  = "user"
    session_state: Optional[dict] = None

class MCQResultsRequest(BaseModel):
    doc_id:          str
    role:            Optional[str]  = "user"
    session_state:   Optional[dict] = None
    all_answer_keys: Optional[dict] = None

class AnalyticsRequest(BaseModel):
    doc_id:           str
    role:             Optional[str]  = "user"
    session_state:    Optional[dict] = None
    all_questions:    Optional[dict] = None
    all_answer_keys:  Optional[dict] = None
    cognitive_states: Optional[dict] = None

class CaptionsRequest(BaseModel):
    text:         str
    duration_sec: float

class PipelineMCQRequest(BaseModel):
    extracted_text: str
    document_title: str = "Tarang Document"
    doc_id:         str = ""

class PipelineAudioTextRequest(BaseModel):
    text: str
    document_title: str = "Tarang Document"
    cognitive_state: str = "deep_focus"
    tts_engine: str = "edge"
    voice_id: Optional[str] = None
    role: str = "user"
    file1_metadata: Optional[dict] = None


# ── System endpoints ──────────────────────────────────────────────────────────

@app.get("/", tags=["System"])
async def root():
    return {
        "service": "Tarang Core Engine Bridge",
        "version": "2.0.1",
        "docs": "/docs",
        "status": "running",
        "mode": "multiprocessing",
    }


@app.get("/health", tags=["System"])
async def health():
    return {"status": "ok", "service": "Tarang Python Bridge", "version": "2.1.0"}


@app.get("/status", tags=["System"])
async def get_status():
    """
    Real-time worker pool status.
    The load balancer polls this to decide where to route requests.
    """
    stats = get_pool_stats()
    busy  = stats["queue_depth"] >= stats["max_queue"]
    return {
        "status": "ok",
        "data": {
            "busy":          busy,
            "queue_depth":   stats["queue_depth"],
            "slots_free":    stats["slots_free"],
            "cpu_workers":   stats["cpu_workers"],
            "max_queue":     stats["max_queue"],
            "message":       "Bridge is at capacity" if busy else "Bridge is ready",
        },
    }


@app.get("/voices", tags=["Core Engine"])
async def list_voices():
    result = get_voices()
    if result["status"] != "success":
        return err(result.get("error", "Failed to list voices"))
    return ok({"voices": result["voices"], "count": result["count"]})


# ── Quiz endpoints ────────────────────────────────────────────────────────────

@app.get("/quiz", tags=["Mode Suggestion"])
async def quiz_questions():
    result = get_quiz_questions()
    return ok(result)


@app.post("/suggest-mode", tags=["Mode Suggestion"])
async def suggest_mode(req: SuggestModeRequest):
    if not req.answers:
        return err("answers dict is required.", 400)
    result = suggest_cognitive_state(req.answers)
    if result["status"] != "success":
        return err(result.get("error", "Mode suggestion failed"))
    return ok(result)


# ── FILE 1 — Document extraction ──────────────────────────────────────────────
#
# v2.0.1 FIX: Now returns BOTH text fields from file1_extractor:
#   • data.text     = optimize_for_presentation() output (paragraph-split prose)
#   • data.raw_text = sanitize_text() output (continuous string)
#
# documentController.js reads extractRes.data.raw_text for splitting.
# Without this field the controller fell back to data.text which was
# already split into ~200-word paragraphs by optimize_for_presentation(),
# causing splitTextIntoParts() to generate 19 parts instead of 2.
#
@app.post("/extract", tags=["Core Engine"])
async def extract_document(file: UploadFile = File(...)):
    suffix   = Path(file.filename).suffix.lower()
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            shutil.copyfileobj(file.file, tmp)

        try:
            result = await run_in_process(extract, filepath=tmp_path, save_output=False)
        except RuntimeError as e:
            if "queue_full" in str(e):
                return queue_full_error()
            raise

        if result["status"] != "success":
            return err(result.get("error", "Extraction failed"))

        raw_text_words = len((result.get("raw_text") or "").split())
        opt_text_words = result.get("word_count", 0)
        logger.info(
            f"/extract COMPLETE | optimized_words={opt_text_words} | "
            f"raw_text_words={raw_text_words} | format={result['format']}"
        )

        # ── CRITICAL: return raw_text so documentController can split correctly ──
        return ok({
            "output_path": None,
            "text":        result["text"],       # optimize_for_presentation() — for TTS
            "raw_text":    result["raw_text"],   # sanitize_text()             — for splitting
            "word_count":  result["word_count"],
            "char_count":  result["char_count"],
            "format":      result["format"],
            "metadata":    result["metadata"],
            "timestamp":   result["timestamp"],
        })
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


# ── FILE 2 — TTS ──────────────────────────────────────────────────────────────

@app.post("/tts", tags=["Core Engine"])
async def generate_tts_audio(req: TTSRequest):
    if not Path(req.extracted_txt_path).exists():
        return err(f"Text file not found: {req.extracted_txt_path}")
    try:
        result = await run_in_process(
            generate_tts,
            source_txt_path=req.extracted_txt_path,
            engine=req.engine,
            output_filename=req.output_filename,
            voice_id=req.voice_id,
            file1_metadata=req.file1_metadata,
        )
    except RuntimeError as e:
        if "queue_full" in str(e):
            return queue_full_error()
        raise
    if result["status"] != "success":
        return err(result.get("error", "TTS failed"))
    return ok({
        "output_path":   result["output_path"],
        "engine_used":   result["engine_used"],
        "duration_sec":  result["duration_sec"],
        "word_count":    result["word_count"],
        "chunks_total":  result["chunks_total"],
        "pace_factor":   result.get("pace_factor", 1.0),
        "timestamp":     result["timestamp"],
    })


# ── FILE 3 — Modulation ───────────────────────────────────────────────────────

@app.post("/modulate", tags=["Core Engine"])
async def modulate(req: ModulateRequest):
    if not Path(req.tts_wav_path).exists():
        return err(f"WAV file not found: {req.tts_wav_path}")
    try:
        result = await run_in_process(
            modulate_audio,
            input_wav_path=req.tts_wav_path,
            cognitive_state=req.cognitive_state,
            output_filename=req.output_filename,
            custom_beat_freq=req.custom_beat_freq,
            custom_depth=req.custom_depth,
        )
    except RuntimeError as e:
        if "queue_full" in str(e):
            return queue_full_error()
        raise
    if result["status"] != "success":
        return err(result.get("error", "Modulation failed"))
    return ok({
        "output_path":     result["output_path"],
        "cognitive_state": result["cognitive_state"],
        "beat_freq_hz":    result["beat_freq_hz"],
        "carrier_freq_hz": result["carrier_freq_hz"],
        "depth":           result["depth"],
        "duration_sec":    result["duration_sec"],
        "sample_rate":     result["sample_rate"],
        "brain_band":      result["brain_band"],
        "description":     result["description"],
        "timestamp":       result["timestamp"],
    })


# ── FILE 4 — Visualization ────────────────────────────────────────────────────

@app.post("/visualize", tags=["Core Engine"])
async def visualize(req: VisualizeRequest):
    if not Path(req.modulated_wav_path).exists():
        return err(f"Audio file not found: {req.modulated_wav_path}")
    try:
        result = await run_in_process(
            generate_visualization,
            modulated_wav_path=req.modulated_wav_path,
            role=req.role,
            cognitive_state=req.cognitive_state,
            beat_freq_hz=req.beat_freq_hz,
            carrier_freq_hz=req.carrier_freq_hz,
            document_title=req.document_title,
            session_number=req.session_number,
            output_stem=req.output_stem,
        )
    except RuntimeError as e:
        if "queue_full" in str(e):
            return queue_full_error()
        raise
    if result["status"] != "success":
        return err(result.get("error", "Visualization failed"))
    return ok({
        "output_path":  result["output_path"],
        "html_content": result.get("html_content", ""),
        "report_type":  result["report_type"],
        "role":         result["role"],
        "timestamp":    result["timestamp"],
    })


# ── FILE 5 — MCQ ──────────────────────────────────────────────────────────────

@app.post("/mcq/init", tags=["MCQ"])
async def mcq_init(req: MCQInitRequest):
    if not Path(req.extracted_txt_path).exists():
        return err(f"Text file not found: {req.extracted_txt_path}")
    try:
        result = await run_in_process(
            initialise_document,
            source_txt_path=req.extracted_txt_path,
            document_title=req.document_title,
            num_questions=req.num_questions,
            custom_s1_to_s2_hours=req.custom_s1_to_s2_hours,
            custom_s2_to_s3_hours=req.custom_s2_to_s3_hours,
        )
    except RuntimeError as e:
        if "queue_full" in str(e):
            return queue_full_error()
        raise
    if result["status"] != "success":
        return err(result.get("error", "MCQ init failed"))
    return ok(result)


@app.post("/mcq/audio-completed", tags=["MCQ"])
async def mcq_audio_completed(req: AudioCompletedRequest):
    if not req.session_state:
        return err("session_state is required.", 400)
    result = audio_completed(doc_id=req.doc_id, role=req.role, session_state=req.session_state)
    if result["status"] != "success":
        return err(result.get("error", "Failed to mark audio complete"))
    return ok(result)

@app.post("/mcq/status", tags=["MCQ"])
async def mcq_status(req: MCQStatusRequest):
    if not req.session_state:
        return err("session_state is required.", 400)
    result = get_session_status(doc_id=req.doc_id, role=req.role, session_state=req.session_state)
    if result["status"] != "success":
        return err(result.get("error", "Failed to get status"), 404)
    return ok(result)

@app.post("/mcq/questions", tags=["MCQ"])
async def mcq_questions(req: GetQuestionsRequest):
    if not req.session_state:
        return err("session_state is required.", 400)
    if not req.questions_data:
        return err("questions_data is required.", 400)
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
    if not req.session_state:
        return err("session_state is required.", 400)
    result = override_window(doc_id=req.doc_id, session_state=req.session_state)
    if result["status"] != "success":
        return err(result.get("error", "Override failed"))
    return ok(result)

@app.post("/mcq/submit", tags=["MCQ"])
async def mcq_submit(req: SubmitTestRequest):
    if not req.session_state:
        return err("session_state is required.", 400)
    if not req.answer_key_data:
        return err("answer_key_data is required.", 400)
    result = submit_test(
        doc_id=req.doc_id, session=req.session, user_answers=req.user_answers,
        role=req.role, session_state=req.session_state, answer_key_data=req.answer_key_data,
    )
    if result["status"] != "success":
        return err(result.get("error", "Submission failed"))
    return ok(result)

@app.post("/mcq/results", tags=["MCQ"])
async def mcq_results(req: MCQResultsRequest):
    if not req.session_state:
        return err("session_state is required.", 400)
    result = get_final_results(
        doc_id=req.doc_id, role=req.role,
        session_state=req.session_state, all_answer_keys=req.all_answer_keys or {},
    )
    if result["status"] != "success":
        return err(result.get("error", "Results not available"), 403)
    return ok(result)


# ── FILE 6 — Analytics ────────────────────────────────────────────────────────

@app.post("/analytics", tags=["Analytics"])
async def analytics(req: AnalyticsRequest):
    if not req.session_state:
        return err("session_state is required.", 400)

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

    try:
        result = await run_in_process(
            generate_analytics,
            doc_id=req.doc_id,
            role=req.role,
            session_state=req.session_state,
            all_questions=all_q,
            all_answer_keys=all_ak,
            cognitive_states=req.cognitive_states,
        )
    except RuntimeError as e:
        if "queue_full" in str(e):
            return queue_full_error()
        raise

    if result["status"] != "success":
        return err(result.get("error", "Analytics failed"))
    return ok({
        "analytics":    result["analytics"],
        "html_content": result["html_content"],
    })


# ── FILE 7 — Captions ─────────────────────────────────────────────────────────

@app.post("/captions", tags=["Captions"])
async def captions(req: CaptionsRequest):
    if not req.text or not req.text.strip():
        return err("No text provided.")
    if req.duration_sec <= 0:
        return err("Invalid audio duration.")
    try:
        result = await run_in_process(
            generate_captions,
            text=req.text,
            duration_sec=req.duration_sec,
        )
    except RuntimeError as e:
        if "queue_full" in str(e):
            return queue_full_error()
        raise
    if result["status"] != "success":
        return err(result.get("error", "Caption generation failed"))
    return ok({
        "captions":        result["captions"],
        "total_segments":  result["total_segments"],
        "duration_sec":    result["duration_sec"],
        "method":          result["method"],
    })


# ── Static file serving ───────────────────────────────────────────────────────

@app.get("/files/{folder}/{filename}", tags=["Files"])
async def serve_file(folder: str, filename: str):
    allowed = {"audio_cache", "reports", "analytics", "extracted", "mcq"}
    if folder not in allowed:
        raise HTTPException(status_code=403, detail=f"Folder '{folder}' not accessible.")
    file_path = PROJECT_ROOT / "storage" / folder / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")
    return FileResponse(path=str(file_path), filename=filename, media_type="application/octet-stream")


# ── Pipeline: /pipeline/audio ─────────────────────────────────────────────────
#
# This endpoint receives ONE part (already split by documentController.js).
# It runs the full audio pipeline on that single part:
#   Extract (already done by controller) → TTS → Modulate → Captions
#
# The file sent here is the raw part text as a .txt file.
# The pipeline runs its own TTS optimization internally — audio quality unchanged.
#
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
    Phase 1: TTS → Modulate → Captions for a single document part.

    The documentController.js:
      1. Calls /extract once to get raw_text
      2. Splits raw_text into parts using splitTextIntoParts()
      3. Calls /pipeline/audio once per part with the part text as a .txt file

    This endpoint receives each part as a plain text file and runs the
    full audio pipeline on it. It does NOT re-split the text.
    """
    logger.info(f"POST /pipeline/audio | file={file.filename} | state={cognitive_state} | title={document_title}")

    # Track ALL temp paths for cleanup — including intermediate WAV files (FIX D)
    tmp_path = txt_path = wav_path = mod_path = None
    suffix   = os.path.splitext(file.filename)[1].lower()

    try:
        # Save uploaded part to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            shutil.copyfileobj(file.file, tmp)

        # ── Step 1: Extract / read the part text ──────────────────────────────
        # If the controller sends a .txt file (raw part text), extract() handles it.
        # If it sends a PDF/DOCX (original file for single-part docs), it extracts normally.
        try:
            r1 = await run_in_process(extract, filepath=tmp_path, save_output=False)
        except RuntimeError as e:
            if "queue_full" in str(e):
                return queue_full_error()
            raise

        if r1["status"] != "success":
            return err(f"Extraction failed: {r1.get('error')}")

        # For audio pipeline, we use the TTS-optimized text (not raw_text).
        # raw_text is only needed by the controller for splitting — splitting
        # has already happened before this endpoint is called.
        extracted_text = r1.get("text") or ""
        word_count     = r1.get("word_count", 0)
        file1_metadata = r1.get("metadata", {})

        if not extracted_text.strip():
            return err("Document extraction produced empty text.")

        logger.info(f"  Part extracted | words={word_count}")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w", encoding="utf-8") as tf:
            tf.write(extracted_text)
            txt_path = tf.name

        # ── Step 2: TTS ───────────────────────────────────────────────────────
        try:
            r2 = await run_in_process(
                generate_tts,
                source_txt_path=txt_path,
                engine=tts_engine,
                voice_id=voice_id or None,
                file1_metadata=file1_metadata,
            )
        except RuntimeError as e:
            if "queue_full" in str(e):
                return queue_full_error()
            raise

        if r2["status"] != "success":
            return err(f"TTS failed: {r2.get('error')}")

        wav_path    = r2["output_path"]   # tracked for cleanup (FIX D)
        pace_factor = r2.get("pace_factor", 1.0)

        # ── Step 3: Modulate ──────────────────────────────────────────────────
        try:
            r3 = await run_in_process(
                modulate_audio,
                input_wav_path=wav_path,
                cognitive_state=cognitive_state,
            )
        except RuntimeError as e:
            if "queue_full" in str(e):
                return queue_full_error()
            raise

        if r3["status"] != "success":
            return err(f"Modulation failed: {r3.get('error')}")

        mod_path     = r3["output_path"]   # tracked for cleanup (FIX D)
        beat_freq_hz = r3["beat_freq_hz"]
        carrier_freq = r3["carrier_freq_hz"]
        duration_sec = r3["duration_sec"]
        brain_band   = r3["brain_band"]

        # ── Step 4: Encode MP3 ────────────────────────────────────────────────
        if not os.path.exists(mod_path):
            return err(f"Modulated audio not found: {mod_path}")
        with open(mod_path, "rb") as mf:
            mp3_b64 = base64.b64encode(mf.read()).decode("utf-8")

        # ── Step 5: Captions ──────────────────────────────────────────────────
        try:
            cap_result = await run_in_process(
                generate_captions,
                text=extracted_text,
                duration_sec=duration_sec,
            )
            captions = cap_result.get("captions", []) if cap_result.get("status") == "success" else []
        except RuntimeError:
            captions = []

        logger.info(
            f"  /pipeline/audio COMPLETE | words={word_count} | "
            f"duration={duration_sec}s | captions={len(captions)}"
        )

        return ok({
            "mp3_b64":         mp3_b64,
            "extracted_text":  extracted_text,
            "document_title":  document_title,
            "word_count":      word_count,
            "duration_sec":    duration_sec,
            "beat_freq_hz":    beat_freq_hz,
            "carrier_freq_hz": carrier_freq,
            "cognitive_state": cognitive_state,
            "brain_band":      brain_band,
            "pace_factor":     pace_factor,
            "captions":        captions,
            "file1_metadata":  file1_metadata,
        })

    except Exception as e:
        logger.error(f"  /pipeline/audio EXCEPTION: {type(e).__name__}: {e}", exc_info=True)
        return err(f"Pipeline audio failed: {str(e)}", 500)

    finally:
        # FIX D: clean up ALL intermediate files including TTS WAV + modulated WAV
        for p in [tmp_path, txt_path, wav_path, mod_path]:
            try:
                if p and os.path.exists(p):
                    os.unlink(p)
            except Exception:
                pass


@app.post("/pipeline/audio-text", tags=["Pipeline"])
async def pipeline_audio_text(req: PipelineAudioTextRequest):
    """
    Faster path for already-extracted text.
    Skips file1 extraction completely and only runs:
      text cleanup/optimization -> TTS -> modulation -> captions
    """
    logger.info(
        "POST /pipeline/audio-text | title=%s | words=%d | state=%s",
        req.document_title,
        len((req.text or "").split()),
        req.cognitive_state,
    )

    txt_path = wav_path = mod_path = None  # FIX D: track all temp files
    try:
        raw_text = sanitize_text(req.text or "")
        if not raw_text.strip():
            return err("Input text is empty.")

        extracted_text = optimize_for_presentation(raw_text)
        word_count = len(extracted_text.split())
        file1_metadata = req.file1_metadata or {}

        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w", encoding="utf-8") as tf:
            tf.write(extracted_text)
            txt_path = tf.name

        try:
            r2 = await run_in_process(
                generate_tts,
                source_txt_path=txt_path,
                engine=req.tts_engine,
                voice_id=req.voice_id or None,
                file1_metadata=file1_metadata,
            )
        except RuntimeError as e:
            if "queue_full" in str(e):
                return queue_full_error()
            raise

        if r2["status"] != "success":
            return err(f"TTS failed: {r2.get('error')}")

        wav_path = r2["output_path"]  # tracked for cleanup (FIX D)
        pace_factor = r2.get("pace_factor", 1.0)

        try:
            r3 = await run_in_process(
                modulate_audio,
                input_wav_path=wav_path,
                cognitive_state=req.cognitive_state,
            )
        except RuntimeError as e:
            if "queue_full" in str(e):
                return queue_full_error()
            raise

        if r3["status"] != "success":
            return err(f"Modulation failed: {r3.get('error')}")

        mod_path = r3["output_path"]  # tracked for cleanup (FIX D)
        beat_freq_hz = r3["beat_freq_hz"]
        carrier_freq = r3["carrier_freq_hz"]
        duration_sec = r3["duration_sec"]
        brain_band = r3["brain_band"]

        if not os.path.exists(mod_path):
            return err(f"Modulated audio not found: {mod_path}")
        with open(mod_path, "rb") as mf:
            mp3_b64 = base64.b64encode(mf.read()).decode("utf-8")

        try:
            cap_result = await run_in_process(
                generate_captions,
                text=extracted_text,
                duration_sec=duration_sec,
            )
            captions = cap_result.get("captions", []) if cap_result.get("status") == "success" else []
        except RuntimeError:
            captions = []

        return ok({
            "mp3_b64": mp3_b64,
            "extracted_text": extracted_text,
            "document_title": req.document_title,
            "word_count": word_count,
            "duration_sec": duration_sec,
            "beat_freq_hz": beat_freq_hz,
            "carrier_freq_hz": carrier_freq,
            "cognitive_state": req.cognitive_state,
            "brain_band": brain_band,
            "pace_factor": pace_factor,
            "captions": captions,
            "file1_metadata": file1_metadata,
        })

    except Exception as e:
        logger.error(f"  /pipeline/audio-text EXCEPTION: {type(e).__name__}: {e}", exc_info=True)
        return err(f"Pipeline audio text failed: {str(e)}", 500)
    finally:
        # FIX D: clean up ALL intermediate files
        for p in [txt_path, wav_path, mod_path]:
            try:
                if p and os.path.exists(p):
                    os.unlink(p)
            except Exception:
                pass




# ── Pipeline: /pipeline/audio/stream (SSE) ──────────────────────────────────────────────
#
# FIX E: Server-Sent Events streaming variant of /pipeline/audio.
# Yields one progress event after each pipeline step so the frontend shows a
# live progress bar instead of a frozen spinner for 4-5 minutes.
#
# Event stream (text/event-stream, each line is a JSON dict):
#
#   data: {"step":"extract_start", "pct":5,  "message":"..."}
#   data: {"step":"extract",       "pct":12, "words":1234}
#   data: {"step":"tts_start",     "pct":15, "message":"..."}
#   data: {"step":"tts",           "pct":58, "duration_sec":1680.2, "chunks":312}
#   data: {"step":"modulate_start","pct":60, "message":"..."}
#   data: {"step":"modulate",      "pct":72, "beat_freq_hz":14.0}
#   data: {"step":"encode",        "pct":82, "message":"..."}
#   data: {"step":"captions",      "pct":96, "segments":420}
#   data: {"step":"done",          "pct":100, "mp3_b64":"...", "captions":[...], ...}
#   data: {"step":"error",         "pct":-1,  "error":"..."}
#
# The "done" payload is identical to the /pipeline/audio JSON response so the
# frontend can share the same completion handler.
#
@app.post("/pipeline/audio/stream", tags=["Pipeline"])
async def pipeline_audio_stream(
    file:            UploadFile = File(...),
    cognitive_state: str        = Form("deep_focus"),
    document_title:  str        = Form("Tarang Document"),
    tts_engine:      str        = Form("edge"),
    voice_id:        str        = Form(""),
    role:            str        = Form("user"),
):
    """
    SSE streaming variant of /pipeline/audio.
    Yields progress events after each pipeline step — no more frozen spinner.

    JS usage:
        const fd = new FormData();
        fd.append("file", myFile);
        fd.append("cognitive_state", "deep_focus");
        // EventSource only supports GET; use fetch + ReadableStream instead:
        const resp = await fetch("/pipeline/audio/stream", {method:"POST", body:fd});
        const reader = resp.body.getReader();
        const decoder = new TextDecoder();
        let buf = "";
        while (true) {
            const {done, value} = await reader.read();
            if (done) break;
            buf += decoder.decode(value, {stream: true});
            const lines = buf.split("\n\n");
            buf = lines.pop();
            for (const line of lines) {
                if (!line.startsWith("data: ")) continue;
                const ev = JSON.parse(line.slice(6));
                updateProgress(ev.pct, ev.step);
                if (ev.step === "done")  handleDone(ev);
                if (ev.step === "error") handleError(ev.error);
            }
        }
    """
    logger.info(
        "POST /pipeline/audio/stream | file=%s | state=%s | title=%s",
        file.filename, cognitive_state, document_title,
    )

    # Buffer the upload immediately so the UploadFile handle is not held across
    # async generator suspension points (yields).
    file_bytes  = await file.read()
    file_suffix = os.path.splitext(file.filename)[1].lower()

    async def _event_stream():
        tmp_path = txt_path = wav_path = mod_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as tmp:
                tmp_path = tmp.name
                tmp.write(file_bytes)

            def _emit(step, pct, **kw):
                return "data: " + json.dumps({"step": step, "pct": pct, **kw}) + "\n\n"

            # ── Step 1: Extract ──────────────────────────────────────────────
            yield _emit("extract_start", 5, message="Extracting document text...")
            try:
                r1 = await run_in_process(extract, filepath=tmp_path, save_output=False)
            except RuntimeError as e:
                if "queue_full" in str(e):
                    yield _emit("error", -1, error="Server at capacity — please retry.")
                    return
                raise
            if r1["status"] != "success":
                yield _emit("error", -1, error="Extraction failed: " + str(r1.get("error")))
                return

            extracted_text = r1.get("text") or ""
            word_count     = r1.get("word_count", 0)
            file1_metadata = r1.get("metadata", {})
            if not extracted_text.strip():
                yield _emit("error", -1, error="Document extraction produced empty text.")
                return
            yield _emit("extract", 12, words=word_count, message="Text extracted.")

            with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w", encoding="utf-8") as tf:
                tf.write(extracted_text)
                txt_path = tf.name

            # ── Step 2: TTS (longest step) ───────────────────────────────────
            yield _emit("tts_start", 15, message="Generating speech audio (longest step)...")
            try:
                r2 = await run_in_process(
                    generate_tts,
                    source_txt_path=txt_path,
                    engine=tts_engine,
                    voice_id=voice_id or None,
                    file1_metadata=file1_metadata,
                )
            except RuntimeError as e:
                if "queue_full" in str(e):
                    yield _emit("error", -1, error="Server at capacity — please retry.")
                    return
                raise
            if r2["status"] != "success":
                yield _emit("error", -1, error="TTS failed: " + str(r2.get("error")))
                return

            wav_path    = r2["output_path"]
            pace_factor = r2.get("pace_factor", 1.0)
            tts_dur     = r2.get("duration_sec", 0)
            chunks_n    = r2.get("chunks_total", 0)
            yield _emit("tts", 58,
                        duration_sec=tts_dur, chunks=chunks_n, pace_factor=pace_factor,
                        message="Speech generated (" + str(int(tts_dur)) + "s, " + str(chunks_n) + " chunks).")

            # ── Step 3: Modulate ─────────────────────────────────────────────
            yield _emit("modulate_start", 60, message="Applying binaural modulation...")
            try:
                r3 = await run_in_process(
                    modulate_audio,
                    input_wav_path=wav_path,
                    cognitive_state=cognitive_state,
                )
            except RuntimeError as e:
                if "queue_full" in str(e):
                    yield _emit("error", -1, error="Server at capacity — please retry.")
                    return
                raise
            if r3["status"] != "success":
                yield _emit("error", -1, error="Modulation failed: " + str(r3.get("error")))
                return

            mod_path     = r3["output_path"]
            beat_freq_hz = r3["beat_freq_hz"]
            carrier_freq = r3["carrier_freq_hz"]
            duration_sec = r3["duration_sec"]
            brain_band   = r3["brain_band"]
            yield _emit("modulate", 72,
                        beat_freq_hz=beat_freq_hz, carrier_freq_hz=carrier_freq,
                        brain_band=brain_band,
                        message="Modulation done (" + str(brain_band) + ", " + str(beat_freq_hz) + "Hz).")

            # ── Step 4: Encode ───────────────────────────────────────────────
            yield _emit("encode_start", 74, message="Encoding audio...")
            if not os.path.exists(mod_path):
                yield _emit("error", -1, error="Modulated audio file not found.")
                return
            with open(mod_path, "rb") as mf:
                mp3_b64 = base64.b64encode(mf.read()).decode("utf-8")
            yield _emit("encode", 82, message="Audio encoded.")

            # ── Step 5: Captions ─────────────────────────────────────────────
            yield _emit("captions_start", 84, message="Generating captions...")
            try:
                cap_result = await run_in_process(
                    generate_captions, text=extracted_text, duration_sec=duration_sec,
                )
                captions = (cap_result.get("captions", [])
                            if cap_result.get("status") == "success" else [])
            except RuntimeError:
                captions = []
            yield _emit("captions", 96,
                        segments=len(captions),
                        message=str(len(captions)) + " caption segments generated.")

            # ── Done ─────────────────────────────────────────────────────────
            logger.info(
                "  /pipeline/audio/stream COMPLETE | words=%d | duration=%.1fs | captions=%d",
                word_count, duration_sec, len(captions),
            )
            yield _emit("done", 100,
                        mp3_b64=mp3_b64,
                        extracted_text=extracted_text,
                        document_title=document_title,
                        word_count=word_count,
                        duration_sec=duration_sec,
                        beat_freq_hz=beat_freq_hz,
                        carrier_freq_hz=carrier_freq,
                        cognitive_state=cognitive_state,
                        brain_band=brain_band,
                        pace_factor=pace_factor,
                        captions=captions,
                        file1_metadata=file1_metadata,
                        message="Pipeline complete.")

        except Exception as exc:
            logger.error("  /pipeline/audio/stream EXCEPTION: %s: %s",
                         type(exc).__name__, exc, exc_info=True)
            try:
                yield "data: " + json.dumps({"step": "error", "pct": -1,
                                              "error": "Pipeline failed: " + str(exc)}) + "\n\n"
            except Exception:
                pass
        finally:
            for p in [tmp_path, txt_path, wav_path, mod_path]:
                try:
                    if p and os.path.exists(p):
                        os.unlink(p)
                except Exception:
                    pass

    return StreamingResponse(
        _event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",   # tell nginx: do NOT buffer — stream immediately
            "Connection":        "keep-alive",
        },
    )


# ── Pipeline: /pipeline/mcq ───────────────────────────────────────────────────

@app.post("/pipeline/mcq", tags=["Pipeline"])
async def pipeline_mcq(req: PipelineMCQRequest):
    """Phase 2: Generate MCQ using local LLM — runs in process pool."""
    logger.info(f"POST /pipeline/mcq | words={len(req.extracted_text.split())}")

    txt_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w", encoding="utf-8") as tf:
            tf.write(req.extracted_text)
            txt_path = tf.name

        try:
            r5 = await run_in_process(
                initialise_document,
                source_txt_path=txt_path,
                document_title=req.document_title,
            )
        except RuntimeError as e:
            if "queue_full" in str(e):
                return queue_full_error()
            raise

        if r5["status"] != "success":
            return err(f"MCQ init failed: {r5.get('error')}")

        return ok({
            "doc_id":              r5["document_id"],
            "sessions_generated":  r5["sessions_generated"],
            "llm_engine":          r5.get("llm_engine", "rule-based"),
            "session_state":       r5["session_state"],
            "sessions_meta":       r5["sessions_meta"],
            "session_1_questions": r5["session_1_questions"],
            "session_2_questions": r5["session_2_questions"],
            "session_3_questions": r5["session_3_questions"],
            "session_1_answers":   r5["session_1_answers"],
            "session_2_answers":   r5["session_2_answers"],
            "session_3_answers":   r5["session_3_answers"],
        })

    except Exception as e:
        logger.error(f"  /pipeline/mcq EXCEPTION: {e}", exc_info=True)
        return err(f"MCQ pipeline failed: {str(e)}", 500)
    finally:
        if txt_path:
            try:
                if os.path.exists(txt_path):
                    os.unlink(txt_path)
            except Exception:
                pass


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
    """Legacy single-call pipeline. Prefer /pipeline/audio + /pipeline/mcq."""
    tmp_path = txt_path = None
    suffix   = os.path.splitext(file.filename)[1]
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            shutil.copyfileobj(file.file, tmp)

        try:
            r1 = await run_in_process(extract, filepath=tmp_path, save_output=False)
        except RuntimeError as e:
            if "queue_full" in str(e):
                return queue_full_error()
            raise

        if r1["status"] != "success":
            return err(f"Extraction failed: {r1.get('error')}")
        extracted_text = r1.get("text", "")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w", encoding="utf-8") as tf:
            tf.write(extracted_text)
            txt_path = tf.name

        try:
            r2 = await run_in_process(
                generate_tts,
                source_txt_path=txt_path,
                engine=tts_engine,
                voice_id=voice_id or None,
                file1_metadata=r1.get("metadata", {}),
            )
        except RuntimeError as e:
            if "queue_full" in str(e):
                return queue_full_error()
            raise

        if r2["status"] != "success":
            return err(f"TTS failed: {r2.get('error')}")

        try:
            r3 = await run_in_process(
                modulate_audio,
                input_wav_path=r2["output_path"],
                cognitive_state=cognitive_state,
            )
        except RuntimeError as e:
            if "queue_full" in str(e):
                return queue_full_error()
            raise

        if r3["status"] != "success":
            return err(f"Modulation failed: {r3.get('error')}")

        r4 = generate_visualization(
            modulated_wav_path=r3["output_path"],
            role=role, cognitive_state=cognitive_state,
            beat_freq_hz=r3["beat_freq_hz"],
            carrier_freq_hz=r3["carrier_freq_hz"],
            document_title=document_title,
        )

        try:
            r5 = await run_in_process(
                initialise_document,
                source_txt_path=txt_path,
                document_title=document_title,
            )
        except RuntimeError as e:
            if "queue_full" in str(e):
                return queue_full_error()
            raise

        if r5["status"] != "success":
            return err(f"MCQ init failed: {r5.get('error')}")

        return ok({
            "doc_id":             r5["document_id"],
            "document_title":     document_title,
            "extracted_text":     extracted_text,
            "cognitive_state":    cognitive_state,
            "beat_freq_hz":       r3["beat_freq_hz"],
            "carrier_freq_hz":    r3["carrier_freq_hz"],
            "duration_sec":       r3["duration_sec"],
            "word_count":         r1["word_count"],
            "llm_engine":         r5.get("llm_engine", "?"),
            "sessions_generated": r5["sessions_generated"],
            "session_state":      r5["session_state"],
            "session_1_questions":r5["session_1_questions"],
            "session_1_answers":  r5["session_1_answers"],
            "session_2_questions":r5["session_2_questions"],
            "session_2_answers":  r5["session_2_answers"],
            "session_3_questions":r5["session_3_questions"],
            "session_3_answers":  r5["session_3_answers"],
        })
    finally:
        for p in [tmp_path, txt_path]:
            try:
                if p and os.path.exists(p):
                    os.unlink(p)
            except Exception:
                pass


# ── Global exception handler ──────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(
        f"UNHANDLED | {request.method} {request.url.path} | "
        f"{type(exc).__name__}: {exc}", exc_info=True
    )
    return JSONResponse(
        status_code=500,
        content={"status": "error", "error": f"Internal error: {str(exc)}"}
    )


# ── Keep-alive ────────────────────────────────────────────────────────────────

async def _keep_alive_loop():
    self_url = os.getenv("RENDER_EXTERNAL_URL", "http://localhost:9801").rstrip("/")
    ping_url = f"{self_url}/health"
    interval = 10 * 60
    await asyncio.sleep(30)
    logger.info(f"[keep-alive] Started — {ping_url} every 10 min")
    async with httpx.AsyncClient(timeout=15.0) as client:
        while True:
            try:
                resp = await client.get(ping_url)
                logger.info(f"[keep-alive] Ping OK {resp.status_code}")
            except Exception as e:
                logger.warning(f"[keep-alive] Ping failed: {e}")
            await asyncio.sleep(interval)


# ── Lifecycle ─────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    logger.info("=" * 60)
    logger.info("Tarang 2.0.1 — Python Bridge (Production) STARTING")
    logger.info(f"  PROJECT_ROOT   : {PROJECT_ROOT}")
    logger.info(f"  Python         : {sys.version.split()[0]}")
    logger.info(f"  CPU workers    : {os.getenv('TARANG_CPU_WORKERS', 'auto')}")
    logger.info(f"  Max queue      : {os.getenv('TARANG_MAX_QUEUE', '50')}")
    logger.info(f"  Mode           : multiprocessing (ProcessPoolExecutor)")
    logger.info(f"  Fix applied    : /extract now returns raw_text for correct splitting")
    asyncio.create_task(_keep_alive_loop())
    logger.info("=" * 60)


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Tarang 2.0.1 — Bridge shutting down")
    await shutdown_pool()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("BRIDGE_PORT", "9801"))
    print(f"\nTarang 2.0.1 — Python Bridge (Production)")
    print(f"=========================================")
    print(f"http://localhost:{port} | Swagger: /docs")
    uvicorn.run("bridge:app", host="0.0.0.0", port=port, reload=False, log_level="info")
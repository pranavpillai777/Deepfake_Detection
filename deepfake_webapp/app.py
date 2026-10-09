"""FastAPI app: upload a video, poll for the result, download Grad-CAM + PDF."""
import logging
import os
import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import inference
from report import build_pdf

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("deepfake")

DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "200"))
JOB_TTL_SECONDS = int(os.getenv("JOB_TTL_SECONDS", "3600"))
MAX_PENDING_JOBS = int(os.getenv("MAX_PENDING_JOBS", "5"))
ALLOWED_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

# In-memory job table: fine for ONE server process. For several workers/servers,
# move this to Redis/a database and use a real queue (Celery/RQ).
jobs = {}
jobs_lock = threading.Lock()
executor = ThreadPoolExecutor(max_workers=1)  # one analysis at a time
detector = None


@asynccontextmanager
async def lifespan(_app):
    global detector
    shutil.rmtree(DATA_DIR, ignore_errors=True)  # leftovers from a previous run
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    detector = inference.Detector()              # load models once
    log.info("Models loaded")
    yield
    executor.shutdown(wait=False, cancel_futures=True)


app = FastAPI(title="Deepfake Video Check", lifespan=lifespan)


def _prune():
    now = time.time()
    with jobs_lock:
        stale = [k for k, v in jobs.items()
                 if v["status"] in ("done", "error") and now - v["created"] > JOB_TTL_SECONDS]
        for k in stale:
            jobs.pop(k)
            shutil.rmtree(DATA_DIR / k, ignore_errors=True)


def _run_job(job_id: str, video_path: Path, original_name: str):
    job = jobs[job_id]
    job["status"] = "running"
    try:
        result, cam = detector.analyze(
            str(video_path), progress_cb=lambda p: job.__setitem__("progress", p)
        )
        job_dir = video_path.parent
        if cam is not None:
            cv2.imwrite(str(job_dir / "gradcam.png"), cam)
        build_pdf(job_dir / "report.pdf", job_id, original_name, result, cam)
        job["result"] = result
        job["progress"] = 1.0
        job["status"] = "done"
    except inference.VideoRejected as e:
        job["status"], job["error"] = "error", str(e)
    except Exception:
        log.exception("Job %s failed", job_id)
        job["status"] = "error"
        job["error"] = "Something went wrong while analyzing this video. Try a different file."
    finally:
        video_path.unlink(missing_ok=True)  # never keep the user's video


@app.get("/healthz")
def healthz():
    return {"ok": detector is not None}


@app.post("/api/analyze", status_code=202)
async def analyze(file: UploadFile = File(...)):
    _prune()

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(415, "Upload an MP4, MOV, AVI, MKV or WebM video.")

    with jobs_lock:
        pending = sum(1 for v in jobs.values() if v["status"] in ("queued", "running"))
    if pending >= MAX_PENDING_JOBS:
        raise HTTPException(503, "The analyzer is busy. Try again in a few minutes.")

    job_id = uuid.uuid4().hex
    job_dir = DATA_DIR / job_id
    job_dir.mkdir(parents=True)
    video_path = job_dir / f"upload{suffix}"

    # Note: Starlette buffers the multipart body before this runs, so this check
    # only protects the disk. Enforce a hard cap at the proxy too (nginx
    # client_max_body_size, or your platform's request-size limit).
    limit = MAX_UPLOAD_MB * 1024 * 1024
    size = 0
    try:
        with open(video_path, "wb") as out:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > limit:
                    raise HTTPException(413, f"That video is larger than {MAX_UPLOAD_MB} MB.")
                out.write(chunk)
    except HTTPException:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise

    with jobs_lock:
        jobs[job_id] = {"status": "queued", "progress": 0.0, "result": None,
                        "error": None, "created": time.time()}
    executor.submit(_run_job, job_id, video_path, file.filename or "video")
    return {"job_id": job_id}


def _get_job(job_id: str) -> dict:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "This result has expired. Upload the video again.")
    return job


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    job = _get_job(job_id)
    return {k: job[k] for k in ("status", "progress", "result", "error")}


@app.get("/api/jobs/{job_id}/gradcam.png")
def job_gradcam(job_id: str):
    _get_job(job_id)
    path = DATA_DIR / job_id / "gradcam.png"
    if not path.exists():
        raise HTTPException(404, "No heatmap for this video.")
    return FileResponse(path, media_type="image/png")


@app.get("/api/jobs/{job_id}/report.pdf")
def job_report(job_id: str):
    job = _get_job(job_id)
    path = DATA_DIR / job_id / "report.pdf"
    if job["status"] != "done" or not path.exists():
        raise HTTPException(404, "The report isn't ready.")
    return FileResponse(path, media_type="application/pdf",
                        filename=f"deepfake_report_{job_id[:8]}.pdf")


# Mounted last so the /api routes win.
app.mount("/", StaticFiles(directory="static", html=True), name="static")

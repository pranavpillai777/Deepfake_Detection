"""Local, single-worker API controlling the unchanged desktop runner."""
import json
import logging
import os
import re
import subprocess
import sys
import threading
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("deepfake.live")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TERMINAL = {"done", "error", "cancelled"}
STAGES = {"starting", "countdown", "selecting_region", "capturing", "analyzing", *TERMINAL}


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def read_shared_text(path):
    """Let Windows replace status.json atomically while we read its old version."""
    if os.name != "nt":
        return path.read_text(encoding="utf-8")
    import ctypes
    import msvcrt
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    create = kernel.CreateFileW
    create.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                       wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    create.restype = wintypes.HANDLE
    close = kernel.CloseHandle
    close.argtypes = [wintypes.HANDLE]
    close.restype = wintypes.BOOL
    # GENERIC_READ, FILE_SHARE_READ|WRITE|DELETE, OPEN_EXISTING.
    handle = create(str(path), 0x80000000, 0x7, None, 3, 0x80, None)
    if handle == wintypes.HANDLE(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        fd = msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
    except BaseException:
        close(handle)
        raise
    with os.fdopen(fd, "r", encoding="utf-8") as source:
        return source.read()


@dataclass
class Session:
    session_id: str
    process: subprocess.Popen
    started: str
    stop_requested: bool = False
    control_error: str | None = None


def create_app(*, report_dir=None, runner_path=None, checkpoint_path=None,
               detector_path=None):
    """Explicit paths also allow tests to substitute a harmless fake runner."""
    reports = Path(report_dir or os.getenv("REPORT_DIR", PROJECT_ROOT / "V6" / "reports")).expanduser().resolve()
    runner = Path(runner_path or PROJECT_ROOT / "V6" / "loggin_runner.py").resolve()
    checkpoint = Path(checkpoint_path or os.getenv("CHECKPOINT_PATH", PROJECT_ROOT / "checkpoints" / "mobilenet_v2_deepfake.pth")).expanduser().resolve()
    detector = Path(detector_path or os.getenv("YOLO_WEIGHTS", PROJECT_ROOT / "checkpoints" / "yolov8n.pt")).expanduser().resolve()
    lock = threading.Lock()
    current = None
    shutdown = threading.Event()
    watchers = []

    def session_dir(session_id):
        if not re.fullmatch(r"[A-Za-z0-9-]{1,100}", session_id):
            raise HTTPException(400, "Invalid session ID: use only letters, digits and dashes.")
        path = (reports / session_id).resolve()
        if path.parent != reports:
            raise HTTPException(400, "Session folder must be inside the reports folder.")
        return path

    def session_file(session_id, name):
        folder = session_dir(session_id)
        path = folder / name
        resolved = path.resolve()
        if resolved.parent != folder or not resolved.is_relative_to(reports):
            raise HTTPException(400, "Session file must be inside its reports folder.")
        return path

    def read_json(path):
        try:
            data = json.loads(read_shared_text(path))
        except FileNotFoundError:
            return None
        except (OSError, ValueError) as exc:
            raise HTTPException(500, f"Cannot read {path.name}: {exc}") from exc
        if not isinstance(data, dict):
            raise HTTPException(500, f"Invalid {path.name}: expected a JSON object.")
        return data

    def deliver_stop(session):
        # The runner creates its own folder with exist_ok=False. Do not create
        # it here: an early Stop is remembered until the runner has created it.
        folder = session_dir(session.session_id)
        if folder.is_dir():
            session_file(session.session_id, "stop").touch(exist_ok=True)
            return True
        return False

    def watch(session):
        while not shutdown.is_set():
            with lock:
                if session.stop_requested and session.process.poll() is None:
                    try:
                        deliver_stop(session)
                    except (OSError, HTTPException) as exc:
                        session.control_error = f"Cannot deliver Stop: {exc}"
                if session.process.poll() is not None:
                    return
            shutdown.wait(0.05)

    def snapshot(session):
        # Poll before reading: after observing process exit, its final atomic
        # status write is already complete, avoiding a false death diagnosis.
        exit_code = session.process.poll()
        data = read_json(session_file(session.session_id, "status.json"))
        if data is None:
            data = {"status": "starting", "timestamp": session.started}
        data = dict(data)
        if data.get("status") not in STAGES:
            data.update(status="error", message="Runner wrote an invalid status.", timestamp=timestamp())
        if exit_code is not None and data.get("status") not in TERMINAL:
            data.update(status="error", message=f"Runner exited with code {exit_code} before writing a final status.", timestamp=timestamp())
        if session.control_error:
            data.update(status="error", message=session.control_error, timestamp=timestamp())
        data.update(session_id=session.session_id, stage=data["status"],
                    process_alive=exit_code is None, exit_code=exit_code,
                    stop_requested=session.stop_requested)
        data.setdefault("message", None)
        return data

    def completed(session_id):
        path = session_file(session_id, "status.json")
        with lock:
            if current is not None and current.session_id == session_id:
                status = snapshot(current)
            else:
                status = read_json(path)
        if status is None:
            raise HTTPException(404, "Session not found.")
        if status.get("status") != "done":
            raise HTTPException(409, status.get("message") or "The analysis is not done yet.")

    @asynccontextmanager
    async def lifespan(_app):
        for label, path in (("Classifier checkpoint", checkpoint), ("Detector weights", detector), ("Runner script", runner)):
            if not path.is_file():
                raise RuntimeError(f"{label} not found: {path}")
            log.info("%s found: %s", label, path)
        reports.mkdir(parents=True, exist_ok=True)
        log.info("Reports folder: %s", reports)
        yield
        # Ask an active desktop session to finish when the server shuts down.
        with lock:
            session = current
            if session is not None and session.process.poll() is None:
                session.stop_requested = True
        if session is not None and session.process.poll() is None:
            try:
                session.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                log.warning("Runner did not finish on shutdown (possibly selecting a region); terminating it.")
                session.process.terminate()
                try:
                    session.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    session.process.kill()
                    session.process.wait()
        shutdown.set()
        for watcher in watchers:
            watcher.join(timeout=1)

    api = FastAPI(title="V6 Local Live Analysis", lifespan=lifespan)

    @api.post("/api/live/start", status_code=202)
    def start():
        nonlocal current
        with lock:
            if current is not None and current.process.poll() is None:
                raise HTTPException(409, "An analysis is already running")
            session_id = uuid.uuid4().hex
            command = [sys.executable, "-u", str(runner),
                       "--session-id", session_id, "--report-dir", str(reports),
                       "--checkpoint", str(checkpoint), "--detector", str(detector)]
            try:
                # No hidden/no-window flags: the runner inherits the interactive
                # desktop and console. Run the backend in your signed-in session.
                process = subprocess.Popen(command, cwd=PROJECT_ROOT)
            except OSError as exc:
                raise HTTPException(500, f"Cannot launch live analysis: {exc}") from exc
            current = Session(session_id, process, timestamp())
            watcher = threading.Thread(target=watch, args=(current,), daemon=True)
            watchers.append(watcher)
            watcher.start()
            return {"session_id": session_id}

    @api.post("/api/live/stop")
    def stop():
        with lock:
            if current is None:
                raise HTTPException(409, "No analysis has been started.")
            if current.process.poll() is not None:
                return {"session_id": current.session_id, "stop_requested": current.stop_requested,
                        "process_alive": False}
            current.stop_requested = True
            try:
                delivered = deliver_stop(current)
            except OSError as exc:
                raise HTTPException(500, f"Cannot create stop file: {exc}") from exc
            return {"session_id": current.session_id, "stop_requested": True,
                    "stop_file_created": delivered, "process_alive": True}

    @api.get("/api/live/status")
    def status():
        with lock:
            if current is None:
                return {"session_id": None, "status": "idle", "stage": "idle",
                        "message": None, "timestamp": timestamp(), "process_alive": False}
            return snapshot(current)

    @api.get("/api/live/result/{session_id:path}")
    def result(session_id: str):
        completed(session_id)
        data = read_json(session_file(session_id, "result.json"))
        if data is None:
            raise HTTPException(404, "The result file is missing.")
        return data

    @api.get("/api/live/report/{session_id:path}")
    def report(session_id: str):
        completed(session_id)
        path = session_file(session_id, "report.pdf")
        if not path.is_file():
            raise HTTPException(404, "The PDF report is missing.")
        return FileResponse(path, media_type="application/pdf", filename="report.pdf")

    return api


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, workers=1)

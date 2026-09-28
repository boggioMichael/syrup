"""The lipreader inference API (see example/docs/api.md).

    python -m lipreader_api                      # http://127.0.0.1:8765
    LIPREADER_API_TOKEN=secret python -m lipreader_api --host 0.0.0.0

Starlette + uvicorn; analysis runs in a worker thread per job; uploads are
deleted the moment their frames have been read; results live in memory
until DELETE or the TTL.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import numpy as np
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, Response
from starlette.routing import Route

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "..", "lipreader")))
from lipreader import __version__, audio, export  # noqa: E402
from lipreader.pipeline import Analyzer, analyze  # noqa: E402
from lipreader.schema import AnalysisResult, Options, VideoInfo  # noqa: E402
from lipreader.video import Frame  # noqa: E402
from lipreader.vsr import registry  # noqa: E402


@dataclass
class Config:
    token: Optional[str] = os.environ.get("LIPREADER_API_TOKEN") or None
    rate_limit_per_minute: int = int(os.environ.get("LIPREADER_RATE_LIMIT", "30"))
    max_upload_mb: int = int(os.environ.get("LIPREADER_MAX_UPLOAD_MB", "512"))
    result_ttl_minutes: int = int(os.environ.get("LIPREADER_RESULT_TTL_MINUTES", "60"))
    allow_local_files: bool = os.environ.get("LIPREADER_ALLOW_LOCAL_FILES") == "1"
    processing: str = os.environ.get("LIPREADER_PROCESSING", "local")  # what the UI shows: local | server
    workers: int = int(os.environ.get("LIPREADER_WORKERS", "1"))
    # The extension, local pages, and a page opened from disk (Origin: null).
    cors_origins: str = os.environ.get("LIPREADER_CORS_ORIGINS", r"chrome-extension://.*|https?://(localhost|127\.0\.0\.1)(:\d+)?|null")
    workdir: str = os.environ.get("LIPREADER_WORKDIR", "")


def now() -> datetime:
    return datetime.now(timezone.utc)


def iso(t: Optional[datetime]) -> Optional[str]:
    return t.isoformat().replace("+00:00", "Z") if t else None


def error(message: str, status: int = 400, code: str = "bad_request", **extra) -> JSONResponse:
    return JSONResponse({"error": message, "code": code, **extra}, status_code=status)


# --- jobs --------------------------------------------------------------------------

@dataclass
class Job:
    id: str
    options: Options
    source: str
    created: datetime
    status: str = "queued"
    progress: float = 0.0
    finished: Optional[datetime] = None
    result: Optional[AnalysisResult] = None
    error: Optional[str] = None
    path: Optional[str] = None          # the video on disk while it is needed
    changed: threading.Event = field(default_factory=threading.Event)

    def to_dict(self, processing: str, ttl: timedelta) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "id": self.id, "status": self.status, "progress": round(self.progress, 3), "createdAt": iso(self.created),
            "finishedAt": iso(self.finished), "expiresAt": iso((self.finished or self.created) + ttl),
            "options": self.options.to_dict(), "error": self.error, "processing": processing, "source": self.source,
        }
        if self.result is not None:
            d["result"] = self.result.to_dict()
        return d


class Fetchers:
    """Turn what the client sent into a local video file."""

    def __init__(self, config: Config, workdir: str):
        self.config = config
        self.workdir = workdir

    def available(self) -> List[str]:
        out = ["upload", "http"]
        if self.config.allow_local_files:
            out.append("local-file")
        if shutil.which("yt-dlp") or self._has_module("yt_dlp"):
            out.append("youtube")
        return out

    @staticmethod
    def _has_module(name: str) -> bool:
        import importlib.util

        return importlib.util.find_spec(name) is not None

    def fetch(self, url: str, job_id: str) -> str:
        target = os.path.join(self.workdir, f"{job_id}.video")
        if url.startswith("file://"):
            if not self.config.allow_local_files:
                raise PermissionError("local files are not allowed (LIPREADER_ALLOW_LOCAL_FILES=1 enables them)")
            path = url[len("file://"):]
            if not os.path.isfile(path):
                raise FileNotFoundError(path)
            os.symlink(os.path.abspath(path), target)
            return target
        if re.match(r"https?://(www\.|m\.)?(youtube\.com|youtu\.be)/", url):
            if "youtube" not in self.available():
                raise RuntimeError("YouTube URLs need yt-dlp on this machine (pip install yt-dlp); or let the extension capture the tab")
            cmd = [shutil.which("yt-dlp") or sys.executable, *([] if shutil.which("yt-dlp") else ["-m", "yt_dlp"]),
                   "-f", "bv*[height<=720][ext=mp4]+ba[ext=m4a]/b[height<=720][ext=mp4]/b", "--no-playlist", "--quiet",
                   "-o", target, url]
            done = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
            if done.returncode != 0:
                raise RuntimeError(f"yt-dlp failed: {done.stderr.strip()[-400:]}")
            if not os.path.exists(target):
                candidates = [f for f in os.listdir(self.workdir) if f.startswith(job_id)]
                if not candidates:
                    raise RuntimeError("yt-dlp wrote nothing")
                os.rename(os.path.join(self.workdir, candidates[0]), target)
            return target
        if url.startswith(("http://", "https://")):
            import httpx

            limit = self.config.max_upload_mb * 1024 * 1024
            with httpx.stream("GET", url, follow_redirects=True, timeout=60) as r:
                r.raise_for_status()
                size = 0
                with open(target, "wb") as f:
                    for chunk in r.iter_bytes():
                        size += len(chunk)
                        if size > limit:
                            raise ValueError(f"download exceeds {self.config.max_upload_mb} MB")
                        f.write(chunk)
            return target
        raise ValueError("unsupported URL; use http(s)://, a YouTube link, or upload the file")


class JobStore:
    def __init__(self, config: Config, workdir: str):
        self.config = config
        self.workdir = workdir
        self.jobs: Dict[str, Job] = {}
        self.lock = threading.Lock()
        self.pool = ThreadPoolExecutor(max_workers=max(1, config.workers))
        self.fetchers = Fetchers(config, workdir)

    def create(self, options: Options, source: str, path: Optional[str] = None, url: Optional[str] = None) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], options=options, source=source, created=now(), path=path)
        with self.lock:
            self.jobs[job.id] = job
        self.pool.submit(self._run, job, url)
        return job

    def _run(self, job: Job, url: Optional[str]) -> None:
        try:
            job.status = "running"
            job.changed.set()
            if url:
                job.path = self.fetchers.fetch(url, job.id)

            def progress(p: float) -> None:
                job.progress = p
                job.changed.set()

            job.result = analyze(job.path, job.options, progress=progress)
            job.status = "done"
            job.progress = 1.0
        except Exception as exc:  # noqa: BLE001 - reported to the client
            job.status = "failed"
            job.error = f"{type(exc).__name__}: {exc}"
        finally:
            self._forget_file(job)
            job.finished = now()
            job.changed.set()

    @staticmethod
    def _forget_file(job: Job) -> None:
        if job.path and os.path.lexists(job.path):
            try:
                os.remove(job.path)
            except OSError:
                pass
        job.path = None

    def get(self, job_id: str) -> Optional[Job]:
        return self.jobs.get(job_id)

    def delete(self, job_id: str) -> bool:
        with self.lock:
            job = self.jobs.pop(job_id, None)
        if job is None:
            return False
        self._forget_file(job)
        job.result = None
        job.status = "deleted"
        return True

    def expire(self) -> None:
        ttl = timedelta(minutes=self.config.result_ttl_minutes)
        cutoff = now() - ttl
        with self.lock:
            stale = [j for j in self.jobs.values() if j.finished and j.finished < cutoff]
        for job in stale:
            self.delete(job.id)


# --- streamed sessions ----------------------------------------------------------------

@dataclass
class Session:
    id: str
    analyzer: Analyzer
    created: datetime
    frames: int = 0
    last_t: float = 0.0
    finished: bool = False
    result: Optional[AnalysisResult] = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    def snapshot(self) -> Dict[str, Any]:
        a = self.analyzer
        if self.result is not None:
            return {"id": self.id, "finished": True, "frames": self.frames, "result": self.result.to_dict()}
        return {
            "id": self.id, "finished": False, "frames": self.frames, "lastTimestamp": self.last_t,
            "segments": [s.to_dict() for s in sorted(a.segments, key=lambda s: s.start)],
            "tracks": [t.to_person_track().to_dict() for t in a.tracker.live],
            "warnings": list(a.warnings),
        }


class SessionStore:
    def __init__(self):
        self.sessions: Dict[str, Session] = {}
        self.lock = threading.Lock()

    def create(self, options: Options, width: int, height: int, fps: float, source: str) -> Session:
        info = VideoInfo(duration=0.0, fps=fps, width=width, height=height, source=source, sampled_fps=fps)
        analyzer = Analyzer(options, info, live=True)
        session = Session(uuid.uuid4().hex[:12], analyzer, now())
        with self.lock:
            self.sessions[session.id] = session
        return session

    def get(self, session_id: str) -> Optional[Session]:
        return self.sessions.get(session_id)

    def delete(self, session_id: str) -> bool:
        with self.lock:
            return self.sessions.pop(session_id, None) is not None


# --- middleware -------------------------------------------------------------------------

class AuthAndLimits(BaseHTTPMiddleware):
    def __init__(self, app, config: Config):
        super().__init__(app)
        self.config = config
        self.buckets: Dict[str, List[float]] = {}
        self.lock = threading.Lock()

    async def dispatch(self, request: Request, call_next):
        if request.method == "OPTIONS":
            return await call_next(request)
        if self.config.token:
            header = request.headers.get("authorization", "")
            if header != f"Bearer {self.config.token}":
                return error("missing or wrong bearer token", 401, "unauthorized")
        if request.method == "POST" and request.url.path in ("/jobs", "/sessions"):
            client = request.client.host if request.client else "?"
            t = time.time()
            with self.lock:
                bucket = [x for x in self.buckets.get(client, []) if t - x < 60]
                if len(bucket) >= self.config.rate_limit_per_minute:
                    retry = int(60 - (t - bucket[0])) + 1
                    return JSONResponse({"error": "rate limit exceeded", "code": "rate_limited"}, status_code=429,
                                        headers={"Retry-After": str(retry)})
                bucket.append(t)
                self.buckets[client] = bucket
        return await call_next(request)


# --- the app --------------------------------------------------------------------------------

def language_table() -> List[Dict[str, Any]]:
    asr = audio.WhisperRecogniser.available()
    out = []
    for status in registry.languages():
        d = status.to_dict()
        d["audio"] = {"available": asr, "model": "whisper" if asr else None,
                      "note": "" if asr else "install faster-whisper (pip install faster-whisper) for the audio modes"}
        out.append(d)
    return out


def create_app(config: Optional[Config] = None) -> Starlette:
    config = config or Config()
    workdir = config.workdir or tempfile.mkdtemp(prefix="lipreader-")
    os.makedirs(workdir, exist_ok=True)
    jobs = JobStore(config, workdir)
    sessions = SessionStore()
    ttl = timedelta(minutes=config.result_ttl_minutes)

    async def health(request: Request):
        jobs.expire()
        return JSONResponse({"ok": True, "version": __version__, "processing": config.processing,
                             "languages": language_table(), "fetchers": jobs.fetchers.available()})

    async def languages(request: Request):
        return JSONResponse(language_table())

    async def create_job(request: Request):
        jobs.expire()
        content_type = request.headers.get("content-type", "")
        try:
            if content_type.startswith("multipart/form-data"):
                form = await request.form()
                upload = form.get("file")
                if upload is None or not hasattr(upload, "read"):
                    return error("multipart body needs a `file` field")
                raw_options = form.get("options")
                options = Options.from_dict(json.loads(raw_options) if raw_options else {})
                job_id = uuid.uuid4().hex[:12]
                path = os.path.join(workdir, f"{job_id}.upload")
                limit = config.max_upload_mb * 1024 * 1024
                size = 0
                with open(path, "wb") as f:
                    while True:
                        chunk = await upload.read(1 << 20)
                        if not chunk:
                            break
                        size += len(chunk)
                        if size > limit:
                            f.close()
                            os.remove(path)
                            return error(f"upload exceeds {config.max_upload_mb} MB", 413, "too_large")
                        f.write(chunk)
                source = getattr(upload, "filename", "") or "upload"
                await upload.close()
                job = jobs.create(options, source, path=path)
            else:
                body = await request.json()
                url = body.get("url")
                if not url:
                    return error("JSON body needs `url`")
                options = Options.from_dict(body.get("options") or {})
                job = jobs.create(options, url, url=url)
        except (ValueError, json.JSONDecodeError) as exc:
            return error(str(exc))
        return JSONResponse(job.to_dict(config.processing, ttl), status_code=202)

    async def get_job(request: Request):
        job = jobs.get(request.path_params["job_id"])
        if job is None:
            return error("no such job", 404, "not_found")
        wait = min(float(request.query_params.get("wait", "0") or 0), 30.0)
        if wait > 0 and job.status in ("queued", "running"):
            job.changed.clear()
            await asyncio.get_running_loop().run_in_executor(None, job.changed.wait, wait)
        return JSONResponse(job.to_dict(config.processing, ttl))

    async def export_job(request: Request):
        job = jobs.get(request.path_params["job_id"])
        if job is None:
            return error("no such job", 404, "not_found")
        if job.result is None:
            return error("job is not done", 409, "not_ready", status=job.status)
        fmt = request.query_params.get("format", "srt").lower()
        if fmt not in export.FORMATS:
            return error(f"format must be one of {export.FORMATS}")
        tracks = request.query_params.getlist("track")
        selected = [int(t) for t in tracks] if tracks else None
        body = export.export(job.result, fmt, selected)
        name = os.path.splitext(os.path.basename(job.source))[0] or "transcript"
        return Response(body, media_type=export.MIME[fmt],
                        headers={"Content-Disposition": f'attachment; filename="{name}.{fmt}"'})

    async def delete_job(request: Request):
        if not jobs.delete(request.path_params["job_id"]):
            return error("no such job", 404, "not_found")
        return Response(status_code=204)

    async def create_session(request: Request):
        try:
            body = await request.json()
            options = Options.from_dict(body.get("options") or {})
            width, height = int(body["width"]), int(body["height"])
            fps = float(body.get("fps", 25.0))
        except (KeyError, ValueError, TypeError, json.JSONDecodeError) as exc:
            return error(f"body needs width, height, fps, options: {exc}")
        session = sessions.create(options, width, height, fps, str(body.get("source", "stream")))
        return JSONResponse(session.snapshot(), status_code=201)

    async def push_frames(request: Request):
        session = sessions.get(request.path_params["session_id"])
        if session is None:
            return error("no such session", 404, "not_found")
        if session.finished:
            return error("session is finished", 409, "finished")
        form = await request.form()
        try:
            timestamps = json.loads(form.get("timestamps") or "[]")
        except json.JSONDecodeError:
            return error("timestamps must be a JSON array of seconds")
        uploads = form.getlist("frames")
        if len(uploads) != len(timestamps):
            return error(f"{len(uploads)} frames but {len(timestamps)} timestamps")
        import cv2

        blobs = []
        for u in uploads:
            blobs.append(await u.read())
            await u.close()

        def work():
            with session.lock:
                for blob, t in zip(blobs, timestamps):
                    array = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)
                    if array is None:
                        continue
                    rgb = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
                    session.analyzer.push(Frame(session.frames, float(t), rgb))
                    session.frames += 1
                    session.last_t = float(t)
                return session.snapshot()

        snapshot = await asyncio.get_running_loop().run_in_executor(jobs.pool, work)
        return JSONResponse(snapshot)

    async def get_session(request: Request):
        session = sessions.get(request.path_params["session_id"])
        if session is None:
            return error("no such session", 404, "not_found")
        return JSONResponse(session.snapshot())

    async def finish_session(request: Request):
        session = sessions.get(request.path_params["session_id"])
        if session is None:
            return error("no such session", 404, "not_found")

        def work():
            with session.lock:
                if not session.finished:
                    session.analyzer.video.duration = session.last_t
                    session.result = session.analyzer.finish()
                    session.finished = True
                return session.snapshot()

        return JSONResponse(await asyncio.get_running_loop().run_in_executor(jobs.pool, work))

    async def delete_session(request: Request):
        if not sessions.delete(request.path_params["session_id"]):
            return error("no such session", 404, "not_found")
        return Response(status_code=204)

    async def not_found(request: Request, exc):
        return error("not found", 404, "not_found")

    routes = [
        Route("/health", health),
        Route("/languages", languages),
        Route("/jobs", create_job, methods=["POST"]),
        Route("/jobs/{job_id}", get_job),
        Route("/jobs/{job_id}/export", export_job),
        Route("/jobs/{job_id}", delete_job, methods=["DELETE"]),
        Route("/sessions", create_session, methods=["POST"]),
        Route("/sessions/{session_id}/frames", push_frames, methods=["POST"]),
        Route("/sessions/{session_id}/finish", finish_session, methods=["POST"]),
        Route("/sessions/{session_id}", get_session),
        Route("/sessions/{session_id}", delete_session, methods=["DELETE"]),
    ]
    middleware = [
        Middleware(CORSMiddleware, allow_origin_regex=config.cors_origins, allow_methods=["*"], allow_headers=["*"], expose_headers=["Content-Disposition"]),
        Middleware(AuthAndLimits, config=config),
    ]
    app = Starlette(routes=routes, middleware=middleware, exception_handlers={404: not_found})
    app.state.config = config
    app.state.jobs = jobs
    app.state.sessions = sessions
    app.state.workdir = workdir
    return app


def main(argv=None):
    import argparse

    import uvicorn

    ap = argparse.ArgumentParser(prog="lipreader-api")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args(argv)
    config = Config()
    if args.host not in ("127.0.0.1", "localhost", "::1") and config.processing == "local":
        config.processing = "server"
    if config.processing == "server" and not config.token:
        print("warning: serving beyond localhost without LIPREADER_API_TOKEN", file=sys.stderr)
    uvicorn.run(create_app(config), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()

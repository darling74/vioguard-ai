"""Background analysis jobs with real progress reporting and guaranteed file cleanup."""
from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Optional

from video_processor import VideoAnalyzer, VideoDecodeError
from violence_model import ModelUnavailableError

log = logging.getLogger(__name__)


def safe_remove(path: Path | str) -> None:
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
    except OSError as exc:  # pragma: no cover - e.g. Windows file lock
        log.warning("Could not delete %s: %s", path, exc)


def purge_stale_files(directory: Path, max_age_seconds: int) -> int:
    """Delete leftovers (e.g. from a crash) older than ``max_age_seconds``."""
    removed, now = 0, time.time()
    for p in directory.glob("*"):
        if p.is_file() and p.name != ".gitkeep" and now - p.stat().st_mtime > max_age_seconds:
            safe_remove(p)
            removed += 1
    return removed


def error_payload(exc: Exception) -> Dict[str, Any]:
    if isinstance(exc, ModelUnavailableError):
        return {"code": exc.code, "message": str(exc), "http": 503}
    if isinstance(exc, VideoDecodeError):
        return {"code": "invalid_video", "message": str(exc), "http": 422}
    return {"code": "internal_error", "message": "Unexpected error while analysing the video.", "http": 500}


class JobManager:
    """Single-worker queue: inference is CPU/GPU bound, so jobs run one at a time."""

    def __init__(self, analyzer: VideoAnalyzer, ttl_seconds: int = 3600):
        self.analyzer = analyzer
        self.ttl = ttl_seconds
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="analysis")
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def submit(self, video_path: Path, original_name: str) -> str:
        self._prune()
        job_id = uuid.uuid4().hex
        job = {
            "id": job_id, "state": "queued", "created": time.time(), "filename": original_name,
            "progress": {"stage": "queued", "percent": 0.0, "segments_done": 0, "segments_total": 0},
            "result": None, "error": None,
        }
        with self._lock:
            self._jobs[job_id] = job
        self._pool.submit(self._run, job_id, Path(video_path))
        return job_id

    def _progress(self, job_id: str, stage: str, frac: float, extra: Dict[str, Any]) -> None:
        # Overall percent: probing+decoding is ~35 % of the work, inference ~65 %.
        percent = {"probing": 1, "decoding": 2 + 33 * frac, "inference": 35 + 65 * frac, "done": 100}.get(stage, 0)
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job["state"] = "running" if stage != "done" else job["state"]
                prog = job["progress"]
                prog.update(stage=stage, percent=round(max(prog["percent"], percent), 1))
                for k in ("segments_done", "segments_total"):
                    if k in extra:
                        prog[k] = extra[k]

    def _run(self, job_id: str, path: Path) -> None:
        try:
            result = self.analyzer.analyze(path, lambda s, f, e: self._progress(job_id, s, f, e))
            with self._lock:
                job = self._jobs[job_id]
                job.update(state="done", result=result)
                job["progress"].update(stage="done", percent=100.0)
            log.info("Job %s done: %s (%.2fs)", job_id, result["label"], result["processing_time_sec"])
        except Exception as exc:
            err = error_payload(exc)
            if err["http"] == 500:
                log.exception("Job %s failed", job_id)
            else:
                log.warning("Job %s failed: %s", job_id, exc)
            with self._lock:
                self._jobs[job_id].update(state="error", error=err)
        finally:
            safe_remove(path)  # uploaded video never outlives its job

    def get(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            job = self._jobs.get(job_id)
            return None if job is None else {k: (dict(v) if isinstance(v, dict) else v) for k, v in job.items()}

    def _prune(self) -> None:
        cutoff = time.time() - self.ttl
        with self._lock:
            for jid in [j for j, v in self._jobs.items() if v["created"] < cutoff and v["state"] in {"done", "error"}]:
                del self._jobs[jid]

"""Flask application: AI violence detection in videos (pretrained X3D-M)."""
from __future__ import annotations

import logging
import threading
import uuid
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Optional

from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge

from config import Settings, ModelSpec, load_model_spec
from jobs import JobManager, error_payload, purge_stale_files, safe_remove
from video_processor import VideoAnalyzer, VideoDecodeError, probe_video
from violence_model import ModelUnavailableError, get_model

log = logging.getLogger("violence_app")


def configure_logging(settings: Settings) -> None:
    root = logging.getLogger()
    if any(getattr(h, "_violence_app", False) for h in root.handlers):
        return
    settings.log_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    file_h = RotatingFileHandler(settings.log_dir / "app.log", maxBytes=2_000_000, backupCount=3, encoding="utf-8")
    console = logging.StreamHandler()
    for h in (file_h, console):
        h.setFormatter(fmt)
        h._violence_app = True  # type: ignore[attr-defined]
        root.addHandler(h)
    root.setLevel(getattr(logging, settings.log_level, logging.INFO))


def api_error(code: str, message: str, status: int):
    return jsonify({"ok": False, "error": {"code": code, "message": message}}), status


def create_app(settings: Optional[Settings] = None, model: Any = None, spec: Optional[ModelSpec] = None) -> Flask:
    """Application factory.

    ``model`` may be injected (automated tests only); in normal use the cached
    pretrained :class:`ViolenceModel` is created from ``models/model_config.json``.
    """
    settings = settings or Settings.from_env()
    settings.ensure_dirs()
    configure_logging(settings)
    spec = spec or load_model_spec(settings.model_config_path)
    model = model if model is not None else get_model(spec, settings)

    app = Flask(__name__)
    app.config.update(SECRET_KEY=settings.secret_key, MAX_CONTENT_LENGTH=settings.max_upload_bytes + 1024 * 1024)
    analyzer = VideoAnalyzer(model, spec, settings.min_video_frames, settings.max_video_seconds,
                             settings.inference_batch_size)
    jobs = JobManager(analyzer, settings.job_ttl_seconds)
    app.extensions["violence"] = {"settings": settings, "spec": spec, "model": model, "jobs": jobs, "analyzer": analyzer}

    removed = purge_stale_files(settings.upload_dir, settings.stale_upload_seconds)
    if removed:
        log.info("Removed %d stale upload(s)", removed)
    if settings.model_preload and model.status().get("state") == "not_loaded":
        def _preload() -> None:
            try:
                model.load()
            except ModelUnavailableError as exc:
                log.error("Model preload failed: %s", exc)
        threading.Thread(target=_preload, name="model-preload", daemon=True).start()

    # ------------------------------------------------------------------ helpers
    def allowed(filename: str) -> bool:
        return "." in filename and filename.rsplit(".", 1)[1].lower() in settings.allowed_extensions

    def accept_upload():
        """Validate and store the upload. Returns (path, original_name) or an error response."""
        status = model.status()
        if not status["available"] and status["state"] != "ready":
            return None, api_error(status["state"], status["message"], 503)
        file = request.files.get("video")
        if file is None or not file.filename:
            return None, api_error("no_file", "No video file was sent (form field 'video').", 400)
        if not allowed(file.filename):
            return None, api_error("bad_extension", "Unsupported file type. Allowed: "
                                   + ", ".join(e.upper() for e in settings.allowed_extensions) + ".", 400)
        ext = file.filename.rsplit(".", 1)[1].lower()
        path = settings.upload_dir / f"{uuid.uuid4().hex}.{ext}"
        file.save(path)
        try:
            if path.stat().st_size == 0:
                raise VideoDecodeError("The uploaded file is empty.")
            if path.stat().st_size > settings.max_upload_bytes:
                raise RequestEntityTooLarge()
            probe_video(path, settings.min_video_frames, settings.max_video_seconds)  # decoding validation
        except RequestEntityTooLarge:
            safe_remove(path)
            return None, api_error("too_large", f"File exceeds the {settings.max_upload_mb} MB limit.", 413)
        except VideoDecodeError as exc:
            safe_remove(path)
            return None, api_error("invalid_video", str(exc), 422)
        except Exception:
            safe_remove(path)
            log.exception("Upload validation failed")
            return None, api_error("internal_error", "Could not validate the uploaded video.", 500)
        return (path, file.filename), None

    # ------------------------------------------------------------------- routes
    @app.get("/")
    def home():
        return render_template("index.html", spec=spec, settings=settings)

    @app.get("/api/health")
    def health():
        return jsonify({
            "ok": True,
            "model": model.status(),
            "settings": {
                "max_upload_mb": settings.max_upload_mb,
                "allowed_extensions": list(settings.allowed_extensions),
                "threshold": spec.threshold,
                "aggregation": spec.aggregation,
                "segment_seconds": spec.segment_seconds,
            },
        })

    @app.post("/api/analyze")
    def analyze_async():
        accepted, err = accept_upload()
        if err:
            return err
        path, name = accepted
        job_id = jobs.submit(path, name)  # the job deletes the file when finished
        return jsonify({"ok": True, "job_id": job_id}), 202

    @app.get("/api/jobs/<job_id>")
    def job_status(job_id: str):
        job = jobs.get(job_id)
        if job is None:
            return api_error("not_found", "Unknown or expired job.", 404)
        body: Dict[str, Any] = {"ok": True, "job_id": job_id, "state": job["state"], "progress": job["progress"]}
        if job["state"] == "done":
            body["result"] = job["result"]
        if job["state"] == "error":
            body["error"] = {k: job["error"][k] for k in ("code", "message")}
        return jsonify(body)

    @app.post("/predict")
    def predict_sync():
        """Synchronous endpoint (compatible with the original project's /predict)."""
        accepted, err = accept_upload()
        if err:
            return err
        path, _ = accepted
        try:
            result = analyzer.analyze(path)
            return jsonify({"ok": True, **result})
        except Exception as exc:
            payload = error_payload(exc)
            if payload["http"] == 500:
                log.exception("Prediction failed")
            return api_error(payload["code"], payload["message"], payload["http"])
        finally:
            safe_remove(path)

    # ----------------------------------------------------------- error handling
    @app.errorhandler(RequestEntityTooLarge)
    def too_large(_exc):
        return api_error("too_large", f"File exceeds the {settings.max_upload_mb} MB limit.", 413)

    @app.errorhandler(HTTPException)
    def http_error(exc: HTTPException):
        return api_error((exc.name or "error").lower().replace(" ", "_"), exc.description or exc.name, exc.code or 500)

    @app.errorhandler(Exception)
    def unhandled(exc: Exception):
        log.exception("Unhandled error")
        return api_error("internal_error", "An unexpected server error occurred.", 500)

    return app


def main() -> None:
    settings = Settings.from_env()
    app = create_app(settings)
    log.info("Starting on http://%s:%d (debug=%s)", settings.host, settings.port, settings.debug)
    if settings.debug:
        log.warning("DEBUG is on - never use this in production.")
        app.run(host=settings.host, port=settings.port, debug=True, use_reloader=False)
        return
    try:
        from waitress import serve
        serve(app, host=settings.host, port=settings.port, threads=4,
              max_request_body_size=settings.max_upload_bytes + 2 * 1024 * 1024)
    except ImportError:
        log.warning("waitress is not installed; using Flask's development server.")
        app.run(host=settings.host, port=settings.port, debug=False)


if __name__ == "__main__":
    main()

"""Application and model configuration.

Everything tunable comes from environment variables (optionally loaded from a
``.env`` file) or from ``models/model_config.json``. No value in here is a
prediction or an accuracy claim.
"""
from __future__ import annotations

import json
import os
import secrets
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

BASE_DIR = Path(__file__).resolve().parent

try:  # python-dotenv is optional
    from dotenv import load_dotenv

    load_dotenv(BASE_DIR / ".env")
except Exception:  # pragma: no cover - missing package is fine
    pass


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _resolve(path_like: str) -> Path:
    p = Path(path_like)
    return p if p.is_absolute() else (BASE_DIR / p)


@dataclass
class ModelSpec:
    """Everything needed to download, load and run the pretrained model."""

    name: str
    display_name: str
    architecture: str
    repo_id: str
    filename: str
    revision: str
    page: str
    expected_size_bytes: int
    sha256: str
    license_note: str
    classes: List[str]
    violent_index: int
    num_frames: int
    frame_size: int
    mean: Tuple[float, float, float]
    std: Tuple[float, float, float]
    scale: float
    threshold: float
    aggregation: str
    topk: int
    min_consecutive: int
    segment_seconds: float
    segment_overlap: float
    single_segment_up_to_seconds: float
    max_segments: int
    reported_metrics: Dict[str, Any] = field(default_factory=dict)

    @property
    def local_filename(self) -> str:
        return Path(self.filename).name


VALID_AGGREGATIONS = ("max", "mean", "topk", "consecutive")


def load_model_spec(path: Path, env: Optional[Dict[str, str]] = None) -> ModelSpec:
    env = os.environ if env is None else env
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    pre, dec, seg, src = raw["preprocessing"], raw["decision"], raw["segmentation"], raw["source"]

    threshold = dec["threshold"]
    if env.get("VIOLENCE_THRESHOLD"):
        threshold = float(env["VIOLENCE_THRESHOLD"])
    aggregation = (env.get("AGGREGATION") or dec["aggregation"]).strip().lower()
    if aggregation not in VALID_AGGREGATIONS:
        raise ValueError(f"AGGREGATION must be one of {VALID_AGGREGATIONS}, got {aggregation!r}")
    if not 0.0 < threshold < 1.0:
        raise ValueError(f"Threshold must be in (0, 1), got {threshold}")

    return ModelSpec(
        name=raw["name"],
        display_name=raw.get("display_name", raw["name"]),
        architecture=raw["architecture"],
        repo_id=src["repo_id"],
        filename=src["filename"],
        revision=src.get("revision", "main"),
        page=src.get("page", ""),
        expected_size_bytes=int(src["expected_size_bytes"]),
        sha256=src["sha256"].lower(),
        license_note=raw.get("license_note", ""),
        classes=list(raw["classes"]),
        violent_index=int(raw["violent_index"]),
        num_frames=int(pre["num_frames"]),
        frame_size=int(pre["frame_size"]),
        mean=tuple(pre["mean"]),  # type: ignore[arg-type]
        std=tuple(pre["std"]),  # type: ignore[arg-type]
        scale=float(pre["scale"]),
        threshold=float(threshold),
        aggregation=aggregation,
        topk=int(dec.get("topk", 2)),
        min_consecutive=int(dec.get("min_consecutive", 1)),
        segment_seconds=float(seg["segment_seconds"]),
        segment_overlap=float(seg["segment_overlap"]),
        single_segment_up_to_seconds=float(seg["single_segment_up_to_seconds"]),
        max_segments=int(seg["max_segments"]),
        reported_metrics=raw.get("reported_metrics", {}),
    )


@dataclass
class Settings:
    host: str = "127.0.0.1"
    port: int = 5000
    debug: bool = False
    secret_key: str = field(default_factory=lambda: secrets.token_hex(16))
    max_upload_mb: int = 200
    allowed_extensions: Tuple[str, ...] = ("mp4", "avi", "mov")
    upload_dir: Path = BASE_DIR / "uploads"
    output_dir: Path = BASE_DIR / "outputs"
    model_dir: Path = BASE_DIR / "models"
    model_config_path: Path = BASE_DIR / "models" / "model_config.json"
    log_dir: Path = BASE_DIR / "logs"
    log_level: str = "INFO"
    device: str = "auto"  # auto | cpu | cuda
    torch_threads: int = 0  # 0 = library default
    inference_batch_size: int = 4
    max_video_seconds: int = 1800
    min_video_frames: int = 8
    job_ttl_seconds: int = 3600
    stale_upload_seconds: int = 3600
    model_preload: bool = True

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb) * 1024 * 1024

    @classmethod
    def from_env(cls) -> "Settings":
        exts = os.environ.get("ALLOWED_EXTENSIONS", "mp4,avi,mov")
        return cls(
            host=os.environ.get("HOST", "127.0.0.1"),
            port=_env_int("PORT", 5000),
            debug=_env_bool("FLASK_DEBUG", False),
            secret_key=os.environ.get("SECRET_KEY") or secrets.token_hex(16),
            max_upload_mb=_env_int("MAX_UPLOAD_MB", 200),
            allowed_extensions=tuple(e.strip().lower().lstrip(".") for e in exts.split(",") if e.strip()),
            upload_dir=_resolve(os.environ.get("UPLOAD_DIR", "uploads")),
            output_dir=_resolve(os.environ.get("OUTPUT_DIR", "outputs")),
            model_dir=_resolve(os.environ.get("MODEL_DIR", "models")),
            model_config_path=_resolve(os.environ.get("MODEL_CONFIG", "models/model_config.json")),
            log_dir=_resolve(os.environ.get("LOG_DIR", "logs")),
            log_level=os.environ.get("LOG_LEVEL", "INFO").upper(),
            device=os.environ.get("DEVICE", "auto").strip().lower(),
            torch_threads=_env_int("TORCH_THREADS", 0),
            inference_batch_size=max(1, _env_int("INFERENCE_BATCH_SIZE", 4)),
            max_video_seconds=_env_int("MAX_VIDEO_SECONDS", 1800),
            min_video_frames=max(1, _env_int("MIN_VIDEO_FRAMES", 8)),
            job_ttl_seconds=_env_int("JOB_TTL_SECONDS", 3600),
            stale_upload_seconds=_env_int("STALE_UPLOAD_SECONDS", 3600),
            model_preload=_env_bool("MODEL_PRELOAD", True),
        )

    def ensure_dirs(self) -> None:
        for d in (self.upload_dir, self.output_dir, self.model_dir, self.log_dir):
            d.mkdir(parents=True, exist_ok=True)

    def weights_path(self, spec: ModelSpec) -> Path:
        return self.model_dir / spec.local_filename

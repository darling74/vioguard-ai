"""Pretrained X3D-M violence classifier (PyTorch / pytorchvideo).

The weights are NOT bundled: ``python download_model.py`` fetches
``final/final_x3d_realtime.pt`` from the Hugging Face repository
``visionlab-ai/school-violence-detection-models`` and verifies its size and SHA-256.

torch is imported lazily so the web app can start and explain what is missing
instead of crashing when the weights or libraries are not installed yet.
"""
from __future__ import annotations

import hashlib
import importlib.util
import logging
import sys
import threading
import types
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from config import ModelSpec, Settings

log = logging.getLogger(__name__)

_WRAPPER_PREFIXES = ("module.", "backbone.", "model.", "net.")


class ModelUnavailableError(RuntimeError):
    """The model cannot be used right now (missing library, weights, or failed load)."""

    def __init__(self, message: str, code: str = "model_unavailable"):
        super().__init__(message)
        self.code = code


# ------------------------------------------------------------ pure checkpoint helpers
def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def unwrap_checkpoint(ckpt: Any) -> Mapping[str, Any]:
    """Find the state dict inside a checkpoint (same candidate keys the model authors use)."""
    if isinstance(ckpt, Mapping):
        for key in ("model", "model_state_dict", "state_dict"):
            inner = ckpt.get(key)
            if isinstance(inner, Mapping):
                return inner
        return ckpt
    raise ModelUnavailableError(
        f"Unexpected checkpoint type {type(ckpt).__name__}; a state dict (mapping) was expected.",
        "bad_checkpoint",
    )


def strip_wrapper_prefixes(state: Mapping[str, Any]) -> Dict[str, Any]:
    """Remove ``module.``/``backbone.`` style prefixes (DataParallel, ViolenceX3D wrapper)."""
    out: Dict[str, Any] = {}
    for key, value in state.items():
        k = key
        changed = True
        while changed:
            changed = False
            for pre in _WRAPPER_PREFIXES:
                if k.startswith(pre):
                    k = k[len(pre):]
                    changed = True
        out[k] = value
    return out


def detect_head_layout(shapes: Mapping[str, Tuple[int, ...]]) -> Tuple[str, int, int]:
    """Work out how the classification head was stored.

    Returns ``(kind, out_features, in_features)`` where kind is ``"linear"`` for
    ``blocks.5.proj.weight`` or ``"sequential"`` for ``blocks.5.proj.1.weight``.
    """
    for key, kind in (("blocks.5.proj.weight", "linear"), ("blocks.5.proj.1.weight", "sequential")):
        shape = shapes.get(key)
        if shape is not None and len(shape) == 2:
            return kind, int(shape[0]), int(shape[1])
    sample = ", ".join(sorted(shapes)[:6])
    raise ModelUnavailableError(
        "Checkpoint does not look like an X3D-M classifier (no 'blocks.5.proj' head found). "
        f"First keys: {sample}",
        "bad_checkpoint",
    )


def probabilities_from_logits(logits: np.ndarray, violent_index: int) -> np.ndarray:
    """(N, C) logits -> (N,) violence probability. C=2 softmax, C=1 sigmoid."""
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim == 1:
        logits = logits[:, None]
    if logits.shape[1] == 1:
        return 1.0 / (1.0 + np.exp(-logits[:, 0]))
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return (e / e.sum(axis=1, keepdims=True))[:, violent_index]


def _install_torchvision_shim() -> None:
    """pytorchvideo 0.1.5 imports torchvision.transforms.functional_tensor, removed in torchvision >= 0.17."""
    try:
        import torchvision.transforms.functional_tensor  # noqa: F401
        return
    except ImportError:
        pass
    import torchvision.transforms.functional as F

    shim = types.ModuleType("torchvision.transforms.functional_tensor")
    shim.rgb_to_grayscale = F.rgb_to_grayscale  # type: ignore[attr-defined]
    sys.modules["torchvision.transforms.functional_tensor"] = shim


# ------------------------------------------------------------------------ the model
class ViolenceModel:
    """Thread-safe, lazily loaded, cached X3D-M violence classifier."""

    def __init__(self, spec: ModelSpec, settings: Settings):
        self.spec = spec
        self.settings = settings
        self.weights_path: Path = settings.weights_path(spec)
        self._lock = threading.RLock()
        self._net = None
        self._torch = None
        self._device = "cpu"
        self._error: Optional[str] = None
        self._loading = False
        self._head: Optional[Tuple[str, int]] = None

    # ----------------------------------------------------------------- status
    @staticmethod
    def _has(module: str) -> bool:
        try:
            return importlib.util.find_spec(module) is not None
        except (ImportError, ValueError):
            return False

    @property
    def is_loaded(self) -> bool:
        return self._net is not None

    def status(self) -> Dict[str, Any]:
        """Cheap status snapshot (never imports torch)."""
        if self.is_loaded:
            state, message = "ready", "Model loaded."
        elif self._loading:
            state, message = "loading", "Model is loading..."
        elif self._error:
            state, message = "error", self._error
        elif not (self._has("torch") and self._has("pytorchvideo")):
            state = "dependency_missing"
            message = "PyTorch / pytorchvideo are not installed. Run: pip install -r requirements.txt"
        elif not self.weights_path.exists():
            state = "weights_missing"
            message = "Model weights not found. Run: python download_model.py"
        else:
            state, message = "not_loaded", "Weights found; model will load on first use."
        return {
            "state": state,
            "message": message,
            "available": state in {"ready", "not_loaded", "loading"},
            "device": self._device if self.is_loaded else self.settings.device,
            "name": self.spec.display_name,
            "architecture": self.spec.architecture,
            "threshold": self.spec.threshold,
            "weights_file": str(self.weights_path.name),
        }

    # ---------------------------------------------------------------- loading
    def _resolve_device(self, torch) -> str:
        want = self.settings.device
        if want == "cuda" and not torch.cuda.is_available():
            log.warning("DEVICE=cuda requested but CUDA is unavailable; using CPU.")
            return "cpu"
        if want == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return want if want in {"cpu", "cuda"} else "cpu"

    def _verify_file(self) -> None:
        size = self.weights_path.stat().st_size
        if self.spec.expected_size_bytes and size != self.spec.expected_size_bytes:
            raise ModelUnavailableError(
                f"{self.weights_path.name} has {size} bytes, expected {self.spec.expected_size_bytes}. "
                "The download is incomplete or the wrong file. Delete it and run: python download_model.py",
                "bad_checkpoint",
            )
        digest = sha256_file(self.weights_path)
        if self.spec.sha256 and digest != self.spec.sha256:
            raise ModelUnavailableError(
                f"SHA-256 mismatch for {self.weights_path.name} (got {digest[:12]}..., "
                f"expected {self.spec.sha256[:12]}...). Delete the file and run: python download_model.py",
                "bad_checkpoint",
            )

    def load(self) -> None:
        """Load the network once; later calls are no-ops."""
        with self._lock:
            if self._net is not None:
                return
            self._loading, self._error = True, None
            try:
                self._load_locked()
            except ModelUnavailableError as exc:
                self._error = str(exc)
                raise
            except Exception as exc:  # unexpected: keep the app alive, report clearly
                self._error = f"Model failed to load: {exc}"
                log.exception("Model load failed")
                raise ModelUnavailableError(self._error, "load_failed") from exc
            finally:
                self._loading = False

    def _load_locked(self) -> None:
        if not self._has("torch"):
            raise ModelUnavailableError("PyTorch is not installed. Run: pip install -r requirements.txt", "dependency_missing")
        if not self._has("pytorchvideo"):
            raise ModelUnavailableError("pytorchvideo is not installed. Run: pip install -r requirements.txt", "dependency_missing")
        if not self.weights_path.exists():
            raise ModelUnavailableError(
                f"Model weights not found at {self.weights_path}. Run: python download_model.py", "weights_missing")

        import torch

        _install_torchvision_shim()
        from pytorchvideo.models.hub import x3d_m

        self._verify_file()  # size + SHA-256 of the pinned official file
        if self.settings.torch_threads > 0:
            torch.set_num_threads(self.settings.torch_threads)
        device = self._resolve_device(torch)

        # The checkpoint is verified against the pinned SHA-256 before loading.
        # First try the restricted loader while allowlisting the NumPy scalar type
        # reported by this checkpoint. Only if that fails, use full pickle loading
        # for this verified, trusted checkpoint.
        try:
            try:
                numpy_scalar = np._core.multiarray.scalar  # NumPy 2.x
            except AttributeError:
                numpy_scalar = np.core.multiarray.scalar  # NumPy 1.x
            with torch.serialization.safe_globals([numpy_scalar]):
                ckpt = torch.load(self.weights_path, map_location="cpu", weights_only=True)
        except Exception as safe_exc:
            log.warning("Restricted checkpoint load failed (%s); retrying full load of SHA-256-verified checkpoint", safe_exc)
            ckpt = torch.load(self.weights_path, map_location="cpu", weights_only=False)

        state = strip_wrapper_prefixes(unwrap_checkpoint(ckpt))
        shapes = {k: tuple(v.shape) for k, v in state.items() if hasattr(v, "shape")}
        kind, out_features, in_features = detect_head_layout(shapes)

        net = x3d_m(pretrained=False)
        head = net.blocks[5]
        linear = torch.nn.Linear(in_features, out_features)
        head.proj = linear if kind == "linear" else torch.nn.Sequential(torch.nn.Dropout(0.3), linear)
        head.activation = None  # always return raw logits; softmax is applied exactly once below

        result = net.load_state_dict(state, strict=False)
        if result.missing_keys:
            raise ModelUnavailableError(
                f"Checkpoint is missing {len(result.missing_keys)} weights required by X3D-M "
                f"(e.g. {result.missing_keys[:3]}). Architecture mismatch.", "bad_checkpoint")
        if result.unexpected_keys:
            log.warning("Ignoring %d unexpected checkpoint keys (e.g. %s)",
                        len(result.unexpected_keys), result.unexpected_keys[:3])

        net.eval().to(device)
        with torch.inference_mode():  # smoke test: shape + finiteness
            dummy = torch.zeros(1, 3, self.spec.num_frames, self.spec.frame_size, self.spec.frame_size, device=device)
            out = net(dummy)
            out = out.reshape(out.shape[0], -1)  # some pytorchvideo heads keep singleton dims
        if out.ndim != 2 or out.shape[1] != out_features or not bool(torch.isfinite(out).all()):
            raise ModelUnavailableError(f"Unexpected smoke-test output shape {tuple(out.shape)}.", "bad_checkpoint")

        self._torch, self._net, self._device, self._head = torch, net, device, (kind, out_features)
        log.info("Loaded %s on %s (head=%s, outputs=%d)", self.spec.name, device, kind, out_features)

    # -------------------------------------------------------------- inference
    def preprocess(self, clips: np.ndarray):
        """uint8 RGB (N,T,H,W,3) -> normalised float tensor (N,3,T,H,W) on the model device."""
        torch = self._torch
        x = torch.from_numpy(np.ascontiguousarray(clips)).to(self._device).float() / self.spec.scale
        mean = torch.tensor(self.spec.mean, device=self._device).view(1, 1, 1, 1, 3)
        std = torch.tensor(self.spec.std, device=self._device).view(1, 1, 1, 1, 3)
        return ((x - mean) / std).permute(0, 4, 1, 2, 3).contiguous()

    def predict_clips(self, clips: np.ndarray) -> np.ndarray:
        """Violence probability for each clip. ``clips``: uint8 RGB, shape (N, T, 224, 224, 3)."""
        expected = (self.spec.num_frames, self.spec.frame_size, self.spec.frame_size, 3)
        if clips.ndim != 5 or tuple(clips.shape[1:]) != expected:
            raise ValueError(f"clips must have shape (N,{expected}), got {tuple(clips.shape)}")
        self.load()
        with self._lock:
            torch = self._torch
            with torch.inference_mode():
                out = self._net(self.preprocess(clips))
                logits = out.reshape(out.shape[0], -1).float().cpu().numpy()
        return probabilities_from_logits(logits, self.spec.violent_index)


_MODEL_CACHE: Dict[str, ViolenceModel] = {}
_CACHE_LOCK = threading.Lock()


def get_model(spec: ModelSpec, settings: Settings) -> ViolenceModel:
    """Process-wide cached model instance (never reloaded per request)."""
    key = f"{spec.name}:{settings.weights_path(spec)}"
    with _CACHE_LOCK:
        if key not in _MODEL_CACHE:
            _MODEL_CACHE[key] = ViolenceModel(spec, settings)
        return _MODEL_CACHE[key]
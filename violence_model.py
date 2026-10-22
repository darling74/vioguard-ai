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

"""Diagnose your installation:  python check_setup.py

Checks Python, libraries, the model weights (size + SHA-256), loads the real
model and runs one forward pass on a blank clip to prove the whole stack works.
(A blank clip only proves the network runs; it says nothing about accuracy.)
"""
from __future__ import annotations

import importlib
import platform
import sys

import numpy as np

from config import Settings, load_model_spec

OK, BAD = "[ OK ]", "[FAIL]"


def check_import(name: str, required: bool = True) -> bool:
    try:
        importlib.import_module(name)
        try:
            from importlib.metadata import version

            ver = version({"cv2": "opencv-python-headless"}.get(name, name))
        except Exception:
            ver = ""
        print(f"{OK} {name} {ver}")
        return True
    except Exception as exc:
        print(f"{BAD if required else '[warn]'} {name}: {type(exc).__name__}: {exc}")
        return not required


def main() -> int:
    print(f"Python {platform.python_version()} on {platform.system()} {platform.machine()}")
    if sys.version_info < (3, 10):
        print(f"{BAD} Python 3.10+ is required (3.12 recommended)")
        return 1
    results = [check_import(n) for n in ("flask", "numpy", "cv2", "torch", "torchvision", "pytorchvideo")]
    check_import("waitress", required=False)  # optional: falls back to Flask's built-in server
    ok = all(results)
    settings = Settings.from_env()
    spec = load_model_spec(settings.model_config_path)
    from violence_model import ModelUnavailableError, get_model

    model = get_model(spec, settings)
    st = model.status()
    bad_state = st["state"] in {"weights_missing", "dependency_missing", "error"}
    print(f"{BAD if bad_state else OK} model: {st['state']} - {st['message']}")
    if not ok or st["state"] in {"weights_missing", "dependency_missing"}:
        print("\nFix the items marked [FAIL] (see README, 'Fixing common errors') and run this again.")
        return 1
    try:
        model.load()
        import torch

        probs = model.predict_clips(np.zeros((1, spec.num_frames, spec.frame_size, spec.frame_size, 3), np.uint8))
        print(f"{OK} model loaded on {model.status()['device']}; forward pass on a blank clip -> P(violence)={probs[0]:.4f}")
        print(f"{OK} torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
    except ModelUnavailableError as exc:
        print(f"{BAD} {exc}")
        return 1
    print("\nAll checks passed. Start the app with:  python app.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())

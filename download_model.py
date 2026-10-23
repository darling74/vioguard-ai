"""Download and verify the pretrained violence-detection weights.

    python download_model.py            # download if missing, verify size + SHA-256
    python download_model.py --force    # re-download
    python download_model.py --verify   # only verify an existing file

Source (official): https://huggingface.co/visionlab-ai/school-violence-detection-models
File: final/final_x3d_realtime.pt  (~12 MB, X3D-M, 16 frames, threshold 0.4)
Only the Python standard library is needed. Set HF_ENDPOINT to use a mirror and
HF_TOKEN if your network requires an authenticated Hugging Face session.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

from config import Settings, load_model_spec


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify(path: Path, expected_size: int, expected_sha: str) -> str | None:
    """Return None when the file is good, otherwise a human-readable problem."""
    if not path.exists():
        return f"{path} does not exist."
    size = path.stat().st_size
    if expected_size and size != expected_size:
        return f"size is {size} bytes, expected {expected_size}."
    digest = sha256_of(path)
    if expected_sha and digest != expected_sha:
        return f"SHA-256 is {digest}, expected {expected_sha}."
    return None


def build_url(repo_id: str, revision: str, filename: str) -> str:
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    return f"{endpoint}/{repo_id}/resolve/{revision}/{filename}?download=true"


def download(url: str, dest: Path, expected_size: int) -> None:
    headers = {"User-Agent": "ai-violence-detection-downloader/1.0"}
    token = os.environ.get("HF_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    part = dest.with_suffix(dest.suffix + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as resp, open(part, "wb") as out:
        total = int(resp.headers.get("Content-Length") or expected_size or 0)
        done = 0
        while True:
            block = resp.read(1 << 18)
            if not block:
                break
            out.write(block)
            done += len(block)
            if total:
                print(f"\r  downloading {done / 1e6:6.1f} / {total / 1e6:.1f} MB", end="", flush=True)
    print()
    os.replace(part, dest)  # atomic: a half-finished download is never mistaken for the model


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="download again even if the file exists")
    ap.add_argument("--verify", action="store_true", help="only verify the existing file")
    args = ap.parse_args(argv)

    settings = Settings.from_env()
    spec = load_model_spec(settings.model_config_path)
    dest = settings.weights_path(spec)
    url = build_url(spec.repo_id, spec.revision, spec.filename)

    print(f"Model : {spec.display_name}")
    print(f"Source: {spec.page}")
    print(f"License note: {spec.license_note}")
    print(f"Target: {dest}")

    if dest.exists() and not args.force:
        problem = verify(dest, spec.expected_size_bytes, spec.sha256)
        if problem is None:
            print("OK: weights already present and verified (size + SHA-256).")
            return 0
        print(f"Existing file is invalid: {problem}")
        if args.verify:
            return 1
    elif args.verify:
        print("Weights are not downloaded yet.")
        return 1

    print(f"Downloading from {url}")
    try:
        download(url, dest, spec.expected_size_bytes)
    except urllib.error.HTTPError as exc:
        hint = {401: "Authentication required: set HF_TOKEN.", 403: "Access denied: your network or proxy may block huggingface.co.",
                404: "File not found: the repository layout may have changed."}.get(exc.code, "")
        print(f"ERROR: HTTP {exc.code} {exc.reason}. {hint}", file=sys.stderr)
        return 2
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        print(f"ERROR: could not download ({exc}). Check your internet connection, firewall or proxy "
              f"(HTTPS_PROXY), or set HF_ENDPOINT to a mirror.", file=sys.stderr)
        return 2

    problem = verify(dest, spec.expected_size_bytes, spec.sha256)
    if problem:
        print(f"ERROR: downloaded file failed verification: {problem}", file=sys.stderr)
        dest.unlink(missing_ok=True)
        return 3
    print("OK: downloaded and verified (size + SHA-256).")
    return 0


if __name__ == "__main__":
    sys.exit(main())

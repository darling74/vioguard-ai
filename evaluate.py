"""Honest accuracy evaluation of the pretrained model on YOUR labelled videos.

    python evaluate.py --data path/to/dataset

The dataset folder needs one sub-folder per class, e.g. ``Violence/`` and
``NonViolence/`` (names such as Fight/NonFight, violent/non_violent also work).

Protocol (designed to avoid leakage):
  1. Videos are split per class into a VALIDATION part and a disjoint TEST part
     (``--val-fraction``, seeded). A video is never in both.
  2. Aggregation mode and decision threshold are chosen on VALIDATION only.
  3. Reported results are measured on TEST only, plus the model authors' fixed
     setting (max aggregation, threshold 0.4) for reference.
  4. Window probabilities are cached, so sweeps never re-run the network.

IMPORTANT: the pretrained model was trained on RWF-2000. Evaluating on RWF-2000
clips is not independent. Use a different dataset (for example the Kaggle
"Real Life Violence Situations" dataset) and note that near-duplicate clips
from one source video can still inflate scores; this script cannot detect that.
Nothing here fabricates numbers: every figure is computed from your videos.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from video_processor import aggregate_scores

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
AGG_MODES = ("max", "topk", "mean", "consecutive")


# ------------------------------------------------------------------ dataset handling
def label_from_folder(name: str):
    n = re.sub(r"[^a-z0-9]", "", name.lower())
    if n.startswith("non") or n.startswith("no") or n in {"normal", "nv", "safe", "peace", "peaceful"}:
        return 0
    if "violen" in n or "fight" in n or n in {"v", "abuse", "assault"}:
        return 1
    return None


def discover_videos(root: Path) -> List[Tuple[Path, int]]:
    items: List[Tuple[Path, int]] = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            label = None
            for parent in p.relative_to(root).parents:  # nearest folder with a recognisable name first
                if parent.name:
                    label = label_from_folder(parent.name)
                    if label is not None:
                        break
            if label is not None:
                items.append((p, label))
    return items


def stratified_split(items: Sequence[Tuple[Path, int]], val_fraction: float, seed: int):
    """Disjoint per-class validation/test split. No item can land in both."""
    rng = random.Random(seed)
    val, test = [], []
    for cls in (0, 1):
        group = [it for it in items if it[1] == cls]
        rng.shuffle(group)
        n_val = int(round(len(group) * val_fraction))
        val.extend(group[:n_val])
        test.extend(group[n_val:])
    return val, test


# ------------------------------------------------------------------------- metrics
def confusion(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, int]:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def wilson_interval(successes: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def metrics_from_counts(c: Dict[str, int]) -> Dict[str, float]:
    tp, tn, fp, fn = c["tp"], c["tn"], c["fp"], c["fn"]
    n = tp + tn + fp + fn
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    lo, hi = wilson_interval(tp + tn, n)
    return {
        "n": n, "accuracy": (tp + tn) / n if n else 0.0, "precision": prec, "recall": rec, "f1": f1,
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
        "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
        "false_negative_rate": fn / (fn + tp) if fn + tp else 0.0,
        "accuracy_ci95_low": lo, "accuracy_ci95_high": hi, **c,
    }


def video_scores(records: Sequence[Dict], mode: str, threshold: float, topk: int, min_consec: int) -> List[float]:
    return [aggregate_scores(r["probs"], mode, threshold, topk, min_consec)[0] for r in records]


def evaluate_setting(records: Sequence[Dict], mode: str, threshold: float, topk: int = 2, min_consec: int = 2):
    y = [r["label"] for r in records]
    scores = video_scores(records, mode, threshold, topk, min_consec)
    pred = [int(s >= threshold) for s in scores]
    return metrics_from_counts(confusion(y, pred))


def choose_setting(records: Sequence[Dict], default_threshold: float, topk: int = 2, min_consec: int = 2):
    """Pick (mode, threshold) by validation F1; ties prefer the authors' threshold. Validation data only."""
    y = [r["label"] for r in records]
    best = None
    for mode in AGG_MODES:
        scores = video_scores(records, mode, 0.5, topk, min_consec)
        candidates = sorted({round(t, 3) for t in np.arange(0.05, 0.96, 0.05)} | {default_threshold})
        for t in candidates:
            m = metrics_from_counts(confusion(y, [int(s >= t) for s in scores]))
            key = (round(m["f1"], 6), round(m["accuracy"], 6), -abs(t - default_threshold))
            if best is None or key > best[0]:
                best = (key, mode, float(t), m)
    return {"mode": best[1], "threshold": best[2], "validation_metrics": best[3]}


# ---------------------------------------------------------------------- inference
def run_inference(items, cache_path: Path, limit: int | None, spec, model, min_frames: int, max_seconds, batch_size: int):
    from video_processor import extract_segment_clips, plan_segments, probe_video, VideoDecodeError

    cache: Dict[str, Dict] = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
    todo = [it for it in items if str(it[0]) not in cache]
    if limit:
        todo = todo[:limit]
    t0 = time.time()
    for i, (path, label) in enumerate(todo, 1):
        try:
            info = probe_video(path, min_frames, max_seconds)
            segs = plan_segments(info, spec)
            clips = extract_segment_clips(path, segs, spec.frame_size)
            probs: List[float] = []
            for s in range(0, len(segs), batch_size):
                probs.extend(float(x) for x in model.predict_clips(clips[s:s + batch_size]))
            cache[str(path)] = {"label": label, "probs": probs, "duration": info.duration}
        except VideoDecodeError as exc:
            cache[str(path)] = {"label": label, "error": str(exc)}
        if i % 10 == 0 or i == len(todo):
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(cache), encoding="utf-8")
            print(f"  {i}/{len(todo)} videos  ({time.time() - t0:.0f}s)", flush=True)
    return cache


# ------------------------------------------------------------------------ report
def fmt(m: Dict[str, float]) -> str:
    return (f"accuracy {m['accuracy']:.2%} (95% CI {m['accuracy_ci95_low']:.1%}-{m['accuracy_ci95_high']:.1%}), "
            f"precision {m['precision']:.2%}, recall {m['recall']:.2%}, F1 {m['f1']:.3f}, "
            f"FP {m['fp']}, FN {m['fn']} (n={m['n']})")


def build_report(meta: Dict, chosen: Dict, test_tuned: Dict, test_default: Dict, val_default: Dict) -> str:
    target_met = test_tuned["accuracy"] >= 0.95
    lines = [
        "# Accuracy evaluation report", "",
        f"*Generated {time.strftime('%Y-%m-%d %H:%M:%S')} by `evaluate.py`. All numbers are computed from the videos listed below.*", "",
        "## Setup",
        f"- Dataset folder: `{meta['data']}`",
        f"- Videos found: {meta['found']} (violent {meta['n_violent']}, non-violent {meta['n_nonviolent']}); "
        f"undecodable/skipped: {meta['skipped']}",
        f"- Split (seed {meta['seed']}): validation {meta['n_val']} videos, test {meta['n_test']} videos; disjoint by construction",
        f"- Model: {meta['model']}", "",
        "## Settings chosen on the validation split only",
        f"- Aggregation: `{chosen['mode']}`, threshold: {chosen['threshold']:.2f}",
        f"- Validation: {fmt(chosen['validation_metrics'])}", "",
        "## Held-out TEST results",
        f"**Tuned settings** ({chosen['mode']}, thr {chosen['threshold']:.2f}): {fmt(test_tuned)}", "",
        f"**Authors' fixed settings** (max, thr 0.40): {fmt(test_default)}", "",
        "Confusion matrix (tuned settings, test):", "",
        "| | Pred non-violent | Pred violent |", "|---|---|---|",
        f"| **True non-violent** | {test_tuned['tn']} | {test_tuned['fp']} |",
        f"| **True violent** | {test_tuned['fn']} | {test_tuned['tp']} |", "",
        "## Verdict",
        f"- Point-estimate accuracy on held-out test: **{test_tuned['accuracy']:.2%}** -> "
        f"{'meets' if target_met else 'does NOT meet'} the 95% target.",
        f"- 95% confidence interval: {test_tuned['accuracy_ci95_low']:.1%} to {test_tuned['accuracy_ci95_high']:.1%}"
        + (" (lower bound is below 95%: the evidence is not conclusive)" if target_met and test_tuned['accuracy_ci95_low'] < 0.95 else ""),
        "", "## Caveats",
        "- The pretrained model was trained on RWF-2000; results on RWF-2000 are not independent.",
        "- Near-duplicate clips from the same source video can leak across splits and inflate results; this script cannot detect them.",
        "- Accuracy on this dataset does not transfer automatically to other cameras, lighting or scenes.",
    ]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--val-fraction", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None, help="process at most N new videos (for a quick trial)")
    ap.add_argument("--cache", type=Path, default=Path("outputs/eval_cache.json"))
    ap.add_argument("--report", type=Path, default=Path("outputs/evaluation_report.md"))
    args = ap.parse_args(argv)

    from config import Settings, load_model_spec
    from violence_model import ModelUnavailableError, get_model

    settings = Settings.from_env()
    spec = load_model_spec(settings.model_config_path)
    items = discover_videos(args.data)
    if not items:
        print(f"No labelled videos found under {args.data}. Expected sub-folders like Violence/ and NonViolence/.", file=sys.stderr)
        return 1
    print(f"Found {len(items)} labelled videos.")
    model = get_model(spec, settings)
    try:
        model.load()
    except ModelUnavailableError as exc:
        print(f"Model unavailable: {exc}", file=sys.stderr)
        return 2
    cache = run_inference(items, args.cache, args.limit, spec, model, settings.min_video_frames,
                          settings.max_video_seconds, settings.inference_batch_size)

    usable = {str(p): lab for p, lab in items if "probs" in cache.get(str(p), {})}
    skipped = len(items) - len(usable)
    items = [(p, l) for p, l in items if str(p) in usable]
    val_items, test_items = stratified_split(items, args.val_fraction, args.seed)
    assert not ({str(p) for p, _ in val_items} & {str(p) for p, _ in test_items}), "split leakage"
    val = [cache[str(p)] for p, _ in val_items]
    test = [cache[str(p)] for p, _ in test_items]
    if not val or not test:
        print("Not enough processed videos for a validation/test split. Process more videos.", file=sys.stderr)
        return 1

    chosen = choose_setting(val, spec.threshold, spec.topk)
    tuned = evaluate_setting(test, chosen["mode"], chosen["threshold"], spec.topk)
    default = evaluate_setting(test, "max", spec.threshold, spec.topk)
    val_default = evaluate_setting(val, "max", spec.threshold, spec.topk)
    meta = {"data": str(args.data), "found": len(items) + skipped, "skipped": skipped, "seed": args.seed,
            "n_violent": sum(1 for _, l in items if l == 1), "n_nonviolent": sum(1 for _, l in items if l == 0),
            "n_val": len(val), "n_test": len(test), "model": spec.display_name}
    report = build_report(meta, chosen, tuned, default, val_default)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")
    args.report.with_suffix(".json").write_text(json.dumps(
        {"meta": meta, "chosen": chosen, "test_tuned": tuned, "test_default": default}, indent=2), encoding="utf-8")
    print(report)
    print(f"Report written to {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Video decoding, segment planning, frame sampling and score aggregation.

The sampling mirrors the way the pretrained X3D-M checkpoint is used by its
authors: 16 frames picked with ``np.linspace`` over a clip, resized (squashed)
to 224x224 with bilinear interpolation and converted to RGB. Longer videos are
cut into ~5 s windows (the length of the RWF-2000 training clips) so each
window looks like a training sample; window scores are then aggregated.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from config import ModelSpec

log = logging.getLogger(__name__)

ProgressCb = Optional[Callable[[str, float, Dict], None]]


class VideoDecodeError(Exception):
    """The file is not a readable video (corrupted, empty, unsupported codec...)."""


class VideoTooLongError(VideoDecodeError):
    """Video longer than the configured limit."""


@dataclass
class VideoInfo:
    frame_count: int
    fps: float
    width: int
    height: int

    @property
    def duration(self) -> float:
        return self.frame_count / self.fps if self.fps > 0 else 0.0

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["duration_sec"] = round(self.duration, 3)
        return d


@dataclass
class Segment:
    index: int
    start_frame: int
    end_frame: int  # exclusive
    frame_indices: List[int]

    def times(self, fps: float) -> Tuple[float, float]:
        return self.start_frame / fps, self.end_frame / fps


# --------------------------------------------------------------------------- probing
def probe_video(path: str | Path, min_frames: int = 8, max_seconds: Optional[float] = None) -> VideoInfo:
    """Open a video, check that frames really decode and return its metadata.

    Container metadata can lie (frame count 0 or wrong), so the first frame is
    decoded and, when the count is missing, the video is counted by decoding.
    """
    path = str(path)
    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            raise VideoDecodeError("The file could not be opened as a video (corrupted or unsupported codec).")
        ok, first = cap.read()
        if not ok or first is None or first.size == 0:
            raise VideoDecodeError("No frame could be decoded from this file (corrupted or empty video).")
        height, width = first.shape[:2]
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if not np.isfinite(fps) or fps < 1.0 or fps > 240.0:
            fps = 30.0
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if count <= 0 or count > 10_000_000:
            count = 1
            while cap.grab():
                count += 1
    finally:
        cap.release()

    info = VideoInfo(frame_count=count, fps=fps, width=int(width), height=int(height))
    if info.frame_count < min_frames:
        raise VideoDecodeError(
            f"The video has only {info.frame_count} frame(s); at least {min_frames} are required."
        )
    if max_seconds and info.duration > max_seconds:
        raise VideoTooLongError(
            f"The video is {info.duration:.0f} s long; the configured maximum is {max_seconds:.0f} s."
        )
    return info


# ------------------------------------------------------------------- segment planning
def sample_indices(start: int, end: int, num_frames: int) -> List[int]:
    """Uniformly spaced frame indices in [start, end) - same rule as the model authors."""
    last = max(end - 1, start)
    return np.linspace(start, last, num=num_frames).astype(int).tolist()


def plan_segments(info: VideoInfo, spec: ModelSpec) -> List[Segment]:
    """Cut the video into analysis windows.

    * up to ``single_segment_up_to_seconds``: one window covering the whole video
      (this is exactly how a training clip is consumed);
    * longer: ``segment_seconds`` windows with ``segment_overlap`` overlap, the
      last window always touching the final frame, capped at ``max_segments``.
    """
    total, fps = info.frame_count, info.fps
    if info.duration <= spec.single_segment_up_to_seconds:
        return [Segment(0, 0, total, sample_indices(0, total, spec.num_frames))]

    win = max(int(round(spec.segment_seconds * fps)), spec.num_frames)
    win = min(win, total)
    stride = max(int(round(win * (1.0 - spec.segment_overlap))), 1)
    starts = list(range(0, total - win + 1, stride))
    if not starts or starts[-1] != total - win:
        starts.append(total - win)
    if len(starts) > spec.max_segments:
        starts = np.unique(np.linspace(0, total - win, spec.max_segments).astype(int)).tolist()
    return [
        Segment(i, s, s + win, sample_indices(s, s + win, spec.num_frames))
        for i, s in enumerate(starts)
    ]


# -------------------------------------------------------------------- frame extraction
def _prepare_frame(frame_bgr: np.ndarray, size: int) -> np.ndarray:
    frame = cv2.resize(frame_bgr, (size, size), interpolation=cv2.INTER_LINEAR)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def extract_segment_clips(
    path: str | Path,
    segments: Sequence[Segment],
    frame_size: int,
    progress: ProgressCb = None,
) -> np.ndarray:
    """Decode the video once and return clips of shape (N, T, H, W, 3), uint8 RGB.

    Every needed frame is decoded sequentially (seeking is unreliable for many
    codecs) and resized immediately, so memory stays small even for 1080p input.
    If the stream ends earlier than its metadata promised, the last decoded
    frame stands in for the missing ones.
    """
    needed = sorted({i for seg in segments for i in seg.frame_indices})
    if not needed:
        raise VideoDecodeError("No frames selected for analysis.")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise VideoDecodeError("The file could not be opened as a video.")

    cache: Dict[int, np.ndarray] = {}
    last_good: Optional[np.ndarray] = None
    pos = 0  # index of the next frame the decoder will return
    last_needed = needed[-1]
    ended_early = False
    try:
        for target in needed:
            while pos < target:
                if not cap.grab():
                    ended_early = True
                    break
                pos += 1
            if ended_early:
                break
            ok, frame = cap.read()
            if not ok or frame is None:
                ended_early = True
                break
            pos += 1
            last_good = _prepare_frame(frame, frame_size)
            cache[target] = last_good
            if progress and (len(cache) % 8 == 0 or target == last_needed):
                progress("decoding", min(target / max(last_needed, 1), 1.0), {})
    finally:
        cap.release()

    if last_good is None:
        raise VideoDecodeError("No frame could be decoded from this file.")
    if ended_early:
        missing = sum(1 for i in needed if i not in cache)
        log.warning("Video ended before metadata frame count; substituting %d frame(s)", missing)
    fallback = last_good

    clips = np.empty((len(segments), len(segments[0].frame_indices), frame_size, frame_size, 3), dtype=np.uint8)
    for si, seg in enumerate(segments):
        prev = None
        for ti, fi in enumerate(seg.frame_indices):
            frame = cache.get(fi)
            if frame is None:
                frame = prev if prev is not None else fallback
            clips[si, ti] = frame
            prev = frame
    if progress:
        progress("decoding", 1.0, {})
    return clips


# --------------------------------------------------------------------------- aggregation
def aggregate_scores(
    probs: Sequence[float],
    mode: str,
    threshold: float,
    topk: int = 2,
    min_consecutive: int = 1,
) -> Tuple[float, bool]:
    """Combine per-window violence probabilities into one video-level score.

    * ``max``: highest window (the model authors' rule; best recall, more false alarms on long clips)
    * ``mean``: average (most conservative)
    * ``topk``: mean of the k highest windows
    * ``consecutive``: needs ``min_consecutive`` adjacent windows above threshold; the score is the
      best (over positions) minimum within a run of that length, so it is threshold-comparable.
    The video is violent when ``score >= threshold``.
    """
    p = np.asarray(list(probs), dtype=np.float64)
    if p.size == 0:
        raise ValueError("No window probabilities to aggregate.")
    if mode == "max":
        score = float(p.max())
    elif mode == "mean":
        score = float(p.mean())
    elif mode == "topk":
        k = max(1, min(int(topk), p.size))
        score = float(np.sort(p)[-k:].mean())
    elif mode == "consecutive":
        r = max(1, min(int(min_consecutive), p.size))
        score = float(max(p[i : i + r].min() for i in range(p.size - r + 1)))
    else:
        raise ValueError(f"Unknown aggregation mode: {mode!r}")
    return score, bool(score >= threshold)


def certainty_label(score: float, threshold: float) -> str:
    """Human wording for how far the score is from the decision threshold."""
    margin = abs(score - threshold) / max(threshold if score < threshold else 1.0 - threshold, 1e-6)
    if margin < 0.25:
        return "borderline"
    if margin < 0.6:
        return "moderate"
    return "strong"


# --------------------------------------------------------------------------- orchestrator
class VideoAnalyzer:
    """Run the whole pipeline for one video file using a (shared) model object."""

    def __init__(self, model, spec: ModelSpec, min_frames: int = 8, max_seconds: Optional[float] = None,
                 batch_size: int = 4):
        self.model = model
        self.spec = spec
        self.min_frames = min_frames
        self.max_seconds = max_seconds
        self.batch_size = max(1, batch_size)

    def analyze(self, path: str | Path, progress: ProgressCb = None) -> Dict:
        t0 = time.perf_counter()
        spec = self.spec

        def emit(stage: str, frac: float, extra: Optional[Dict] = None) -> None:
            if progress:
                progress(stage, frac, extra or {})

        emit("probing", 0.0)
        info = probe_video(path, self.min_frames, self.max_seconds)
        segments = plan_segments(info, spec)
        emit("decoding", 0.0, {"segments_total": len(segments)})
        clips = extract_segment_clips(path, segments, spec.frame_size, progress)

        probs: List[float] = []
        for start in range(0, len(segments), self.batch_size):
            batch = clips[start : start + self.batch_size]
            probs.extend(float(x) for x in self.model.predict_clips(batch))
            emit("inference", len(probs) / len(segments),
                 {"segments_done": len(probs), "segments_total": len(segments)})

        score, violent = aggregate_scores(probs, spec.aggregation, spec.threshold, spec.topk, spec.min_consecutive)
        elapsed = time.perf_counter() - t0

        seg_out = []
        for seg, p in zip(segments, probs):
            s, e = seg.times(info.fps)
            seg_out.append({
                "index": seg.index,
                "start_sec": round(s, 2),
                "end_sec": round(e, 2),
                "violence_probability": round(p, 4),
                "label": "violence" if p >= spec.threshold else "non_violence",
            })
        peak = max(seg_out, key=lambda d: d["violence_probability"])
        certainty = certainty_label(score, spec.threshold)
        pct = score * 100
        if violent:
            explanation = (
                f"The strongest window ({peak['start_sec']}s-{peak['end_sec']}s) scored {pct:.1f}% violence "
                f"probability, at or above the decision threshold of {spec.threshold * 100:.0f}%. "
                f"The result is {certainty}."
            )
        else:
            explanation = (
                f"No window reached the decision threshold of {spec.threshold * 100:.0f}% "
                f"(highest: {pct:.1f}%). The result is {certainty}."
            )
        explanation += (" The score is the model's softmax probability, not a calibrated guarantee; "
                        "published validation F1 for this model is about 0.93, so errors are expected.")
        result = {
            "label": "violence" if violent else "non_violence",
            "verdict": "VIOLENCE DETECTED" if violent else "NON-VIOLENCE DETECTED",
            "prediction": "Violence" if violent else "NonViolence",  # backwards compatible field
            "violence_probability": round(score, 4),
            "confidence": round(score if violent else 1.0 - score, 4),
            "certainty": certainty,
            "threshold": spec.threshold,
            "aggregation": spec.aggregation,
            "segment_count": len(segments),
            "frames_analyzed": len(segments) * spec.num_frames,
            "segments": seg_out,
            "video": info.to_dict(),
            "processing_time_sec": round(elapsed, 3),
            "model": {"name": spec.name, "architecture": spec.architecture, "source": spec.page},
            "explanation": explanation,
        }
        emit("done", 1.0)
        return result

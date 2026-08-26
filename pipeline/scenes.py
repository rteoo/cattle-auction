"""Scene-change detection for content-aware screenshot sampling.

An auction video's screen is a mostly-static "lot board" graphic that only
changes when the lot changes. Sampling one frame every N seconds is blind to
that: a short lot can pass entirely between two samples (no OCR evidence at
all), while a long lot gets the same board re-OCR'd 6+ times for nothing.
This module finds the timestamps where the screen actually changed, so the
screenshot stage can sample by content instead of by clock.
"""
import re
import statistics
import subprocess
from pathlib import Path

# ffmpeg's metadata filter prints two lines per analyzed frame: the frame's
# presentation timestamp, then (on the next line) its scene-change score.
# These regexes pull the (pts_time, scene_score) pairs off stderr in the
# order ffmpeg printed them.
_PTS_RE = re.compile(r"pts_time:([0-9]+(?:\.[0-9]+)?)")
_SCORE_RE = re.compile(r"lavfi\.scene_score=([0-9]+(?:\.[0-9]+)?)")

# Decoding the full video is the expensive part, so detection runs at a
# reduced scale and frame rate: "did the screen change" needs neither full
# resolution nor full frame rate, and this cuts the decode volume roughly
# 15x on a one-hour video versus analyzing at native size/fps.
_ANALYSIS_WIDTH = 320
_ANALYSIS_FPS = 2

# Absolute floor separating a real screen change from compression noise.
# The fixed 0.15 threshold commonly cited for ffmpeg's scene filter is
# calibrated for natural video (camera motion, cuts between shots) and is
# reported to detect ZERO changes on a static, light-background graphic
# overlay — which is what our lot board is. Hence the adaptive cutoff below.
#
# NOT YET CALIBRATED ON REAL AUCTION FOOTAGE. This floor is carried over from
# the prior art this was adapted from (slide-deck screencasts, where a slide
# change scores ~0.024-0.035 and identical frames ~0.00003). Re-measure against
# a real auction video before trusting scene sampling in production, and
# adjust this constant if the lot board scores differently.
_ABS_FLOOR = 0.008

# In natural video the baseline noise floor is much higher (camera shake,
# motion, compression artifacts), so the cutoff has to scale with it: a
# multiple of the video's own median score, which stands in for that
# video's noise level. On our mostly-static lot board the median stays tiny,
# so the absolute floor above dominates instead.
_MEDIAN_MULTIPLIER = 8.0


def _parse_scene_scores(stderr: str) -> list[tuple[float, float]]:
    """Extract (timestamp, scene_score) pairs from ffmpeg's stderr, in order."""
    pairs: list[tuple[float, float]] = []
    pending_ts: float | None = None
    for line in (stderr or "").splitlines():
        pts_match = _PTS_RE.search(line)
        if pts_match:
            pending_ts = float(pts_match.group(1))
            continue
        score_match = _SCORE_RE.search(line)
        if score_match and pending_ts is not None:
            pairs.append((pending_ts, float(score_match.group(1))))
            pending_ts = None
    return pairs


def _scene_threshold(scores: list[float]) -> float:
    """Score cutoff adapted to this specific video.

    A fixed cutoff can't serve both cases: a static graphic overlay changes
    by a small amount, while ordinary camera motion in natural video already
    exceeds that. The cutoff is the larger of an absolute floor and a
    multiple of this video's own median score, which approximates that
    video's baseline noise level.
    """
    if not scores:
        return _ABS_FLOOR
    return max(_ABS_FLOOR, statistics.median(scores) * _MEDIAN_MULTIPLIER)


def _collapse_close_timestamps(detected: list[float], min_gap: float) -> list[float]:
    """Prepend 0.0 and drop detections closer than `min_gap` to the last kept one.

    The first board is never a "change", so 0.0 always leads. Two detections
    close together are almost always the same transition (a fade, a bullet
    animating in); keeping both just doubles OCR work for no new information.
    """
    kept = [0.0]
    for value in detected:
        if value - kept[-1] >= min_gap:
            kept.append(value)
    return kept


def detect_scene_changes(
    video_path: Path,
    *,
    min_gap: float = 2.0,
    threshold: float | None = None,
    timeout: float = 1800.0,
) -> list[float]:
    """Find timestamps (seconds) where the screen actually changed.

    `threshold` overrides the adaptive cutoff manually; with None (the
    default) it is derived from this video's own score distribution via
    `_scene_threshold`, which is what makes detection work on both a static
    lot board and ordinary footage.

    Returns an empty list when ffmpeg fails, times out, or no change is
    detected. An empty list is a legitimate result here (e.g. nothing scored
    above the cutoff), not an error swallowed silently — the caller decides
    the fallback (typically fixed-interval sampling).
    """
    command = [
        "ffmpeg",
        "-hide_banner",
        "-nostats",
        "-i",
        str(video_path),
        "-an",
        "-vf",
        (
            f"scale={_ANALYSIS_WIDTH}:-2,fps={_ANALYSIS_FPS},"
            "select='gt(scene,-1)',metadata=print:key=lavfi.scene_score"
        ),
        "-f",
        "null",
        "-",
    ]
    try:
        # Our auction videos run 4+ hours and this pass decodes the whole
        # thing, so a stuck ffmpeg process must not hang the pipeline
        # forever. Timing out just means we fall back to interval sampling.
        proc = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return []

    if proc.returncode != 0:
        return []

    pairs = _parse_scene_scores(proc.stderr or "")
    if not pairs:
        return []

    cut = threshold if threshold is not None else _scene_threshold([score for _, score in pairs])
    detected = sorted({round(ts, 2) for ts, score in pairs if score >= cut})
    if not detected:
        return []

    return _collapse_close_timestamps(detected, min_gap)

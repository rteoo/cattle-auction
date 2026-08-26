"""Tests for scene-change detection (pure unit tests, no ffmpeg execution)."""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline import scenes


def _fake_stderr(pairs):
    """Build canned ffmpeg metadata-print stderr from (pts_time, score) pairs."""
    lines = []
    for ts, score in pairs:
        lines.append(f"frame:0    pts:0      pts_time:{ts}")
        lines.append(f"lavfi.scene_score={score}")
    return "\n".join(lines)


def test_parses_pts_and_score_pairs_in_order():
    stderr = _fake_stderr([(0.0, 0.0001), (0.5, 0.02), (1.0, 0.0002)])
    pairs = scenes._parse_scene_scores(stderr)
    assert pairs == [(0.0, 0.0001), (0.5, 0.02), (1.0, 0.0002)]


def test_detect_scene_changes_uses_parsed_scores(monkeypatch):
    # 3.0 is far enough from 0.0 to survive the default 2.0s min_gap.
    stderr = _fake_stderr([(0.0, 0.0001), (3.0, 0.5), (3.2, 0.0002)])

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr=stderr)

    monkeypatch.setattr(scenes.subprocess, "run", fake_run)

    result = scenes.detect_scene_changes(Path("video.mp4"))

    assert result[0] == 0.0
    assert any(abs(ts - 3.0) < 1e-6 for ts in result)
    assert 3.2 not in result


def test_adaptive_threshold_catches_change_fixed_015_would_miss(monkeypatch):
    # Baseline noise sits around 0.003; one real change at 0.03 — well below
    # the commonly cited fixed 0.15 cutoff, but ~10x this video's own
    # median, so the adaptive cutoff (median * 8) catches it while 0.15
    # would find nothing at all.
    pairs = [(float(i), 0.003) for i in range(10)]
    pairs.append((10.0, 0.03))
    stderr = _fake_stderr(pairs)
    scores = [score for _, score in pairs]

    assert all(score < 0.15 for score in scores)
    assert scenes._scene_threshold(scores) < 0.03

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr=stderr)

    monkeypatch.setattr(scenes.subprocess, "run", fake_run)

    result = scenes.detect_scene_changes(Path("video.mp4"))

    assert any(abs(ts - 10.0) < 1e-6 for ts in result)


def test_scene_threshold_is_max_of_floor_and_median_multiple():
    assert scenes._scene_threshold([]) == scenes._ABS_FLOOR
    # Median 0.0001 -> median*8 well under the floor, so the floor wins.
    assert scenes._scene_threshold([0.0001, 0.0002, 0.0001]) == scenes._ABS_FLOOR
    # Median 0.01 -> median*8 = 0.08, comfortably above the floor.
    assert scenes._scene_threshold([0.01, 0.01, 0.01]) == 0.08


def test_min_gap_collapses_close_detections(monkeypatch):
    pairs = [(0.0, 0.0), (10.0, 0.5), (10.5, 0.5), (11.0, 0.5), (20.0, 0.5)]
    stderr = _fake_stderr(pairs)

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr=stderr)

    monkeypatch.setattr(scenes.subprocess, "run", fake_run)

    result = scenes.detect_scene_changes(Path("video.mp4"), min_gap=2.0, threshold=0.4)

    assert result == [0.0, 10.0, 20.0]


def test_zero_is_always_first_timestamp(monkeypatch):
    pairs = [(5.0, 0.5), (10.0, 0.5)]
    stderr = _fake_stderr(pairs)

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr=stderr)

    monkeypatch.setattr(scenes.subprocess, "run", fake_run)

    result = scenes.detect_scene_changes(Path("video.mp4"), threshold=0.1)

    assert result[0] == 0.0
    assert result == [0.0, 5.0, 10.0]


def test_returns_empty_when_no_score_clears_threshold(monkeypatch):
    stderr = _fake_stderr([(0.0, 0.0001), (1.0, 0.0002)])

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr=stderr)

    monkeypatch.setattr(scenes.subprocess, "run", fake_run)

    assert scenes.detect_scene_changes(Path("video.mp4")) == []


def test_returns_empty_on_nonzero_exit(monkeypatch):
    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="ffmpeg error")

    monkeypatch.setattr(scenes.subprocess, "run", fake_run)

    assert scenes.detect_scene_changes(Path("video.mp4")) == []


def test_returns_empty_on_timeout(monkeypatch):
    def fake_run(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout", 1800.0))

    monkeypatch.setattr(scenes.subprocess, "run", fake_run)

    assert scenes.detect_scene_changes(Path("video.mp4")) == []

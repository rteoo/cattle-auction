"""Tests for the screenshot stage (interval and scene sampling)."""
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline import screenshotter


def _make_fake_popen(duration):
    """Fake ffmpeg process for the single-pass interval extraction path.

    Writes the same number of dummy frame files a real `fps=1/interval` pass
    would produce, named the way ffmpeg itself names them (frame_%06d.jpg),
    without running ffmpeg.
    """

    class FakePopen:
        def __init__(self, cmd, stdout=None, text=None):
            self.cmd = cmd
            self.returncode = 0
            self.stdout = iter([])
            pattern_arg = next(a for a in cmd if "%06d" in a)
            pattern_path = Path(pattern_arg)
            vf_value = cmd[cmd.index("-vf") + 1]
            interval = int(vf_value.split("fps=1/")[1])
            total_frames = int(duration // interval) + 1
            for i in range(1, total_frames + 1):
                out = pattern_path.parent / pattern_path.name.replace("%06d", f"{i:06d}")
                out.write_bytes(b"fake")

        def wait(self):
            return None

    return FakePopen


def _fake_run_writes_output(calls):
    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        Path(cmd[-1]).write_bytes(b"fake")
        return subprocess.CompletedProcess(cmd, 0)

    return fake_run


def test_scene_sampling_unions_detected_and_safety_grid(monkeypatch, tmp_path):
    monkeypatch.setattr(screenshotter, "_video_duration", lambda video_path: 130.0)
    monkeypatch.setattr(screenshotter, "detect_scene_changes", lambda video_path, **kw: [0.0, 12.3, 45.0])

    calls = []
    monkeypatch.setattr(screenshotter.subprocess, "run", _fake_run_writes_output(calls))

    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"video")

    shots = screenshotter.extract_screenshots(
        video_path, tmp_path, "vid", sampling="scene", safety_interval=60,
    )

    # union of {0.0, 12.3, 45.0} (detected) and {0, 60, 120} (safety grid),
    # nothing closer than 2.0s so nothing collapses.
    assert [s.seconds for s in shots] == [0, 12, 45, 60, 120]
    assert len(calls) == 5


def test_scene_sampling_falls_back_to_interval_when_no_changes_detected(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(screenshotter, "detect_scene_changes", lambda video_path, **kw: [])
    monkeypatch.setattr(screenshotter, "_video_duration", lambda video_path: 65.0)
    monkeypatch.setattr(screenshotter.subprocess, "Popen", _make_fake_popen(duration=65.0))

    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"video")

    shots = screenshotter.extract_screenshots(
        video_path, tmp_path, "vid", interval=30, sampling="scene",
    )

    assert [s.seconds for s in shots] == [0, 30, 60]
    out = capsys.readouterr().out.lower()
    assert "falling back to interval" in out


def test_scene_frame_command_uses_jpeg_quality_flag_and_extension(monkeypatch, tmp_path):
    monkeypatch.setattr(screenshotter, "detect_scene_changes", lambda video_path, **kw: [0.0])
    monkeypatch.setattr(screenshotter, "_video_duration", lambda video_path: 5.0)

    calls = []
    monkeypatch.setattr(screenshotter.subprocess, "run", _fake_run_writes_output(calls))

    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"video")

    shots = screenshotter.extract_screenshots(
        video_path, tmp_path, "vid", sampling="scene",
    )

    assert shots[0].path.suffix == ".jpg"
    assert "-q:v" in calls[-1]
    assert calls[-1][calls[-1].index("-q:v") + 1] == "2"


def test_interval_sampling_keeps_ffmpeg_frame_numbering(monkeypatch, tmp_path):
    """Interval mode must stay byte-for-byte what it was before scene sampling."""
    monkeypatch.setattr(screenshotter, "_video_duration", lambda video_path: 65.0)
    monkeypatch.setattr(screenshotter.subprocess, "Popen", _make_fake_popen(duration=65.0))

    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"video")

    shots = screenshotter.extract_screenshots(
        video_path, tmp_path, "vid", interval=30, sampling="interval",
    )

    assert [s.seconds for s in shots] == [0, 30, 60]
    assert shots[0].path.name == "frame_000001.jpg"
    assert shots[1].path.name == "frame_000002.jpg"
    for shot in shots:
        assert shot.path.exists()


def test_scene_sampling_skips_failed_frames_without_aborting(monkeypatch, tmp_path, capsys):
    """One unreadable moment must not kill a run that may already be hours in."""
    monkeypatch.setattr(screenshotter, "_video_duration", lambda video_path: 5.0)
    monkeypatch.setattr(
        screenshotter, "detect_scene_changes", lambda video_path, **kw: [0.0, 10.0, 20.0]
    )

    def fake_run(cmd, **kwargs):
        output = Path(cmd[-1])
        # Fail only the middle timestamp.
        if "00010s00" in output.name:
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="boom")
        output.write_bytes(b"fake")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(screenshotter.subprocess, "run", fake_run)

    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"video")

    shots = screenshotter.extract_screenshots(
        video_path, tmp_path, "vid", sampling="scene", safety_interval=60,
    )

    assert [s.seconds for s in shots] == [0, 20]
    assert "1 frame(s) could not be extracted" in capsys.readouterr().out


def test_zero_byte_frame_is_treated_as_a_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(screenshotter, "_video_duration", lambda video_path: 5.0)
    monkeypatch.setattr(screenshotter, "detect_scene_changes", lambda video_path, **kw: [0.0, 10.0])

    def fake_run(cmd, **kwargs):
        output = Path(cmd[-1])
        # ffmpeg exits 0 but leaves an empty file behind.
        output.write_bytes(b"" if "00010s00" in output.name else b"fake")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(screenshotter.subprocess, "run", fake_run)

    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"video")

    shots = screenshotter.extract_screenshots(
        video_path, tmp_path, "vid", sampling="scene", safety_interval=60,
    )

    assert [s.seconds for s in shots] == [0]


def test_scene_cache_survives_an_unrelated_interval_change(monkeypatch, tmp_path):
    """`--screenshot-interval` does not shape scene output, so it must not invalidate it."""
    screenshots_dir = tmp_path / "screenshots_vid"
    screenshots_dir.mkdir()
    frame = screenshots_dir / "frame_000001_00000s00.jpg"
    frame.write_bytes(b"frame")
    index_path = tmp_path / "screenshots_vid.json"
    index_path.write_text(
        json.dumps(
            {
                "sampling": "scene",
                "interval": 30,
                "safety_interval": 60,
                "source_key": screenshotter._source_key(tmp_path / "video.mp4"),
                "screenshots": [{"seconds": 0, "timestamp_str": "00:00:00", "path": str(frame)}],
            }
        ),
        encoding="utf-8",
    )

    def fail(*args, **kwargs):
        raise AssertionError("must not re-extract when only `interval` changed in scene mode")

    monkeypatch.setattr(screenshotter, "detect_scene_changes", fail)
    monkeypatch.setattr(screenshotter.subprocess, "run", fail)

    shots = screenshotter.extract_screenshots(
        tmp_path / "video.mp4", tmp_path, "vid", interval=45, sampling="scene", safety_interval=60,
    )

    assert [s.seconds for s in shots] == [0]


def test_stale_frames_are_cleared_before_a_fresh_extraction(monkeypatch, tmp_path):
    """Debris from an interrupted run must not be globbed into the new index."""
    screenshots_dir = tmp_path / "screenshots_vid"
    screenshots_dir.mkdir()
    (screenshots_dir / "frame_000009.jpg").write_bytes(b"debris")
    (screenshots_dir / "frame_000010_00300s00.jpg").write_bytes(b"debris")

    monkeypatch.setattr(screenshotter, "_video_duration", lambda video_path: 5.0)
    monkeypatch.setattr(screenshotter.subprocess, "Popen", _make_fake_popen(duration=5.0))

    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"video")

    # No index file at all, so nothing signals that a previous run half-finished.
    shots = screenshotter.extract_screenshots(
        video_path, tmp_path, "vid", interval=30, sampling="interval",
    )

    assert [s.seconds for s in shots] == [0]
    assert not (screenshots_dir / "frame_000009.jpg").exists()
    assert not (screenshots_dir / "frame_000010_00300s00.jpg").exists()


def test_legacy_list_shaped_cache_still_loads_and_is_never_invalidated(tmp_path):
    screenshots_dir = tmp_path / "screenshots_vid"
    screenshots_dir.mkdir()
    frame = screenshots_dir / "frame_000001.jpg"
    frame.write_bytes(b"frame")
    index_path = tmp_path / "screenshots_vid.json"
    index_path.write_text(
        json.dumps([{"seconds": 0, "timestamp_str": "00:00:00", "path": str(frame)}]),
        encoding="utf-8",
    )

    # Requested params differ wildly from anything the legacy cache could
    # have recorded; it must still be trusted as-is.
    shots = screenshotter.extract_screenshots(
        tmp_path / "video.mp4", tmp_path, "vid", interval=45, sampling="scene", safety_interval=15,
    )

    assert len(shots) == 1
    assert shots[0].seconds == 0
    assert shots[0].path == frame


def test_param_mismatch_invalidates_cache_and_reextracts(monkeypatch, tmp_path, capsys):
    screenshots_dir = tmp_path / "screenshots_vid"
    screenshots_dir.mkdir()
    # Old cache from interval=30 over a longer video: two frames.
    stale_frame_1 = screenshots_dir / "frame_000001_00000s00.jpg"
    stale_frame_2 = screenshots_dir / "frame_000002_00030s00.jpg"
    stale_frame_1.write_bytes(b"stale")
    stale_frame_2.write_bytes(b"stale")
    index_path = tmp_path / "screenshots_vid.json"
    index_path.write_text(
        json.dumps(
            {
                "sampling": "interval",
                "interval": 30,
                "safety_interval": 60,
                "source_key": screenshotter._source_key(tmp_path / "video.mp4"),
                "screenshots": [
                    {"seconds": 0, "timestamp_str": "00:00:00", "path": str(stale_frame_1)},
                    {"seconds": 30, "timestamp_str": "00:00:30", "path": str(stale_frame_2)},
                ],
            }
        ),
        encoding="utf-8",
    )

    # New request: interval=10 over a 5s video produces only one frame, so a
    # leftover frame_000002 file surviving would prove the cache clear didn't run.
    monkeypatch.setattr(screenshotter, "_video_duration", lambda video_path: 5.0)
    monkeypatch.setattr(screenshotter.subprocess, "Popen", _make_fake_popen(duration=5.0))

    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"video")

    shots = screenshotter.extract_screenshots(
        video_path, tmp_path, "vid", interval=10, sampling="interval",
    )

    assert [s.seconds for s in shots] == [0]
    # Both stale frames are gone and the new pass wrote its own frame_000001.jpg.
    assert not stale_frame_1.exists()
    assert not stale_frame_2.exists()
    assert shots[0].path.name == "frame_000001.jpg"
    assert shots[0].path.read_bytes() != b"stale"
    out = capsys.readouterr().out.lower()
    assert "re-extracting" in out


def test_matching_params_uses_cache_without_reextracting(monkeypatch, tmp_path):
    screenshots_dir = tmp_path / "screenshots_vid"
    screenshots_dir.mkdir()
    frame = screenshots_dir / "frame_000001_00000s00.jpg"
    frame.write_bytes(b"frame")
    index_path = tmp_path / "screenshots_vid.json"
    index_path.write_text(
        json.dumps(
            {
                "sampling": "interval",
                "interval": 30,
                "safety_interval": 60,
                "source_key": screenshotter._source_key(tmp_path / "video.mp4"),
                "screenshots": [{"seconds": 0, "timestamp_str": "00:00:00", "path": str(frame)}],
            }
        ),
        encoding="utf-8",
    )

    def fail_popen(*args, **kwargs):
        raise AssertionError("ffmpeg should not run when cache matches requested params")

    monkeypatch.setattr(screenshotter.subprocess, "Popen", fail_popen)

    shots = screenshotter.extract_screenshots(
        tmp_path / "video.mp4", tmp_path, "vid", interval=30, sampling="interval", safety_interval=60,
    )

    assert [s.seconds for s in shots] == [0]
    assert frame.exists()


def test_modern_cache_with_missing_frame_is_reextracted(monkeypatch, tmp_path):
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"video")
    missing_frame = tmp_path / "screenshots_vid" / "frame_000001.jpg"
    index_path = tmp_path / "screenshots_vid.json"
    index_path.write_text(
        json.dumps(
            {
                "sampling": "interval",
                "interval": 30,
                "safety_interval": 60,
                "source_key": screenshotter._source_key(video_path),
                "screenshots": [
                    {
                        "seconds": 0,
                        "timestamp_str": "00:00:00",
                        "path": str(missing_frame),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(screenshotter, "_video_duration", lambda path: 5.0)
    monkeypatch.setattr(screenshotter.subprocess, "Popen", _make_fake_popen(duration=5.0))

    shots = screenshotter.extract_screenshots(video_path, tmp_path, "vid")

    assert [shot.seconds for shot in shots] == [0]
    assert shots[0].path.is_file()


def test_scene_timestamps_never_collide_on_the_ocr_dict_key(monkeypatch, tmp_path):
    """pipeline/ocr.py keys by timestamp_str, so two frames must never share one.

    Detections deliberately land sub-second apart and on top of grid points;
    the _MIN_GAP collapse is what keeps every surviving frame on its own second.
    """
    monkeypatch.setattr(screenshotter, "_video_duration", lambda video_path: 200.0)
    monkeypatch.setattr(
        screenshotter,
        "detect_scene_changes",
        lambda video_path, **kw: [0.0, 0.4, 59.6, 60.2, 60.4, 119.7, 120.3],
    )

    calls = []
    monkeypatch.setattr(screenshotter.subprocess, "run", _fake_run_writes_output(calls))

    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"video")

    shots = screenshotter.extract_screenshots(
        video_path, tmp_path, "vid", sampling="scene", safety_interval=60,
    )

    keys = [s.timestamp_str for s in shots]
    assert len(keys) == len(set(keys)), f"duplicate OCR keys: {keys}"
    # And the frame files themselves stay distinct.
    assert len({s.path for s in shots}) == len(shots)


def test_screenshot_cache_source_key_distinguishes_video_resolution():
    params = {
        "legacy": False,
        "sampling": "interval",
        "interval": 30,
        "safety_interval": 60,
        "source_key": "video_ocr_vid_480p.mp4",
    }

    assert not screenshotter._params_match(
        params,
        sampling="interval",
        interval=30,
        safety_interval=60,
        source_key="video_ocr_vid_720p.mp4",
    )

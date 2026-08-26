import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

from pipeline.scenes import detect_scene_changes

# Minimum spacing (seconds) between kept timestamps in scene sampling.
#
# Load-bearing invariant, not just a tuning knob: pipeline/ocr.py keys its
# results dict by `timestamp_str` (HH:MM:SS), so two frames that round to the
# same second would silently overwrite each other and lose one frame's OCR
# text. Any value >= 1.0 makes that collision impossible. Do not lower this
# below 1.0 without giving OCR results a key that survives sub-second spacing.
_MIN_GAP = 2.0


@dataclass
class Screenshot:
    seconds: int
    timestamp_str: str  # HH:MM:SS
    path: Path


def extract_screenshots(
    video_path: Path,
    output_dir: Path,
    video_id: str,
    interval: int = 30,
    sampling: str = "interval",
    safety_interval: int = 60,
) -> list[Screenshot]:
    """Extract screenshots from the OCR video, by fixed interval or by scene change.

    `sampling="interval"` samples one frame every `interval` seconds — blind to
    content, but simple and predictable.

    `sampling="scene"` samples where the auction "lot board" graphic actually
    changed (see `pipeline.scenes.detect_scene_changes`), unioned with a coarse
    safety grid every `safety_interval` seconds so scene detection under-firing
    never loses coverage entirely. This tracks lot boundaries far more closely
    than a fixed grid, while still producing far fewer frames than sampling
    every `interval` seconds. Falls back to interval sampling if detection
    finds nothing (e.g. no lot-board graphic in the video at all).
    """
    screenshots_dir = output_dir / f"screenshots_{video_id}"
    index_path = output_dir / f"screenshots_{video_id}.json"

    cached = _load(index_path)
    if cached is not None:
        cached_screenshots, cached_params = cached
        if _params_match(cached_params, sampling, interval, safety_interval):
            print("  Screenshots already extracted, loading from cache.")
            return cached_screenshots
        print(
            "  Screenshot sampling parameters changed since the cached run "
            f"(requested sampling={sampling} interval={interval} "
            f"safety_interval={safety_interval}); re-extracting."
        )

    screenshots_dir.mkdir(parents=True, exist_ok=True)
    # Always extract into an empty directory. Debris from an interrupted run,
    # or from a previous sampling mode, would otherwise be picked up by the
    # interval path's frame glob and mapped onto the wrong timestamps.
    _clear_frames(screenshots_dir)

    if sampling == "scene":
        scene_timestamps = detect_scene_changes(video_path)
        if not scene_timestamps:
            print("  Scene detection found no changes; falling back to interval sampling.")
            screenshots = _extract_interval(video_path, screenshots_dir, interval)
        else:
            duration = _video_duration(video_path)
            timestamps = _merge_with_safety_grid(scene_timestamps, duration, safety_interval)
            screenshots = _extract_at_timestamps(video_path, screenshots_dir, timestamps)
    else:
        screenshots = _extract_interval(video_path, screenshots_dir, interval)

    print(f"  Extracted {len(screenshots)} screenshots.")
    _save(screenshots, index_path, sampling=sampling, interval=interval, safety_interval=safety_interval)
    return screenshots


def _merge_with_safety_grid(
    scene_timestamps: list[float],
    duration: float,
    safety_interval: int,
    min_gap: float = _MIN_GAP,
) -> list[float]:
    """Union of detected scene changes and a coarse fixed grid.

    The grid guarantees we never lose coverage if scene detection under-fires
    on some stretch of video; the union is still far sparser than sampling
    every `interval` seconds. Timestamps closer than `min_gap` collapse to
    the earlier one.
    """
    safety_interval = max(1, int(safety_interval))
    grid = []
    t = 0.0
    while t <= duration:
        grid.append(t)
        t += safety_interval
    if not grid:
        grid = [0.0]

    combined = sorted(set(scene_timestamps) | set(grid))
    collapsed: list[float] = []
    for ts in combined:
        if collapsed and ts - collapsed[-1] < min_gap:
            continue
        collapsed.append(ts)
    return collapsed


def _extract_interval(video_path: Path, screenshots_dir: Path, interval: int) -> list[Screenshot]:
    """Extract one frame every `interval` seconds with a single ffmpeg pass."""
    duration = _video_duration(video_path)
    total_frames = int(duration // interval) + 1

    _run_ffmpeg_with_progress(video_path, screenshots_dir, interval, duration, total_frames)

    # ffmpeg numbers frames sequentially as frame_000001.jpg, frame_000002.jpg, ...
    # Frame N (1-indexed) sits at timestamp (N-1)*interval seconds.
    frames = sorted(screenshots_dir.glob("frame_*.jpg"))
    screenshots = []
    for i, frame_path in enumerate(frames):
        secs = i * interval
        screenshots.append(Screenshot(seconds=secs, timestamp_str=_fmt(secs), path=frame_path))
    return screenshots


def _extract_at_timestamps(
    video_path: Path,
    screenshots_dir: Path,
    timestamps: list[float],
) -> list[Screenshot]:
    """Extract one frame per timestamp, each with its own ffmpeg seek-and-grab call."""
    screenshots: list[Screenshot] = []
    failed = 0

    with Progress(
        TextColumn("  [progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("[cyan]{task.fields[ts]}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task(
            f"Extracting frames (scene-based, {len(timestamps)} frames)",
            total=len(timestamps),
            ts="00:00:00",
        )

        for i, ts in enumerate(timestamps, start=1):
            secs = int(round(ts))
            timestamp_str = _fmt(secs)
            output = screenshots_dir / _frame_filename(i, ts)
            command = [
                "ffmpeg", "-y",
                "-ss", str(ts),
                "-i", str(video_path),
                "-frames:v", "1",
                "-q:v", "2",
                "-hide_banner", "-loglevel", "error",
                str(output),
            ]
            proc = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            progress.update(task, advance=1, ts=timestamp_str)

            # Scene sampling issues one ffmpeg call per timestamp, so a single
            # unreadable moment must not abort a run that may already be hours
            # in. A skipped frame simply contributes no OCR evidence.
            if proc.returncode != 0 or not output.exists() or output.stat().st_size == 0:
                failed += 1
                continue

            screenshots.append(Screenshot(seconds=secs, timestamp_str=timestamp_str, path=output))

    if failed:
        print(f"  [warning] {failed} frame(s) could not be extracted and were skipped.")

    return screenshots


def _run_ffmpeg_with_progress(
    video_path: Path,
    screenshots_dir: Path,
    interval: int,
    duration: float,
    total_frames: int,
) -> None:
    proc = subprocess.Popen(
        [
            "ffmpeg",
            "-i", str(video_path),
            "-vf", f"fps=1/{interval}",
            "-q:v", "2",
            str(screenshots_dir / "frame_%06d.jpg"),
            "-progress", "pipe:1",
            "-nostats",
            "-hide_banner",
            "-loglevel", "error",
        ],
        stdout=subprocess.PIPE,
        text=True,
    )

    with Progress(
        TextColumn("  [progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("[cyan]{task.fields[ts]}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task(
            f"Extracting frames (every {interval}s)",
            total=total_frames,
            ts="00:00:00",
        )

        for line in proc.stdout:
            line = line.strip()
            if line.startswith("out_time="):
                ts = line.split("=", 1)[1]
                secs = _parse_time(ts)
                frame = int(secs // interval)
                progress.update(task, completed=frame, ts=ts[:8])

    proc.wait()
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, "ffmpeg")


def _video_duration(video_path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_streams", str(video_path),
        ],
        capture_output=True, text=True, check=True,
    )
    data = json.loads(result.stdout)
    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video":
            return float(stream["duration"])
    raise RuntimeError(f"Could not determine duration of {video_path}")


def _parse_time(ts: str) -> float:
    # Format from ffmpeg -progress: "HH:MM:SS.ffffff" or "N/A"
    try:
        parts = ts.split(":")
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    except Exception:
        return 0.0


def _fmt(seconds: int) -> str:
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def _frame_filename(index: int, timestamp: float) -> str:
    return f"frame_{index:06d}_{_safe_ts(timestamp)}.jpg"


def _safe_ts(timestamp: float) -> str:
    return f"{timestamp:08.2f}".replace(".", "s")


def _save(
    screenshots: list[Screenshot],
    path: Path,
    *,
    sampling: str,
    interval: int,
    safety_interval: int,
) -> None:
    data = {
        "sampling": sampling,
        "interval": interval,
        "safety_interval": safety_interval,
        "screenshots": [
            {"seconds": s.seconds, "timestamp_str": s.timestamp_str, "path": str(s.path)}
            for s in screenshots
        ],
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _load(path: Path) -> tuple[list[Screenshot], dict] | None:
    """Load a cached screenshot index, if any, along with the params it was made with.

    The legacy format (written before sampling parameters were tracked) is a
    bare JSON list. It is treated as sampling="interval" with an unknown
    interval and is never invalidated by a parameter mismatch — an unknown
    interval means "trust it".
    """
    if not path.exists():
        return None

    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        screenshots = [
            Screenshot(seconds=d["seconds"], timestamp_str=d["timestamp_str"], path=Path(d["path"]))
            for d in raw
        ]
        return screenshots, {"legacy": True}

    screenshots = [
        Screenshot(seconds=d["seconds"], timestamp_str=d["timestamp_str"], path=Path(d["path"]))
        for d in raw["screenshots"]
    ]
    params = {
        "legacy": False,
        "sampling": raw.get("sampling"),
        "interval": raw.get("interval"),
        "safety_interval": raw.get("safety_interval"),
    }
    return screenshots, params


def _params_match(
    cached_params: dict,
    sampling: str,
    interval: int,
    safety_interval: int,
) -> bool:
    if cached_params.get("legacy"):
        return True
    if cached_params.get("sampling") != sampling:
        return False
    # Compare only the knobs that shape this mode's output: scene sampling
    # ignores `interval`, interval sampling ignores the safety grid. Otherwise
    # an unrelated flag change would throw away expensive cached frames.
    if sampling == "scene":
        return cached_params.get("safety_interval") == safety_interval
    return cached_params.get("interval") == interval


def _clear_frames(screenshots_dir: Path) -> None:
    if not screenshots_dir.exists():
        return
    for frame_path in screenshots_dir.glob("frame_*"):
        frame_path.unlink(missing_ok=True)

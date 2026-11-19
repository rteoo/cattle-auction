import json
import math
import re
import subprocess
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path

from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

from pipeline.checkpoint import write_json
from pipeline.scenes import detect_scene_changes

_MIN_GAP = 2.0
_FRAME_INDEX_VERSION = 2
_SHOWINFO_PTS = re.compile(r"showinfo.*\bn:\s*\d+.*\bpts_time:\s*([\d.eE+\-]+)")


@dataclass
class Screenshot:
    seconds: float
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
    source_key = _source_key(video_path)

    cached = _load(index_path)
    if cached is not None:
        cached_screenshots, cached_params = cached
        if (
            _params_match(
                cached_params,
                sampling,
                interval,
                safety_interval,
                source_key=source_key,
            )
            and _cached_frames_usable(cached_screenshots)
        ):
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

    if not screenshots:
        raise RuntimeError("No timestamped video frames could be extracted.")
    print(f"  Extracted {len(screenshots)} screenshots.")
    _save(
        screenshots,
        index_path,
        sampling=sampling,
        interval=interval,
        safety_interval=safety_interval,
        source_key=source_key,
    )
    return screenshots


def _merge_with_safety_grid(
    scene_timestamps: list[float],
    duration: float,
    safety_interval: int,
    min_gap: float = _MIN_GAP,
) -> list[float]:
    """Union of detected scene changes and a coarse fixed grid.

    Keep transitions first, then try to capture the outgoing board shortly
    before each transition. Fill the remaining gaps with the safety grid.
    """
    safety_interval = max(1, int(safety_interval))
    grid = []
    t = 0.0
    while t <= duration:
        grid.append(t)
        t += safety_interval
    if not grid:
        grid = [0.0]

    kept: list[float] = []

    def add(ts: float) -> None:
        if not 0 <= ts < duration:
            return
        index = bisect_left(kept, ts)
        if index and ts - kept[index - 1] + 1e-6 < min_gap:
            return
        if index < len(kept) and kept[index] - ts + 1e-6 < min_gap:
            return
        kept.insert(index, ts)

    for ts in sorted(scene_timestamps):
        add(ts)
    for ts in sorted(scene_timestamps):
        add(round(ts - min_gap, 3))
    for ts in grid:
        add(ts)
    return kept


def _extract_interval(video_path: Path, screenshots_dir: Path, interval: int) -> list[Screenshot]:
    """Extract one frame every `interval` seconds with a single ffmpeg pass."""
    duration = _video_duration(video_path)
    total_frames = int(duration // interval) + 1

    timestamps = _run_ffmpeg_with_progress(video_path, screenshots_dir, interval, total_frames)

    frames = sorted(screenshots_dir.glob("frame_*.jpg"))
    if len(frames) != len(timestamps):
        raise RuntimeError("FFmpeg frame files and source timestamps do not match.")
    return [Screenshot(seconds=ts, timestamp_str=_fmt(int(ts)), path=path)
            for ts, path in zip(timestamps, frames, strict=True)]


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
            output = screenshots_dir / _frame_filename(i, ts)
            command = [
                "ffmpeg", "-y",
                "-ss", str(ts),
                "-i", str(video_path),
                "-vf", "showinfo",
                "-frames:v", "1",
                "-q:v", "2",
                "-hide_banner", "-loglevel", "info",
                str(output),
            ]
            proc = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            progress.update(task, advance=1, ts=_fmt(int(ts)))

            # Scene sampling issues one ffmpeg call per timestamp, so a single
            # unreadable moment must not abort a run that may already be hours
            # in. A skipped frame simply contributes no OCR evidence.
            match = _SHOWINFO_PTS.search(proc.stderr or "")
            if proc.returncode != 0 or not match or not output.exists() or output.stat().st_size == 0:
                failed += 1
                continue

            seconds = ts + float(match[1])
            screenshots.append(Screenshot(seconds=seconds, timestamp_str=_fmt(int(seconds)), path=output))

    if failed:
        print(f"  [warning] {failed} frame(s) could not be extracted and were skipped.")

    return screenshots


def _run_ffmpeg_with_progress(
    video_path: Path,
    screenshots_dir: Path,
    interval: int,
    total_frames: int,
) -> list[float]:
    timestamps = []
    proc = subprocess.Popen(
        [
            "ffmpeg",
            "-i", str(video_path),
            "-vf", f"setpts=PTS-STARTPTS,select='gte(t,selected_n*{interval})',showinfo",
            "-fps_mode", "passthrough",
            "-q:v", "2",
            str(screenshots_dir / "frame_%06d.jpg"),
            "-nostats",
            "-hide_banner",
            "-loglevel", "info",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
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
            match = _SHOWINFO_PTS.search(line)
            if match:
                timestamps.append(float(match[1]))
                progress.update(task, completed=len(timestamps), ts=_fmt(int(timestamps[-1])))

    proc.wait()
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, "ffmpeg")
    return timestamps


def _video_duration(video_path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_streams", "-show_format", str(video_path),
        ],
        capture_output=True, text=True, check=True,
    )
    data = json.loads(result.stdout)
    stream = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), {})
    for candidate in (stream.get("duration"), data.get("format", {}).get("duration")):
        try:
            duration = float(candidate)
        except (ValueError, TypeError):
            continue
        if math.isfinite(duration) and duration > 0:
            return duration
    raise RuntimeError(f"Could not determine duration of {video_path}")


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
    source_key: dict,
) -> None:
    data = {
        "version": _FRAME_INDEX_VERSION,
        "sampling": sampling,
        "interval": interval,
        "safety_interval": safety_interval,
        "source_key": source_key,
        "screenshots": [
            {"seconds": s.seconds, "timestamp_str": s.timestamp_str, "path": str(s.path)}
            for s in screenshots
        ],
    }
    write_json(path, data)


def _load(path: Path) -> tuple[list[Screenshot], dict] | None:
    """Load a cached screenshot index, if any, along with the params it was made with.

    Legacy timestamps were estimates. Read their shape so resume can
    invalidate them and extract source-timestamped frames.
    """
    if not path.exists():
        return None

    try:
        return _parse_index(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError, KeyError, TypeError):
        print("  Screenshot index is unreadable, re-extracting.")
        return None


def _parse_index(raw) -> tuple[list[Screenshot], dict]:
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
        "version": raw.get("version"),
        "sampling": raw.get("sampling"),
        "interval": raw.get("interval"),
        "safety_interval": raw.get("safety_interval"),
        "source_key": raw.get("source_key"),
    }
    return screenshots, params


def _params_match(
    cached_params: dict,
    sampling: str,
    interval: int,
    safety_interval: int,
    *,
    source_key=None,
) -> bool:
    if cached_params.get("version") != _FRAME_INDEX_VERSION:
        return False
    if cached_params.get("sampling") != sampling:
        return False
    cached_source_key = cached_params.get("source_key")
    if cached_source_key != source_key:
        return False
    # Compare only the knobs that shape this mode's output: scene sampling
    # ignores `interval`, interval sampling ignores the safety grid. Otherwise
    # an unrelated flag change would throw away expensive cached frames.
    if sampling == "scene":
        return cached_params.get("safety_interval") == safety_interval
    return cached_params.get("interval") == interval


def _source_key(video_path: Path) -> dict:
    """Return a stable identity for the media source behind the frames."""
    identity = {"path": str(video_path.resolve())}
    try:
        stat = video_path.stat()
    except OSError:
        identity.update({"size": None, "mtime_ns": None})
    else:
        identity.update({"size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    return identity


def _cached_frames_usable(screenshots: list[Screenshot]) -> bool:
    return bool(screenshots) and all(
        shot.path.is_file() and shot.path.stat().st_size > 0
        for shot in screenshots
    )


def _clear_frames(screenshots_dir: Path) -> None:
    if not screenshots_dir.exists():
        return
    for frame_path in screenshots_dir.glob("frame_*"):
        frame_path.unlink(missing_ok=True)

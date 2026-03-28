import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn


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
) -> list[Screenshot]:
    """Extract one frame every `interval` seconds using ffmpeg."""
    screenshots_dir = output_dir / f"screenshots_{video_id}"
    index_path = output_dir / f"screenshots_{video_id}.json"

    if index_path.exists():
        print(f"  Screenshots already extracted, loading from cache.")
        return _load(index_path, screenshots_dir)

    screenshots_dir.mkdir(parents=True, exist_ok=True)

    duration = _video_duration(video_path)
    total_frames = int(duration // interval) + 1

    _run_ffmpeg_with_progress(video_path, screenshots_dir, interval, duration, total_frames)

    # Build index: frame N (1-indexed) → timestamp (N-1)*interval seconds
    frames = sorted(screenshots_dir.glob("frame_*.jpg"))
    screenshots = []
    for i, frame_path in enumerate(frames):
        secs = i * interval
        screenshots.append(Screenshot(
            seconds=secs,
            timestamp_str=_fmt(secs),
            path=frame_path,
        ))

    print(f"  Extracted {len(screenshots)} screenshots (every {interval}s).")
    _save(screenshots, index_path)
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


def _save(screenshots: list[Screenshot], path: Path) -> None:
    data = [{"seconds": s.seconds, "timestamp_str": s.timestamp_str, "path": str(s.path)} for s in screenshots]
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))


def _load(path: Path, screenshots_dir: Path) -> list[Screenshot]:
    data = json.loads(path.read_text())
    return [Screenshot(seconds=d["seconds"], timestamp_str=d["timestamp_str"], path=Path(d["path"])) for d in data]

import json
import re
import subprocess
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import yt_dlp


_YOUTUBE_HOSTS = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "music.youtube.com",
    "youtu.be",
    "www.youtu.be",
    "youtube-nocookie.com",
    "www.youtube-nocookie.com",
}
_VIDEO_ID_RE = re.compile(r"[A-Za-z0-9_-]{11}\Z")
_PATH_VIDEO_ID_RE = re.compile(r"[A-Za-z0-9_-]+\Z")


def get_video_id(url: str) -> str:
    parsed = urlparse(url)
    try:
        host = (parsed.hostname or "").lower()
    except ValueError as exc:
        raise ValueError("Invalid YouTube URL host.") from exc
    if host not in _YOUTUBE_HOSTS:
        raise ValueError("URL must use a recognized YouTube host.")

    if host in {"youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com"}:
        qs = parse_qs(parsed.query)
        if "v" in qs:
            return _validate_video_id(qs["v"][0])
    if host in {"youtu.be", "www.youtu.be"}:
        candidate = parsed.path.strip("/")
        if "/" in candidate:
            raise ValueError("Short YouTube URL contains an invalid video ID path.")
        return _validate_video_id(candidate)
    # Fallback: ask yt-dlp
    with yt_dlp.YoutubeDL({"quiet": True}) as ydl:
        info = ydl.extract_info(url, download=False)
        return _validate_video_id(info["id"])


def _validate_video_id(video_id: str) -> str:
    if not isinstance(video_id, str) or not _VIDEO_ID_RE.fullmatch(video_id):
        raise ValueError("YouTube video ID must be exactly 11 URL-safe characters.")
    return video_id


def _has_content(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _is_fresh_nonempty_file(path: Path, source: Path) -> bool:
    try:
        path_stat = path.stat()
        source_stat = source.stat()
    except OSError:
        return False
    return path.is_file() and path_stat.st_size > 0 and path_stat.st_mtime_ns >= source_stat.st_mtime_ns


def _validate_path_video_id(video_id: str) -> str:
    """Reject IDs that could escape the per-video output directory."""
    if not isinstance(video_id, str) or not _PATH_VIDEO_ID_RE.fullmatch(video_id):
        raise ValueError("Video ID must contain only URL-safe ID characters.")
    return video_id


def get_video_info(url: str, output_dir: Path, video_id: str) -> dict:
    """Fetch and cache video title + description from YouTube."""
    video_id = _validate_path_video_id(video_id)
    cache_path = output_dir / f"video_info_{video_id}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    with yt_dlp.YoutubeDL({"quiet": True}) as ydl:
        info = ydl.extract_info(url, download=False)

    result = {
        "title": info.get("title", ""),
        "description": (info.get("description") or "")[:2000],  # cap length
        # Duration feeds the transcript coverage check and the run cost estimate.
        # Cached files written before this field existed simply lack it.
        "duration": info.get("duration"),
    }
    cache_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def download_audio(url: str, output_dir: Path, video_id: str) -> Path:
    """Download audio only and convert it to 16 kHz mono WAV for transcription."""
    video_id = _validate_path_video_id(video_id)
    audio_path = output_dir / f"audio_{video_id}.wav"
    source_path = _find_audio_source(output_dir, video_id)

    if source_path is None:
        # Use the CLI directly so --remote-components is guaranteed to be passed.
        # The Python API doesn't reliably forward newer flags like this one.
        subprocess.run(
            [
                "yt-dlp",
                "--remote-components", "ejs:github",
                "--force-overwrites",
                "-f", "ba[ext=m4a]/ba",
                "-o", str(output_dir / f"audio_source_{video_id}.%(ext)s"),
                url,
            ],
            check=True,
        )
        source_path = _find_audio_source(output_dir, video_id)
        if source_path is None or not _has_content(source_path):
            raise FileNotFoundError(f"Could not find downloaded audio source for {video_id}")
    else:
        print("  Audio source already exists, skipping download.")

    if not _is_fresh_nonempty_file(audio_path, source_path):
        if audio_path.exists():
            audio_path.unlink()
        subprocess.run(
            [
                "ffmpeg",
                "-i", str(source_path),
                "-ar", "16000",
                "-ac", "1",
                "-c:a", "pcm_s16le",
                str(audio_path),
                "-y",
                "-hide_banner",
                "-loglevel", "error",
            ],
            check=True,
        )
        if not _has_content(audio_path):
            raise FileNotFoundError(f"Could not create extracted audio for {video_id}")
    else:
        print(f"  Audio already extracted, skipping.")

    return audio_path


def download_ocr_video(
    url: str,
    output_dir: Path,
    video_id: str,
    height: int = 480,
) -> Path:
    """Download a low-resolution video for OCR screenshot extraction."""
    video_id = _validate_path_video_id(video_id)
    if height not in (480, 720):
        raise ValueError("OCR video height must be 480 or 720.")

    video_path = output_dir / f"video_ocr_{video_id}_{height}p.mp4"
    if _has_content(video_path):
        print(f"  OCR video already exists, skipping download.")
        return video_path
    if video_path.exists():
        video_path.unlink()

    # Prefer video-only streams because screenshots do not need audio. Fall
    # back to muxed streams if YouTube does not expose a matching video-only MP4.
    format_selector = (
        f"bv*[height<={height}][ext=mp4]/"
        f"bv*[height<={height}]/"
        f"best[height<={height}][ext=mp4]/"
        f"best[height<={height}]"
    )
    subprocess.run(
        [
            "yt-dlp",
            "--remote-components", "ejs:github",
            "--force-overwrites",
            "-f", format_selector,
            "-o", str(video_path),
            "--merge-output-format", "mp4",
            url,
        ],
        check=True,
    )
    if not _has_content(video_path):
        raise FileNotFoundError(f"Could not find downloaded OCR video for {video_id}")
    return video_path


def download(
    url: str,
    output_dir: Path,
    video_id: str,
    ocr_video_height: int = 480,
) -> tuple[Path, Path]:
    """Compatibility wrapper. Returns (ocr_video_path, audio_path)."""
    audio_path = download_audio(url, output_dir, video_id)
    video_path = download_ocr_video(url, output_dir, video_id, height=ocr_video_height)
    return video_path, audio_path


def _find_audio_source(output_dir: Path, video_id: str) -> Path | None:
    candidates = sorted(output_dir.glob(f"audio_source_{video_id}.*"))
    for path in candidates:
        if path.suffix != ".part" and _has_content(path):
            return path
    return None

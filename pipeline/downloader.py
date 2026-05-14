import json
import subprocess
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import yt_dlp


def get_video_id(url: str) -> str:
    parsed = urlparse(url)
    if "youtube.com" in parsed.netloc:
        qs = parse_qs(parsed.query)
        if "v" in qs:
            return qs["v"][0]
    if "youtu.be" in parsed.netloc:
        return parsed.path.lstrip("/").split("?")[0]
    # Fallback: ask yt-dlp
    with yt_dlp.YoutubeDL({"quiet": True}) as ydl:
        info = ydl.extract_info(url, download=False)
        return info["id"]


def get_video_info(url: str, output_dir: Path, video_id: str) -> dict:
    """Fetch and cache video title + description from YouTube."""
    cache_path = output_dir / f"video_info_{video_id}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    with yt_dlp.YoutubeDL({"quiet": True}) as ydl:
        info = ydl.extract_info(url, download=False)

    result = {
        "title": info.get("title", ""),
        "description": (info.get("description") or "")[:2000],  # cap length
    }
    cache_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def download_audio(url: str, output_dir: Path, video_id: str) -> Path:
    """Download audio only and convert it to 16 kHz mono WAV for transcription."""
    audio_path = output_dir / f"audio_{video_id}.wav"
    source_path = _find_audio_source(output_dir, video_id)

    if source_path is None:
        # Use the CLI directly so --remote-components is guaranteed to be passed.
        # The Python API doesn't reliably forward newer flags like this one.
        subprocess.run(
            [
                "yt-dlp",
                "--remote-components", "ejs:github",
                "-f", "ba[ext=m4a]/ba",
                "-o", str(output_dir / f"audio_source_{video_id}.%(ext)s"),
                url,
            ],
            check=True,
        )
        source_path = _find_audio_source(output_dir, video_id)
        if source_path is None:
            raise FileNotFoundError(f"Could not find downloaded audio source for {video_id}")
    else:
        print("  Audio source already exists, skipping download.")

    if not audio_path.exists():
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
    if height not in (480, 720):
        raise ValueError("OCR video height must be 480 or 720.")

    video_path = output_dir / f"video_ocr_{video_id}_{height}p.mp4"
    if video_path.exists():
        print(f"  OCR video already exists, skipping download.")
        return video_path

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
            "-f", format_selector,
            "-o", str(video_path),
            "--merge-output-format", "mp4",
            url,
        ],
        check=True,
    )
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
        if path.suffix != ".part":
            return path
    return None

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
        return json.loads(cache_path.read_text())

    with yt_dlp.YoutubeDL({"quiet": True}) as ydl:
        info = ydl.extract_info(url, download=False)

    result = {
        "title": info.get("title", ""),
        "description": (info.get("description") or "")[:2000],  # cap length
    }
    cache_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def download(url: str, output_dir: Path, video_id: str) -> tuple[Path, Path]:
    """Download video and extract audio. Returns (video_path, audio_path)."""
    video_path = output_dir / f"video_{video_id}.mp4"
    audio_path = output_dir / f"audio_{video_id}.wav"

    if not video_path.exists():
        # Use the CLI directly so --remote-components is guaranteed to be passed.
        # The Python API doesn't reliably forward newer flags like this one.
        subprocess.run(
            [
                "yt-dlp",
                "--remote-components", "ejs:github",
                "-o", str(video_path),
                "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
                "--merge-output-format", "mp4",
                url,
            ],
            check=True,
        )
    else:
        print(f"  Video already exists, skipping download.")

    if not audio_path.exists():
        subprocess.run(
            [
                "ffmpeg",
                "-i", str(video_path),
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

    return video_path, audio_path

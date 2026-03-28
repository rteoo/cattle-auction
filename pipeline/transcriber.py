"""
Transcription backends:

  mlx   — mlx-whisper on Apple Silicon via Metal (default, requires: uv sync --extra local)
  cpp   — whisper.cpp via the whisper-cli binary (requires: brew install whisper-cpp)
  groq  — Groq cloud API, Whisper Large v3 Turbo ($0.04/hr, ~$0.20 for 5h video)
            Requires: GROQ_API_KEY env var. Audio is split into <20 MB chunks automatically.
"""

import json
import os
import subprocess
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path

from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


@dataclass
class Segment:
    start: float
    end: float
    text: str


# ── Public entry point ────────────────────────────────────────────────────────

def transcribe(
    audio_path: Path,
    output_path: Path,
    backend: str = "mlx",
    whisper_model: str = "medium",
    cpp_model_path: Path | None = None,
) -> list[Segment]:
    """Transcribe audio in PT-BR. Saves/loads output_path for checkpointing."""
    if output_path.exists():
        print(f"  Transcript already exists, loading from cache.")
        return _load(output_path)

    print(f"  Transcribing with backend={backend!r}, model={whisper_model!r}...")

    if backend == "mlx":
        segments = _transcribe_mlx(audio_path, whisper_model)
    elif backend == "cpp":
        segments = _transcribe_cpp(audio_path, whisper_model, cpp_model_path)
    elif backend == "groq":
        segments = _transcribe_groq(audio_path)
    else:
        raise ValueError(f"Unknown transcription backend: {backend!r}. Choose mlx, cpp, or groq.")

    print(f"  Transcribed {len(segments)} segments.")
    _save(segments, output_path)
    return segments


# ── Backend: MLX (Apple Silicon, Metal-accelerated) ──────────────────────────

def _transcribe_mlx(audio_path: Path, model: str = "medium") -> list[Segment]:
    try:
        import mlx_whisper
    except ImportError:
        raise ImportError(
            "mlx-whisper is not installed. Run:\n"
            "  uv sync --extra local"
        )

    model_repo = f"mlx-community/whisper-{model}-mlx"
    print(f"  Using MLX model: {model_repo}")

    result_holder: list = [None]
    error_holder: list = [None]

    def _run():
        try:
            result_holder[0] = mlx_whisper.transcribe(
                str(audio_path),
                path_or_hf_repo=model_repo,
                language="pt",
                verbose=False,
            )
        except Exception as e:
            error_holder[0] = e

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    with Progress(
        SpinnerColumn(),
        TextColumn("  [progress.description]{task.description}"),
        TimeElapsedColumn(),
    ) as progress:
        progress.add_task("MLX Whisper transcribing...", total=None)
        thread.join()

    if error_holder[0]:
        raise error_holder[0]

    return [
        Segment(start=s["start"], end=s["end"], text=s["text"].strip())
        for s in result_holder[0]["segments"]
    ]


# ── Backend: whisper.cpp (Metal-accelerated, requires brew install whisper-cpp) ─

def _transcribe_cpp(
    audio_path: Path,
    model: str = "medium",
    model_path: Path | None = None,
) -> list[Segment]:
    if model_path is None:
        model_path = _find_cpp_model(model)

    with tempfile.TemporaryDirectory() as tmpdir:
        out_base = Path(tmpdir) / "out"

        proc = subprocess.Popen(
            [
                "whisper-cli",
                "-m", str(model_path),
                "-l", "pt",
                "-f", str(audio_path),
                "--output-json",
                "-of", str(out_base),
                "--threads", "8",
                "--print-progress",
            ],
            stderr=subprocess.PIPE,
            text=True,
        )

        with Progress(
            TextColumn("  [progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task("whisper.cpp transcribing...", total=100)
            for line in proc.stderr:
                if "progress =" in line:
                    try:
                        pct = int(line.split("progress =")[1].strip().rstrip("%"))
                        progress.update(task, completed=pct)
                    except (ValueError, IndexError):
                        pass

        proc.wait()
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, "whisper-cli")

        data = json.loads(out_base.with_suffix(".json").read_text())

    # whisper.cpp JSON: {"transcription": [{"timestamps": {"from": "HH:MM:SS,mmm"}, "text": "..."}]}
    segments = []
    for entry in data.get("transcription", []):
        start = _parse_cpp_timestamp(entry["timestamps"]["from"])
        end = _parse_cpp_timestamp(entry["timestamps"]["to"])
        segments.append(Segment(start=start, end=end, text=entry["text"].strip()))
    return segments


def _find_cpp_model(model: str) -> Path:
    candidates = [
        Path.home() / ".cache" / "whisper" / f"ggml-{model}.bin",
        Path(f"/opt/homebrew/share/whisper-cpp/ggml-{model}.bin"),
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"whisper.cpp model 'ggml-{model}.bin' not found.\n"
        f"Download it with:\n"
        f"  brew install whisper-cpp\n"
        f"  whisper-cpp-download-ggml-model {model}\n"
        f"Or pass --cpp-model /path/to/ggml-{model}.bin"
    )


def _parse_cpp_timestamp(ts: str) -> float:
    # Format: "HH:MM:SS,mmm"
    h, m, rest = ts.split(":")
    s, ms = rest.split(",")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


# ── Backend: Groq cloud API (Whisper Large v3 Turbo) ─────────────────────────

_GROQ_MAX_BYTES = 20 * 1024 * 1024   # 20 MB safe limit (API limit is 25 MB)
_GROQ_MODEL = "whisper-large-v3-turbo"
_CHUNK_MINUTES = 15                   # minutes per chunk when splitting


def _transcribe_groq(audio_path: Path) -> list[Segment]:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY environment variable is not set.")

    from groq import Groq
    client = Groq(api_key=api_key)

    # Convert to MP3 at 32kbps to keep file sizes small
    mp3_path = audio_path.with_suffix(".mp3")
    if not mp3_path.exists():
        print("  Converting audio to MP3 for Groq upload...")
        subprocess.run(
            [
                "ffmpeg", "-i", str(audio_path),
                "-codec:a", "libmp3lame", "-b:a", "32k",
                str(mp3_path), "-y", "-hide_banner", "-loglevel", "error",
            ],
            check=True,
        )

    if mp3_path.stat().st_size <= _GROQ_MAX_BYTES:
        with Progress(
            SpinnerColumn(),
            TextColumn("  [progress.description]{task.description}"),
            TimeElapsedColumn(),
        ) as progress:
            progress.add_task("Groq transcribing (single request)...", total=None)
            segments = _groq_call(client, mp3_path, offset=0)
        return segments

    # File is too large — split into chunks
    duration = _audio_duration(mp3_path)
    chunk_secs = _CHUNK_MINUTES * 60
    total_chunks = -(-int(duration) // chunk_secs)  # ceiling div

    all_segments: list[Segment] = []

    with Progress(
        TextColumn("  [progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("[cyan]{task.fields[info]}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task(
            f"Groq transcribing ({total_chunks} chunks)...",
            total=total_chunks,
            info="",
        )

        for idx, offset in enumerate(range(0, int(duration), chunk_secs), 1):
            chunk_path = audio_path.parent / f"chunk_{idx:03d}.mp3"
            if not chunk_path.exists():
                subprocess.run(
                    [
                        "ffmpeg", "-i", str(mp3_path),
                        "-ss", str(offset), "-t", str(chunk_secs),
                        "-codec:a", "copy",
                        str(chunk_path), "-y", "-hide_banner", "-loglevel", "error",
                    ],
                    check=True,
                )
            progress.update(task, info=f"chunk {idx}/{total_chunks} (offset {offset}s)")
            segments = _groq_call(client, chunk_path, offset=offset)
            all_segments.extend(segments)
            progress.update(task, advance=1)

    return all_segments


def _groq_call(client, audio_path: Path, offset: float) -> list[Segment]:
    with open(audio_path, "rb") as f:
        response = client.audio.transcriptions.create(
            file=(audio_path.name, f.read()),
            model=_GROQ_MODEL,
            language="pt",
            response_format="verbose_json",
            timestamp_granularities=["segment"],
        )
    return [
        Segment(
            start=seg["start"] + offset,
            end=seg["end"] + offset,
            text=seg["text"].strip(),
        )
        for seg in response.segments
    ]


def _audio_duration(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_streams", str(path),
        ],
        capture_output=True, text=True, check=True,
    )
    data = json.loads(result.stdout)
    for stream in data.get("streams", []):
        if stream.get("codec_type") == "audio":
            return float(stream["duration"])
    raise RuntimeError(f"Could not determine duration of {path}")


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def _save(segments: list[Segment], path: Path) -> None:
    data = [{"start": s.start, "end": s.end, "text": s.text} for s in segments]
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))


def _load(path: Path) -> list[Segment]:
    data = json.loads(path.read_text())
    return [Segment(**d) for d in data]

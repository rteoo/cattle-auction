# Cattle Auction Extractor

Extracts structured lot data from Brazilian cattle auction YouTube videos.

## How it works

1. **Download** — fetches the video with `yt-dlp` and extracts a 16kHz mono audio track
2. **Transcribe** — transcribes audio in PT-BR using MLX Whisper (local, Metal), whisper.cpp (local, Metal), or Groq API (cloud)
3. **Screenshots** — extracts one frame every 30 seconds with `ffmpeg`, with a live progress bar
4. **OCR** — reads text visible on screen using RapidOCR (ONNX-based, fast, no native deps), with a live progress bar
5. **Aggregate** — merges transcript segments and OCR results into 10-minute windows
6. **Extract lots** — sends each window to an LLM with a structured PT-BR prompt to pull out lot data (number, sex, category, count, breed, price, sold status)
7. **Extract metadata** — scans the first windows to extract auction-level info: date, city, auctioneer, farm, auction type
8. **Output** — saves `lots_<video_id>.json`, `metadata_<video_id>.json`, and prints a summary table

Each stage is checkpointed. Interrupted runs resume automatically from where they left off.

## Requirements

- Python 3.11+
- `uv` for dependency management
- `ffmpeg` installed on the system

```bash
brew install ffmpeg
```

## Setup

```bash
git clone <repo>
cd cattle-auction
uv sync --no-install-project                        # base deps
uv sync --extra local --no-install-project          # + mlx-whisper (Apple Silicon only)
```

Fill in your API keys in `.env`:

```bash
# .env
ANTHROPIC_API_KEY=sk-...
OPENAI_API_KEY=sk-...
GROQ_API_KEY=gsk-...
OPENROUTER_API_KEY=sk-or-...
```

Keys are loaded automatically from `.env` on every run. Only set the keys you need — the defaults use OpenAI, so `OPENAI_API_KEY` is the minimum required.

## Usage

```bash
uv run python main.py <youtube_url> [OPTIONS]
```

### Transcription backends

| Backend | Flag | Speed | Cost |
|---|---|---|---|
| MLX Whisper | `--transcriber mlx` | Fast (Metal, Apple Silicon) | Free |
| whisper.cpp | `--transcriber cpp` | Fast (Metal, Apple Silicon) | Free |
| Groq API | `--transcriber groq` | ~228× realtime | ~$0.20 / 5h video |

**whisper.cpp setup** (if using `--transcriber cpp`):
```bash
brew install whisper-cpp
whisper-cpp-download-ggml-model medium
```

### Options

| Flag | Default | Description |
|---|---|---|
| `--transcriber` | `groq` | Transcription backend: `mlx`, `cpp`, or `groq` |
| `--whisper-model` | `medium` | Model size for mlx/cpp: `tiny` / `base` / `small` / `medium` / `large-v3` |
| `--cpp-model` | auto | Path to ggml model file (whisper.cpp only) |
| `--provider` | `openai` | LLM provider: `openai`, `claude`, or `openrouter` |
| `--model` | `gpt-4o-mini` | Model name or alias (e.g. `gemini-2.5-flash-lite`) |
| `--screenshot-interval` | `30` | Seconds between captured frames |
| `--output-dir` | `output` | Base directory for all generated files |
| `--no-resume` | off | Ignore cached stages and rerun everything |
| `--metadata / --no-metadata` | on | Display auction metadata (date, city, auctioneer, farm, type) |
| `--summary / --no-summary` | on | Display summary statistics (totals, averages, counts by category) |
| `--table / --no-table` | on | Display full table of all lots with detailed information |

### Examples

```bash
# Default: All outputs (metadata, summary, table)
uv run python main.py "https://www.youtube.com/watch?v=..."

# Local MLX transcription (Apple Silicon only, requires --extra local)
uv run python main.py "https://www.youtube.com/watch?v=..." --transcriber mlx

# OpenRouter with Gemini (alias)
uv run python main.py "https://www.youtube.com/watch?v=..." \
  --provider openrouter --model gemini-2.5-flash-lite

# Claude extraction
uv run python main.py "https://www.youtube.com/watch?v=..." --provider claude

# Show only metadata and summary (no table)
uv run python main.py "https://www.youtube.com/watch?v=..." --no-table

# Show only the count (metadata/summary, no table)
uv run python main.py "https://www.youtube.com/watch?v=..." --no-table

# Show only the table (no metadata or summary)
uv run python main.py "https://www.youtube.com/watch?v=..." --no-metadata --no-summary

# Force full rerun (ignore all cached stages)
uv run python main.py "https://www.youtube.com/watch?v=..." --no-resume
```

## Output

All files are written to `output/<video_id>/`:

| File | Contents |
|---|---|
| `video_<video_id>.mp4` | Downloaded video |
| `audio_<video_id>.wav` | 16kHz mono audio for Whisper |
| `transcript_<video_id>.json` | Timestamped transcript segments |
| `screenshots_<video_id>/` | JPEG frames at every N seconds |
| `screenshots_<video_id>.json` | Index of frames with timestamps |
| `ocr_results_<video_id>.json` | Screen text per timestamp |
| `lots_<video_id>.json` | Extracted lots (array) |
| `metadata_<video_id>.json` | Auction metadata (date, city, auctioneer, farm, type) |
| `result_<video_id>.json` | Final result with metadata and lots |

### Lot schema

```json
{
  "lot_number": 12,
  "sex": "macho",
  "category": "garrote",
  "num_animals": 30,
  "age_months": 18,
  "breed": "Nelore",
  "unit_price": 3200.00,
  "total_price": null,
  "sold": true,
  "timestamp_start": "01:24:35",
  "notes": null
}
```

## Estimated run times (5-hour video, Apple Silicon M2)

| Stage | mlx / cpp | groq |
|---|---|---|
| Download | ~5–15 min (network) | same |
| Transcribe | ~20–40 min (medium) | ~1–2 min ($0.20) |
| Screenshots (30s interval) | ~2–3 min | same |
| OCR (~600 frames) | ~5–10 min | same |
| LLM extraction (~30 windows) | ~3–8 min | same |

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

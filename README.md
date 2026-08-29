# Cattle Auction Extractor

Extracts structured lot data from Brazilian cattle auction YouTube videos.

## How it works

1. **Download audio** — fetches audio only with `yt-dlp` and extracts a 16kHz mono audio track
2. **Transcribe** — transcribes audio in PT-BR using MLX Whisper (local, Metal), whisper.cpp (local, Metal), or Groq API (cloud), then gates the result for known Whisper hallucinations (caption-credit boilerplate, repeated-phrase loops, mostly-silent audio)
3. **Download OCR video** — fetches a low-resolution video for screenshots, 480p by default or 720p when requested
4. **Screenshots** — extracts frames with `ffmpeg`, with a live progress bar. By default one frame every 30 seconds; with `--frame-sampling scene` it instead captures the moments the lot board actually changes, plus a coarse safety grid
5. **OCR** — reads text visible on screen using RapidOCR (ONNX-based, fast, no native deps), with a live progress bar
6. **Aggregate** — merges transcript segments and OCR results into 10-minute windows
7. **Extract lots** — sends each window to an LLM with a structured PT-BR prompt to pull out lot data (number, sex, category, count, breed, price, sold status)
8. **Extract metadata** — scans the first windows to extract auction-level info: date, city, auctioneer, farm, auction type
9. **Output** — saves `lots_<video_id>.json`, `metadata_<video_id>.json`, and prints a summary table plus the run's estimated USD cost

Each stage is checkpointed. Interrupted runs resume automatically from where they left off.

## Requirements

- Python 3.11+
- `uv` for dependency management
- `ffmpeg` installed on the system
- `deno` runtime (used by yt-dlp for YouTube download)

```bash
# macOS
brew install ffmpeg deno

# Windows
winget install Gyan.FFmpeg DenoLand.Deno
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
OPENAI_API_KEY=sk-...
GROQ_API_KEY=gsk-...
OPENROUTER_API_KEY=sk-or-...   # only if using --provider openrouter
```

Keys are loaded automatically from `.env` on every run. Only set the keys you need. The default run uses OpenRouter for extraction and Groq for transcription, so `OPENROUTER_API_KEY` and `GROQ_API_KEY` are required for the default path. Use `OPENAI_API_KEY` only when running with `--provider openai`.

## Usage

```bash
uv run python main.py <youtube_url> [OPTIONS]
uv run python main.py <youtube_url_1> <youtube_url_2> [OPTIONS]
uv run python main.py --batch-file links.txt --batch-name maio-2026 [OPTIONS]
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
| `--provider` | `openrouter` | LLM provider: `openrouter` or `openai` |
| `--screenshot-interval` | `30` | Seconds between captured frames (interval sampling) |
| `--frame-sampling` | `interval` | `interval` samples on a fixed clock; `scene` samples where the lot board changes |
| `--safety-interval` | `60` | Scene sampling only: seconds between safety-grid frames added on top of detections |
| `--ocr-video-height` | `480` | Maximum video height for OCR screenshots: `480` or `720` |
| `--output-dir` | `output` | Base directory for all generated files |
| `--no-resume` | off | Recompute derived checkpoints; downloaded source media is preserved |
| `--metadata / --no-metadata` | on | Display auction metadata (date, city, auctioneer, farm, type) |
| `--summary / --no-summary` | on | Display summary statistics (totals, averages, counts by category) |
| `--table / --no-table` | on | Display full table of all lots with detailed information |
| `--batch-file` | off | Text file with one YouTube URL per line; blank lines and `#` comments are ignored |
| `--batch-name` | timestamp | Name for the saved batch report folder under `output/batches/` |
| `--stop-on-error` | off | In batch mode, stop after the first failed URL instead of continuing |

### Examples

```bash
# Default: OpenRouter Gemini 2.5 Flash-Lite extraction + Groq transcription
uv run python main.py "https://www.youtube.com/watch?v=..."

# Local MLX transcription (Apple Silicon only, requires --extra local)
uv run python main.py "https://www.youtube.com/watch?v=..." --transcriber mlx

# OpenRouter extraction (default provider)
uv run python main.py "https://www.youtube.com/watch?v=..." --provider openrouter

# OpenAI extraction alternative
uv run python main.py "https://www.youtube.com/watch?v=..." --provider openai

# Show only metadata and summary (no table)
uv run python main.py "https://www.youtube.com/watch?v=..." --no-table

# Show only the table (no metadata or summary)
uv run python main.py "https://www.youtube.com/watch?v=..." --no-metadata --no-summary

# Use higher-resolution 720p video for OCR screenshots
uv run python main.py "https://www.youtube.com/watch?v=..." --ocr-video-height 720

# Sample frames where the lot board changes instead of on a fixed clock
uv run python main.py "https://www.youtube.com/watch?v=..." --frame-sampling scene

# Run a batch from a text file, one URL per line
uv run python main.py --batch-file links.txt --batch-name maio-2026 --no-table

# Run a small batch directly from the command line
uv run python main.py "https://www.youtube.com/watch?v=..." "https://youtu.be/..." --no-table

# Recompute all derived checkpoints (downloaded source media is preserved)
uv run python main.py "https://www.youtube.com/watch?v=..." --no-resume
```

Batch mode runs URLs sequentially. Each video keeps its normal checkpoint folder at `output/<video_id>/`; after the batch finishes, the CLI prints a comparison table and writes `output/batches/<batch_name>/batch_summary.json` plus `output/batches/<batch_name>/comparison.md`. Batch mode continues after failed URLs by default, records the error in the report, and exits non-zero if any item failed.

## Output

All files are written to `output/<video_id>/`:

| File | Contents |
|---|---|
| `audio_source_<video_id>.<ext>` | Downloaded audio-only source |
| `audio_<video_id>.wav` | 16kHz mono audio for Whisper |
| `video_ocr_<video_id>_480p.mp4` | Default low-resolution video for OCR screenshots |
| `video_ocr_<video_id>_720p.mp4` | Optional higher-resolution OCR video when `--ocr-video-height 720` is used |
| `transcript_<video_id>.json` | Timestamped transcript segments |
| `screenshots_<video_id>/` | JPEG frames at every N seconds |
| `screenshots_<video_id>.json` | Index of frames with timestamps |
| `ocr_results_<video_id>.json` | Screen text per timestamp |
| `lots_<video_id>.json` | Extracted lots (array) |
| `metadata_<video_id>.json` | Auction metadata (date, city, auctioneer, farm, type) |
| `result_<video_id>.json` | Final result with metadata and lots |
| `batches/<batch_name>/batch_summary.json` | Batch totals, per-video rows, failure records, comparison winners |
| `batches/<batch_name>/comparison.md` | Human-readable batch summary and comparison table |

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
| Download audio | network-dependent; audio only | same |
| Transcribe | ~20–40 min (medium) | ~1–2 min ($0.20) |
| Download OCR video | network-dependent; defaults to 480p | same |
| Screenshots (30s interval) | ~2–3 min | same |
| OCR (~600 frames) | ~5–10 min | same |
| LLM extraction (~30 windows) | ~3–8 min | same |

## LLM providers

| Provider | Flag | Default model | Auth |
|---|---|---|---|
| OpenRouter | `--provider openrouter` | `google/gemini-2.5-flash-lite-preview-09-2025` | `OPENROUTER_API_KEY` |
| OpenAI | `--provider openai` | `gpt-4.1-mini` | `OPENAI_API_KEY` |

## Model benchmark

The shipped model catalog is intentionally narrow and benchmark-driven:

| Provider | Model | Cost/video | Speed | Coverage | Accuracy (MAPE) |
|---|---|---:|---:|---:|---:|
| **openrouter** (default) | `google/gemini-2.5-flash-lite-preview-09-2025` | ~$0.05 | 13-24s | 92-100% | 1.9-2.0% |
| openai (alt) | `gpt-4.1-mini` | ~$0.13 | 31s | 100% | 0.1% |

Use `bench/` for the current benchmark harness. `benchmark.py` is a single-video comparison script that now targets the same two shipping models.

## Testing

The project includes a comprehensive unit test suite with 139 tests covering:

- **Model validation** — `Lot` and `AuctionResult` data validation, Brazilian number format coercion, price mis-parsing guards, required field checks
- **LLM response parsing** — JSON extraction with extra-text tolerance, lot merging, sold field detection
- **Data aggregation** — Window overlap logic, transcript + OCR merging, broadcast clock filtering, empty window placeholders
- **Summary statistics** — Animal counts by category and sex, average prices by category, sold/unsold tracking
- **Batch reports** — URL-file loading, sequential batch runs, saved totals, failure records, comparison winners
- **Downloader format selection** — Audio-only transcription source, 480p default OCR video, 720p OCR alternative

Run tests:

```bash
uv run pytest tests/ -v              # Run all tests with verbose output
uv run pytest tests/test_lot_model.py -v  # Run model tests only
```

All tests are pure unit tests with no external dependencies: no API calls, live video downloads, or large fixtures. Batch tests use temporary files for URL/report handling only.

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

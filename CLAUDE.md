# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Python CLI pipeline that extracts structured lot data from Brazilian cattle auction YouTube videos. It downloads the video, transcribes the audio with Whisper (PT-BR), takes screenshots every 30s and runs OCR on them, then sends aggregated transcript+screen data to an LLM to extract lot information.

## Setup

```bash
uv sync --no-install-project          # base deps (groq, rapidocr, etc.)
uv sync --extra local --no-install-project  # + mlx-whisper (Apple Silicon only)
```

Requires `ffmpeg` installed on the system (`brew install ffmpeg`).

API keys are loaded automatically from `.env` in the project root:
```
ANTHROPIC_API_KEY=
OPENAI_API_KEY=
GROQ_API_KEY=
OPENROUTER_API_KEY=
```

## Running

```bash
uv run python main.py <youtube_url> [OPTIONS]

# Options:
#   --provider      openai|openrouter|ollama    (default: openai)
#   --model         model name or alias        (default: gpt-4.1-nano / google/gemma-4-31b-it:free / qwen3.5:397b-cloud)
#                   OpenAI models: gpt-4.1-nano, gpt-4o-mini, gpt-5-nano
#                   alias: gemini-2.5-flash-lite -> google/gemini-2.5-flash-lite-preview-09-2025
#   --transcriber        mlx|cpp|groq               (default: groq)
#   --whisper-model      medium                     (default: medium, used by mlx and cpp)
#   --cpp-model          /path/to/ggml.bin         (whisper.cpp only, auto-detected if omitted)
#   --screenshot-interval 30                      (seconds between frames)
#   --no-resume                                   (ignore cached stage outputs)
#   --metadata / --no-metadata                    (default: on, show auction metadata header)
#   --summary / --no-summary                      (default: on, show summary statistics)
#   --table / --no-table                          (default: on, show full lots table)
#   --output-dir         output                    (base directory)
```

## Transcription backends

| Backend | Flag | Speed | Cost | Notes |
|---|---|---|---|---|
| MLX Whisper | `--transcriber mlx` | Fast (Metal) | Free | Apple Silicon only; requires `--extra local`; shows spinner |
| whisper.cpp | `--transcriber cpp` | Fast (Metal) | Free | Requires `brew install whisper-cpp` + model download; shows real % progress bar |
| Groq API | `--transcriber groq` | ~228× realtime | ~$0.20/5h video | Requires `GROQ_API_KEY`; audio chunked to <20 MB; shows chunk progress bar |

**whisper.cpp model setup:**
```bash
brew install whisper-cpp
whisper-cpp-download-ggml-model medium
```

## Pipeline stages and checkpoints

Each stage writes a checkpoint file. On rerun, if the file exists the stage is skipped automatically (unless `--no-resume` is passed).

| Stage | Output file |
|---|---|
| Download | `output/<id>/video_<id>.mp4`, `audio_<id>.wav` |
| Transcribe | `output/<id>/transcript_<id>.json` |
| Screenshots | `output/<id>/screenshots_<id>.json` + `screenshots_<id>/` dir |
| OCR | `output/<id>/ocr_results_<id>.json` |
| Extraction | `output/<id>/lots_<id>.json` |
| Metadata | `output/<id>/metadata_<id>.json` |
| Final result | `output/<id>/result_<id>.json` |

## Architecture

- `main.py` — CLI entry point, orchestrates stages, renders summary table; loads `.env` via `python-dotenv`
- `pipeline/downloader.py` — yt-dlp CLI download (uses `--remote-components ejs:github` + Deno for YouTube n-challenge) + ffmpeg audio extraction
- `pipeline/transcriber.py` — three backends: Groq API (default, chunk progress bar), MLX Whisper (thread + spinner), whisper.cpp (Popen + parse `--print-progress` stderr); Groq converts audio to 32kbps MP3 and splits into 15-min chunks when file exceeds 20 MB
- `pipeline/screenshotter.py` — ffmpeg frame extraction every N seconds; streams `-progress pipe:1` to show a Rich progress bar
- `pipeline/ocr.py` — RapidOCR (ONNX-based) on all screenshots with Rich progress bar; no native PaddlePaddle dependency
- `pipeline/aggregator.py` — merges transcript segments + OCR into 10-minute overlapping windows
- `pipeline/extractor.py` — sends windows to LLM, parses JSON response, deduplicates by lot_number; supports model aliases via `_MODEL_ALIASES`; also runs `extract_metadata()` on first 3 windows to extract date/city/auctioneer/farm/type
- `models/lot.py` — Pydantic models: `Lot`, `AuctionResult`
- `prompts/extraction.txt` — PT-BR system prompt for lot extraction
- `prompts/metadata.txt` — PT-BR system prompt for auction metadata extraction (date, city, auctioneer, farm, type)

## LLM providers

| Provider | Flag | Auth | Default model | Notes |
|---|---|---|---|---|
| OpenAI | `--provider openai` | `OPENAI_API_KEY` | gpt-4.1-nano | Default provider |
| OpenRouter | `--provider openrouter` | `OPENROUTER_API_KEY` | google/gemma-4-31b-it:free | OpenAI-compatible SDK, custom base URL |
| Ollama | `--provider ollama` | — | qwen3.5:397b-cloud | Local/cloud via http://127.0.0.1:11434 |

Model aliases resolved in `extractor._MODEL_ALIASES`:
- `gemini-2.5-flash-lite` -> `google/gemini-2.5-flash-lite-preview-09-2025`

## LLM extraction details

- Windows are 10 minutes with 1-minute overlap to avoid splitting lots across boundaries
- Each window sends already-found lot numbers so the LLM skips duplicates
- If the same lot number appears in multiple windows, fields are merged: non-price fields use first-non-null; price fields (`unit_price`, `total_price`) use last-non-null so the final hammer price overwrites an opening ask
- `_parse_response()` in `extractor.py` tolerates LLM responses with extra text around the JSON array

## Key data model

```python
class Lot(BaseModel):
    lot_number: int
    sex: str           # macho / fêmea / misto
    category: str      # bezerro, novilha, garrote, boi, vaca, touro, etc.
    num_animals: int
    age_months: int | None
    breed: str         # Nelore, Anelorado, Mestiço, Gir, etc.
    unit_price: float | None
    total_price: float | None
    sold: bool | None  # True = arrematado, False = retirado/não vendido, None = indefinido
    timestamp_start: str | None   # HH:MM:SS
    notes: str | None
```

## Testing

### Test suite structure

- `tests/test_lot_model.py` — 21 tests for `Lot` + `AuctionResult` model validation, Brazilian number format coercion, `sold` field, required fields
- `tests/test_extractor.py` — 18 tests for `_validate_lots()`, `_parse_response()` (extra-text tolerance), `_merge()` (first-non-null-wins, sold=False preservation)
- `tests/test_aggregator.py` — 18 tests for `_fmt()`, `_parse_ts()` round-trip, `aggregate()` window logic (overlap, OCR, empty placeholders)
- `tests/test_summary.py` — 20 tests for `_calculate_summary()` (totals, sex counts, top categories, avg prices, sold counts)

### Running tests

```bash
uv run pytest tests/ -v                # All tests (77 total)
uv run pytest tests/test_lot_model.py  # Model validation only
uv run pytest tests/ -k test_br_       # Specific pattern (e.g., BR number format tests)
```

All tests are pure unit tests — no external API calls, file I/O, or fixture dependencies. They validate the core data transformation logic:
- Brazilian number format parsing (3.100 = 3100.00, not 3.10)
- Quantity vs lot number disambiguation
- Sold status detection from LLM responses
- Window aggregation with overlap preservation
- Summary statistics computation

## License

This project is released under the MIT License. Dependencies are compatible (MIT, Apache 2.0, BSD).

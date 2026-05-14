# AGENTS.md

This is the operating guide for GPT-5.5/Codex agents working in this repository. Treat it as the repo-specific source of truth for how to reason about changes, run validation, and avoid damaging expensive pipeline state.

## Project Snapshot

`cattle-auction` is a Python 3.11 CLI pipeline for extracting structured lot data from Brazilian cattle auction YouTube videos.

The pipeline:

1. Downloads YouTube audio only with `yt-dlp`.
2. Extracts 16 kHz mono audio with `ffmpeg`.
3. Transcribes PT-BR audio with Groq Whisper, MLX Whisper, or whisper.cpp.
4. Downloads a low-resolution OCR video, 480p by default or 720p when requested.
5. Extracts screenshots every N seconds.
6. Runs OCR over screenshots with RapidOCR.
7. Merges transcript and OCR into overlapping time windows.
8. Sends each window to an LLM to extract lot records.
9. Extracts auction metadata.
10. Writes checkpointed JSON artifacts and prints Rich summary/table output.

The domain is Brazilian cattle auctions. Preserve PT-BR terminology, Brazilian number formats, and cattle-specific extraction rules unless there is strong test-backed evidence to change them.

## Source Of Truth

Use this priority order when docs disagree:

1. Current code and tests.
2. This `AGENTS.md`.
3. `README.md`.
4. Historical notes, comments, or benchmark writeups.

The current CLI in `main.py` supports one URL, multiple positional URLs, or `--batch-file`. It supports `--provider openrouter|openai` only, defaults to `openrouter`, and does not expose a generic `--model` or `ollama` option.

## Setup

```bash
uv sync --no-install-project
uv sync --extra local --no-install-project  # optional, Apple Silicon MLX Whisper
```

System tools:

```bash
brew install ffmpeg deno
brew install whisper-cpp                    # only for --transcriber cpp
whisper-cpp-download-ggml-model medium      # only for --transcriber cpp
```

Environment variables are loaded from `.env` by `main.py` through `python-dotenv`:

```bash
OPENROUTER_API_KEY=
OPENAI_API_KEY=
GROQ_API_KEY=
```

`ANTHROPIC_API_KEY` may exist in local env files, but the current code does not use Anthropic.

## Run Commands

Primary CLI:

```bash
uv run python main.py <youtube_url> [OPTIONS]
uv run python main.py <youtube_url_1> <youtube_url_2> [OPTIONS]
uv run python main.py --batch-file links.txt --batch-name maio-2026 [OPTIONS]
```

Important options:

```bash
--provider openrouter|openai       # default: openrouter
--transcriber mlx|cpp|groq         # default: groq
--whisper-model medium             # used by mlx/cpp
--cpp-model /path/to/ggml.bin      # optional whisper.cpp model path
--screenshot-interval 30           # seconds between frames
--ocr-video-height 480             # OCR video height: 480 or 720
--output-dir output
--no-resume                        # clear stage checkpoints before running
--metadata / --no-metadata
--summary / --no-summary
--table / --no-table
--batch-file links.txt              # one YouTube URL per line; blank/# lines ignored
--batch-name maio-2026              # saved under output/batches/<name>/
--stop-on-error                     # batch mode stops at first failed URL
```

Batch mode runs URLs sequentially. Each video still writes its normal artifacts under `output/<video_id>/`; the batch-level report is saved under `output/batches/<batch_name>/batch_summary.json` with a companion `comparison.md`. Batch mode continues after failed URLs by default, records the error in the report, and exits non-zero if any item failed.

Default LLM models are resolved in `pipeline/extractor.py`:

```python
_DEFAULT_MODELS = {
    "openrouter": "google/gemini-2.5-flash-lite-preview-09-2025",
    "openai": "gpt-4.1-mini",
}
```

## Validation

Run focused tests for normal code changes:

```bash
uv run pytest tests/ -v
```

Useful targeted runs:

```bash
uv run pytest tests/test_lot_model.py -v
uv run pytest tests/test_extractor.py -v
uv run pytest tests/test_aggregator.py -v
uv run pytest tests/test_summary.py -v
uv run pytest tests/test_downloader.py -v
uv run pytest tests/ -k test_br_ -v
```

The unit tests are pure and should not call external APIs, download videos, or require large fixture files. If a change needs live YouTube, ffmpeg, OCR, Groq, OpenRouter, or OpenAI validation, state that explicitly and keep it separate from the unit test signal.

## Checkpoint Discipline

Pipeline stages write resumable artifacts under `output/<video_id>/`.

| Stage | Artifacts |
|---|---|
| Video info | `video_info_<id>.json` |
| Audio source | `audio_source_<id>.<ext>`, `audio_<id>.wav` |
| OCR video | `video_ocr_<id>_480p.mp4`, `video_ocr_<id>_720p.mp4` when requested |
| Transcribe | `transcript_<id>.json` |
| Screenshots | `screenshots_<id>.json`, `screenshots_<id>/` |
| OCR | `ocr_results_<id>.json` |
| Lot extraction | `lots_<id>.json` |
| Metadata | `metadata_<id>.json` |
| Final result | `result_<id>.json` |
| Batch summary | `batches/<batch_name>/batch_summary.json`, `batches/<batch_name>/comparison.md` |

Do not delete or regenerate `output/` artifacts casually. Full runs can be slow and may incur API cost. Use `--no-resume` only when the task explicitly requires invalidating cached stage outputs.

## Architecture Map

```text
cattle-auction/
├── AGENTS.md                 ← agent operating guide
├── README.md                 ← user-facing overview, may lag code
├── main.py                   ← Click CLI, stage orchestration, Rich output
├── models/
│   └── lot.py                ← Pydantic Lot and AuctionResult models
├── pipeline/
│   ├── downloader.py         ← YouTube metadata/download + ffmpeg audio extraction
│   ├── transcriber.py        ← Groq, MLX Whisper, and whisper.cpp backends
│   ├── screenshotter.py      ← ffmpeg frame extraction with progress
│   ├── ocr.py                ← RapidOCR screenshot text extraction
│   ├── aggregator.py         ← transcript/OCR merge into overlapping windows
│   └── extractor.py          ← LLM clients, JSON parsing, merge, sanity checks
├── prompts/
│   ├── extraction.txt        ← PT-BR lot extraction prompt
│   ├── metadata.txt          ← PT-BR metadata extraction prompt
│   └── verify.txt            ← outlier price verification prompt
└── tests/
    ├── test_lot_model.py     ← model validation and BR price coercion
    ├── test_extractor.py     ← response parsing, merge, sanity checks
    ├── test_aggregator.py    ← time windows and OCR/transcript merging
    ├── test_summary.py       ← summary statistics
    ├── test_batch.py         ← batch URL loading, sequential runs, reports
    └── test_downloader.py    ← audio-only download and OCR video resolution
```

## Core Data Model

`models/lot.py` defines the durable schema:

```python
class Lot(BaseModel):
    lot_number: int
    sex: str
    category: str
    num_animals: int
    age_months: int | None = None
    breed: str
    unit_price: float | None = None
    total_price: float | None = None
    sold: bool | None = None
    timestamp_start: str | None = None
    notes: str | None = None
```

Important invariants:

- `lot_number`, `sex`, `category`, `num_animals`, and `breed` are required.
- `category` is normalized for common plurals such as `garrotes -> garrote`.
- Brazilian price strings use dot as thousands separator and comma as decimal separator.
- Numeric prices below 100 are usually LLM mis-parses of Brazilian thousands formatting and are multiplied by 1000.
- `sold=True` means arrematado; `sold=False` means not sold/withdrawn; `sold=None` means unknown.

## Extraction Rules

`pipeline/aggregator.py` creates 10-minute windows with 1-minute overlap. The overlap is intentional and protects lots that span a boundary.

`pipeline/extractor.py` is deliberately defensive:

- `_parse_response()` accepts a JSON array even if the LLM wraps it in extra text.
- Each prompt includes already-found lot numbers so the LLM can skip duplicates.
- `_merge()` deduplicates by `lot_number`.
- Non-price fields prefer first non-null values.
- Price fields prefer later non-null values because later windows often contain final hammer prices.
- `sold=True` is final; explicit `sold=False` is preserved over unknown.
- Single-window hallucination bursts are limited by `_MAX_NEW_LOTS_PER_WINDOW`.
- Post-merge price bounds use Tukey IQR on the auction's own observed unit prices.
- Outlier prices can be rechecked with `prompts/verify.txt` before being kept, corrected, or nulled.
- `total_price` is recomputed from `unit_price * num_animals` when needed.

When changing extraction behavior, update or add tests in `tests/test_extractor.py` and keep domain examples in PT-BR where that improves clarity.

## Transcription Backends

| Backend | Flag | Notes |
|---|---|---|
| Groq API | `--transcriber groq` | Default. Requires `GROQ_API_KEY`. Converts audio to 32 kbps MP3 and splits large files into 15-minute chunks. |
| MLX Whisper | `--transcriber mlx` | Apple Silicon local backend. Requires `uv sync --extra local`. |
| whisper.cpp | `--transcriber cpp` | Local backend using `whisper-cli`. Requires `whisper-cpp` and a ggml model. |

Do not make the default transcriber local-only; the repo currently optimizes the default path for fast cloud transcription.

## External Tools And Costs

Be explicit before running commands that can:

- Download multi-hour videos.
- Call Groq, OpenRouter, or OpenAI.
- Re-run OCR or LLM extraction on large cached outputs.
- Clear checkpoints with `--no-resume`.
- Run batch mode over multiple URLs, because costs and runtime multiply by video count.

For routine implementation work, prefer unit tests over live pipeline runs.

## Coding Guidelines

- Follow the existing simple module structure; this is a CLI pipeline, not a framework.
- Keep changes narrow and testable.
- Prefer structured parsing and Pydantic validation over ad hoc string handling.
- Preserve checkpoint/resume behavior when modifying stages.
- Preserve Rich progress output for long-running stages.
- Keep prompts in `prompts/` and Python logic in `pipeline/`; avoid embedding long prompt text in code.
- Do not introduce external services or model providers without CLI, env var, tests, and docs updates.
- Avoid touching generated `output/` data unless the task is specifically about artifacts.

## Common Change Areas

- Model validation or Brazilian price handling: update `models/lot.py` and `tests/test_lot_model.py`.
- LLM JSON parsing, merge semantics, price sanity checks, or provider defaults: update `pipeline/extractor.py` and `tests/test_extractor.py`.
- Window sizing, overlap, timestamp formatting, or OCR/transcript merge behavior: update `pipeline/aggregator.py` and `tests/test_aggregator.py`.
- CLI display, batch reports, or summary numbers: update `main.py`, `tests/test_summary.py`, and/or `tests/test_batch.py`.
- Download, transcription, screenshots, or OCR runtime behavior: update the matching `pipeline/` module and add focused tests where practical. Downloader format and resolution changes belong in `pipeline/downloader.py` and `tests/test_downloader.py`.

## PR Readiness

Before handing off a code change:

1. Run the focused test file for the touched behavior.
2. Run `uv run pytest tests/ -v` unless the change is docs-only or the user explicitly skips it.
3. Report any live pipeline validation that was not run.
4. Note any cost-bearing or environment-dependent behavior that remains unverified.

For docs-only changes to `AGENTS.md`, a syntax-free review is usually enough; no pytest run is required unless code changed.

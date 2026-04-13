# Changelog

## [1.2.7] — 2026-04-12

### Maintenance
- Pre-release changes

## [1.2.6] — 2026-04-09

### Maintenance
- Pre-release changes

## [1.2.5] — 2026-04-06

### 🔧 Maintenance
- Pre-release changes
- Pre-release changes


All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.2.1] - 2026-03-31

### Fixed

- **Broadcast Clock Filtering** — Screen overlays display wall-clock time (e.g., "20:05:08") which was being picked up by OCR and incorrectly used as lot timestamps. Added token filter to remove HH:MM:SS patterns from OCR data before processing.
- **OCR Layout Clarification** — Auction screen text was being misinterpreted by LLM due to unclear layout structure. Added explicit pattern documentation with worked examples showing "LOTE | farm | VALORPORANIMAL | lot_number | R$ | price | description" order.
- **BR Float Mis-parsing** — When LLM outputs Brazilian thousand-separator format (e.g., "5.160") as JSON float, the JSON parser reads it as 5.16 instead of 5160. Added validator heuristic: prices between 0 and 100 are multiplied by 1000 (unrealistic for cattle per-head prices). Also strengthened prompt with explicit examples of this error pattern.
- **Summary Alignment** — Fixed indentation inconsistency where "Lotes" line had 3 spaces while other lines had 2. Split "Preço por Categoria" into M/F sub-lines to prevent text wrapping misalignment.

### Improved

- **Table Sorting** — Lots table now sorted by timestamp_start from earliest to latest lot appearance in video. Lots without timestamp are displayed last.
- **Summary Statistics** — Replaced sex-based price averages with category-based price averages, providing more granular and actionable price insights. Summary now shows price by category (bezerro, garrote, vaca, etc.) instead of by sex.
- **Category Logic** — Improved sex field inference: sex is now always derived from category name (e.g., bezerro→macho, bezerra→fêmea). "Misto" is reserved for genuinely mixed lots only.

### Added

- **Enhanced Testing** — Added 5 new unit tests covering broadcast clock filtering, BR float validation, and edge cases. Total test count: 86 tests (all passing).

### Technical Details

- Window aggregation now filters OCR tokens before merging with transcript segments
- Price coercion validator now applies multiplication guard for realistic cattle price ranges
- Summary statistics computation refactored to support both sex and category aggregations
- All 86 unit tests pass with no external dependencies

## [1.2.0] - 2026-03-28

### Added

- **Unit Test Suite** — Comprehensive test coverage for pure-logic layers with 77 tests:
  - `tests/test_lot_model.py` (21 tests) — Validates `Lot` and `AuctionResult` models, including Brazilian number format coercion (3.100 = 3100.00), `sold` field handling, and required field validation
  - `tests/test_extractor.py` (18 tests) — Tests `_validate_lots`, `_parse_response` (with extra-text tolerance), and `_merge` (first-non-null-wins with sold=False preservation)
  - `tests/test_aggregator.py` (18 tests) — Tests `_fmt`, `_parse_ts` (round-trip), and `aggregate` window logic (overlap, OCR inclusion, empty placeholder)
  - `tests/test_summary.py` (20 tests) — Tests `_calculate_summary` (totals, sex animal counts, top categories, average prices excluding nulls/zero, sold/unsold counts)
- **pytest Integration** — Added `pytest` as a dev dependency via `uv add --dev pytest`

### Technical Details

- All tests are pure unit tests — no external API calls, file I/O, or fixtures required
- Tests validate data transformation logic: model validation, LLM response parsing, window aggregation, and summary statistics
- Run all tests with `uv run pytest tests/ -v`
- Tests ensure data quality rules (BR number format, quantity vs lot number, sold status detection) are correctly validated

## [1.1.1] - 2026-03-28

### Added

- **Auction Metadata Extraction** — New stage 6/6 extracts event-level data: date, city, auctioneer name, farm/expositor, auction type (corte/reprodução/elite), and additional notes. Scans first 3 windows via dedicated PT-BR prompt.
- **Sold Status Field** — Each lot now has `sold: bool | None` to indicate if arrematado (true), retirado/não vendido (false), or undefined (null). Displayed as ✓/✗/- in summary table.
- **Data Quality Rules** — Enhanced prompt to fix common extraction errors:
  - Brazilian number format clarification (3.100 = 3100.00, not 3.10)
  - Quantity vs lot number confusion (lots 135/136 style errors)
  - Detection of unsold lots from auction language

### Fixed

- Price parsing for Portuguese format (thousand separator as period)
- Lot number / animal quantity confusion in LLM responses
- Missing status tracking for withdrawn/unsold lots

## [1.1.0] - 2026-03-28

### Changed

- **OCR Engine** — Replaced PaddleOCR with RapidOCR (ONNX-based). Eliminates PaddlePaddle dependency, supporting Python 3.14+ without build compatibility issues. RapidOCR maintains equivalent accuracy while improving installation reliability.
- **Default Transcriber** — Changed default from `mlx` to `groq`. Groq API provides ~228× realtime speed and costs ~$0.20 per 5-hour video. Users can still use MLX (Apple Silicon) or whisper.cpp locally via `--transcriber` flag.

### Fixed

- **OpenCV Conflict** — Resolved `opencv-python` vs `opencv-contrib-python` conflict on macOS by pinning to `opencv-python-headless`. Prevents `cv2.cvtColor` AttributeError when running OCR.
- **Python 3.14 Support** — Project now installs cleanly on Python 3.14 without native build failures.
- **Author Attribution** — Git history now correctly shows TeodoroRodrigo as sole author.

## [1.0.0] - 2026-03-27

### Added

- **Video Download** — Download Brazilian cattle auction videos from YouTube using yt-dlp with n-challenge solving via JavaScript/Deno
- **Audio Extraction** — Extract 16kHz mono audio tracks for transcription using ffmpeg
- **Multiple Transcription Backends**
  - MLX Whisper for Apple Silicon (Metal acceleration, fast, free)
  - whisper.cpp for local transcription (Metal-accelerated, fast, free)
  - Groq API for cloud transcription (~228× realtime speed, ~$0.20 per 5-hour video)
- **Screenshot Extraction** — Extract JPEG frames every N seconds with live progress bar
- **Optical Character Recognition (OCR)** — RapidOCR (ONNX-based) for Portuguese text extraction from screenshots with confidence filtering
- **Data Aggregation** — Merge transcript segments and OCR results into 10-minute overlapping windows to preserve lot boundaries
- **LLM-Based Extraction** — Send aggregated windows to LLM providers with Portuguese prompts to extract structured lot data
- **Multiple LLM Providers**
  - OpenAI (gpt-4o-mini default)
  - Anthropic Claude (claude-sonnet-4-6)
  - OpenRouter with model aliasing (e.g., gemini-2.5-flash-lite)
- **Checkpoint System** — Each pipeline stage writes checkpoint files; interrupted runs resume automatically
- **Progress Tracking** — Rich progress bars for all long-running operations (transcription, screenshots, OCR, extraction)
- **Structured Output** — JSON output files with lot data including:
  - Lot number, sex (macho/fêmea/misto), category (bezerro/novilha/garrote/boi/vaca/touro/etc.)
  - Number of animals, age in months, breed
  - Unit price and total price per lot
  - Timestamp of lot appearance in video
  - Additional notes
- **Summary Table** — Print formatted table of extracted lots after processing
- **Configuration via .env** — Load API keys from `.env` file using python-dotenv
- **CLI with Click** — Full command-line interface with options for:
  - LLM provider and model selection
  - Transcription backend and model size
  - Screenshot interval customization
  - Output directory configuration
  - Cache clearing (`--no-resume` flag)
- **MIT License** — Open-source under MIT for unrestricted use

### Technical Details

- **Platform**: macOS (Apple Silicon optimized), Python 3.11+
- **Dependency Management**: `uv` for fast virtual environment and package management
- **External Tools**: ffmpeg, yt-dlp, whisper-cpp (optional), ffprobe
- **Key Libraries**:
  - pydantic for data validation
  - rich for terminal UI
  - click for CLI
  - anthropic, openai, groq for LLM providers
  - rapidocr for text recognition
- **Estimated Processing Time**: ~30-60 minutes for a 5-hour video on Apple Silicon M2 (varies by backend and configuration)

[1.2.1]: https://github.com/TeodoroRodrigo/cattle-auction/releases/tag/v1.2.1
[1.2.0]: https://github.com/TeodoroRodrigo/cattle-auction/releases/tag/v1.2.0
[1.1.1]: https://github.com/TeodoroRodrigo/cattle-auction/releases/tag/v1.1.1
[1.1.0]: https://github.com/TeodoroRodrigo/cattle-auction/releases/tag/v1.1.0
[1.0.0]: https://github.com/TeodoroRodrigo/cattle-auction/releases/tag/v1.0.0

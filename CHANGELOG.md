# Changelog

## [1.3.1] — 2026-04-26

Patch release for the post-audit hardening pass.

### Fixed

- **README command drift** — Updated setup, provider, examples, model benchmark, and test-count docs so users no longer see removed `--model` or `ollama` commands.
- **Invalid screenshot intervals** — `--screenshot-interval` now rejects zero and negative values at CLI parse time instead of failing later in screenshot extraction.
- **OCR-only aggregation** — `aggregate()` now builds windows from OCR timestamps when transcript segments are empty, preserving visual evidence if transcription fails or misses content.
- **Verification price parsing** — Outlier verification now reuses Brazilian price coercion, so responses like `R$ 3.100,00` are accepted as `3100.0`.
- **First-window burst filtering** — The >12-lots guard now keeps lots with direct evidence in the current window instead of dropping every first-window lot because no previous lots exist yet.
- **Single-video benchmark drift** — `benchmark.py` now targets the two shipping models and uses the same verification prompt path as production extraction.

### Added

- Regression tests for invalid screenshot intervals, OCR-only aggregation, BR-formatted verification prices, and OCR-supported first-window burst filtering.
- `AGENTS.md` Codex/GPT-5.5 operating guide for the repo.

### Testing

- `132` unit tests passing.

## [1.3.0] — 2026-04-21

Major extraction-quality release. Diagnosed and eliminated the R$ 55,000-per-bezerro hallucination class of bugs, added a statistical outlier framework, introduced a targeted per-lot verification pass, and narrowed the model catalog to two benchmark-validated choices.

### Fixed

- **Impossible prices leaking into output** — LLM would occasionally emit `unit_price > total_price` (e.g. lot 10: unit=R$ 55,000, total=R$ 2,920) and those values would propagate through the pipeline unchallenged. Added shape-invariant checks: `unit_price ≤ total_price` and `unit_price × num_animals ≈ total_price` (20 % tolerance). When invariants fail, the offending fields are nulled so a later window can refill them rather than a corrupt number poisoning the merge.
- **`coerce_price` amplifying hallucinations** — When the LLM emitted a small float like `30.0`, the BR-thousand-separator repair multiplied it to R$ 30,000. Kept the repair (it still handles the common `3.0 → 3000` case) but now shape invariants and statistical bounds catch the amplified result downstream.
- **Cross-window field stitching** — Merge could combine `num_animals` from one window with `total_price` from another, producing a lot that was internally inconsistent even though each contributing observation was not. Post-merge sanity re-check catches and cleans this.
- **Division-by-zero in product-consistency check** — When both `unit_price` and `total_price` were 0, the relative-error calculation crashed. Now guarded.
- **Hallucination bursts within a single window** — The LLM sometimes invented sequential fake lots (e.g., 26 lots in one 10-min window). Detector: if a window returns > 12 lots, keep only those that reference already-found lot numbers; drop the rest.

### Added

- **Statistical outlier detection — Tukey's 1.5·IQR fence** — Per-head price bounds are now derived from the auction's own price distribution (no hard-coded floors/ceilings). Chosen after empirical comparison of mean ± 3σ, median ± 3σ, 1.5·IQR, 3·IQR, log-MAD, and p2-p98 trimming. Tukey 1.5·IQR was the only method robust to the 15-20 % hallucination contamination typical in raw LLM output. Lower fence is clamped at 0 (prices can't be negative).
- **Outlier verification pass** — Lots flagged as statistical outliers get a focused second-pass LLM call against their source evidence. Three possible verdicts: `confirm` (keep), `correct` (use a different price found in the evidence), `discard` (null). This lets the pipeline *recover* legitimate high-priced touros that would otherwise be statistically filtered — a purely-statistical filter can't distinguish an R$ 10k touro from a hallucination.
- **Correction-guardrail** — When the verification LLM proposes a "correction," the new price is re-checked against the Tukey fence. If the correction is still an outlier (observed in practice: "R$ 55,000 → R$ 42"), the correction is rejected and the lot is discarded instead of silently accepting an implausible fix.
- **Token-usage tracking in `LLMClient`** — Running counters for `input_tokens`, `output_tokens`, and `n_calls` so callers can estimate $ cost from published per-token prices. Surfaced in `summary.json` for each benchmark run.
- **Prompt hardening** — Added explicit invariant rules (`total_price ≈ unit_price × num_animals`, price cannot exceed total), plausibility bounds (R$ 1,500-10,000 per-head typical, R$ 500 floor, R$ 20,000 ceiling except touros), anti-hallucination directives (prefer null over guessing, don't invent sequential lot numbers, expect ≤ 8 lots per 10-min window), and consolidation rules (don't emit the same lot_number multiple times in one response).
- **Verification prompt** (`prompts/verify.txt`) — New focused prompt for the second-pass lot audit. Portuguese-language, with explicit BR number-format rules, OCR layout reminders, and reference price ranges per cattle category.
- **Benchmark harness** (`bench/`) — Reproducible model comparison across multiple videos:
  - `bench/run_single.py` — runs one (video, provider, model) combo on cached transcript+OCR, writes `lots.json` + `summary.json` + `log.txt` per combo.
  - `bench/orchestrate.py` — parallel runner (4-way concurrency) over the full video × model matrix.
  - `bench/analyze.py` — three-section report: reference-based accuracy, raw extraction stats, token usage + cost estimates.
  - Empirical 5-videos × 10-models benchmark documented; results drove the v1.3.0 model-catalog simplification.

### Changed

- **Model catalog narrowed to two benchmark-validated options:**
  - **Default**: `openrouter` → `google/gemini-2.5-flash-lite-preview-09-2025` ($0.10/M in · $0.40/M out · ~$0.05/video · 13-24 s per short auction). Best speed + cost, near-parity accuracy.
  - **Alternative**: `openai` → `gpt-4.1-mini` ($0.40/M in · $1.60/M out · ~$0.13/video). Best accuracy on fine-grained category names.
- **Price merge across windows now applies post-merge sanity**: cross-window stitching that produces internally inconsistent field combinations gets cleaned.
- **Sanity check ordering** reworked so the least-destructive action wins: when `unit × n ≠ total`, trust unit (the hammer price) and clear only the mismatched total; only null both when unit is also out of statistical bounds.

### Removed

- **Ollama provider** from `LLMClient` (never used in benchmarks, added complexity).
- **`_MODEL_ALIASES`** — no aliases needed with only two supported models.
- **`--model` CLI flag** — model is now determined by `--provider`; picking via alias or arbitrary string is no longer exposed.
- **`gpt-5-*` special-case** in `max_tokens` handling (dropped with the model list; can be restored when/if gpt-5 becomes a shipped option).

### Performance / behavior

- **Realistic extracted prices.** On the canonical test video (Quirinópolis 2026-04-19): average extracted price dropped from **R$ 4,976** (dominated by two R$ 30-55k hallucinations) to **R$ 3,293** — matching real cattle auction prices in the region.
- **Invariant violations eliminated.** 2 `unit > total` violations in v1.2.7 → **0** across the same dataset post-release.
- **Outlier recovery in action.** Legitimate high-priced touros (e.g. a R$ 10,000 reproductor on ZCwxnhUZjQM) now pass through because the verification step confirms them against the audio; a pure statistical filter would have nulled them.
- **Benchmark rankings** (5 videos × 10 models): Gemini Flash-Lite Preview (default) and GPT-4.1-mini tied on coverage; Gemini is 2.8× cheaper and 3.8× faster on average; GPT-4.1-mini leads slightly on category-name fidelity.

### Testing

- **127 tests passing** (up from 77 in v1.2.x; 50 new tests in this release).
- New test classes: `TestSanityCheckInvariants`, `TestSanityCheckStatisticalBounds`, `TestComputePriceBounds`, `TestValidateLotsShapeOnly`, `TestPostMergeSanity`, `TestParseHHMMSS`, `TestVerifyLot` (with mocked LLM client covering confirm / correct / discard verdicts, malformed JSON, LLM errors, missing-timestamp edge cases).

### Documentation

- `CLAUDE.md` rewritten for the two-model catalog, with provider-comparison table citing cost/speed/coverage numbers from the benchmark.
- Benchmark harness documented inline in `bench/*.py`.

### Breaking changes

- Calls using `--provider ollama` or `--model <custom>` will no longer work. Migration: drop the `--model` flag and pick `--provider openrouter` (default) or `--provider openai`.

---

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

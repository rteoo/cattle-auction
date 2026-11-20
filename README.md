# cattle-auction

<p align="center">
  <img src="docs/cattle-auction-icon.svg" width="128" alt="cattle-auction icon">
</p>

<p align="center">
  A command-line pipeline that turns Brazilian cattle auction videos on YouTube
  into structured, checkpointed lot data.
</p>

<p align="center">
  <a href="https://github.com/rteoo/cattle-auction/actions/workflows/tests.yml"><img src="https://github.com/rteoo/cattle-auction/actions/workflows/tests.yml/badge.svg" alt="Test status"></a>
  <a href="https://github.com/rteoo/cattle-auction/tags"><img src="https://img.shields.io/github/v/tag/rteoo/cattle-auction?label=stable" alt="Stable tag"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT license"></a>
</p>

Give cattle-auction a YouTube link to a *leilão de gado*. It transcribes the
auctioneer in Portuguese, reads the on-screen lot board with OCR, and asks an
LLM to extract every lot: number, sex, category, head count, age, breed, price
per head, and whether it sold. Results are written as JSON and printed as a
summary table, together with the run's estimated cost.

## Highlights

- Audio-only download for transcription and a separate low-resolution video for
  OCR, so a multi-hour auction never needs a full-quality download.
- PT-BR transcription through Groq (cloud, default), MLX Whisper, or whisper.cpp,
  with a quality gate for caption-credit hallucinations, repetition loops, and
  mostly-silent audio.
- Frame sampling on a fixed clock or at the moments the lot board changes.
- Brazilian number handling: `3.100` is R$ 3,100, never R$ 3.10.
- Overlapping extraction windows, exact lot labels, OCR board-marker checks,
  and a second pass for uncertain sale outcomes and hammer prices.
- Auction metadata: date, city, auctioneer, farm, and auction type.
- Batch mode over many videos with a comparison report.
- Every stage is checkpointed; interrupted runs resume where they stopped, down to
  the last finished LLM window or Groq audio chunk.
- A per-run USD cost estimate for LLM tokens and cloud transcription.

## Quick start

Run cattle-auction from the [v2.0.0 source release](https://github.com/rteoo/cattle-auction/releases/tag/v2.0.0).
It needs Python 3.11 or later, [uv](https://docs.astral.sh/uv/), `ffmpeg`, and a
[YouTube JavaScript runtime](https://github.com/yt-dlp/yt-dlp/wiki/EJS).
Put Deno 2.3+ or Node.js 22+ on `PATH`. yt-dlp prefers Deno and can use Node.js
when Deno is unavailable or unsupported. The locked Python dependencies include
the matching yt-dlp EJS scripts. The upstream OCR package declares support
through Python 3.12.
On macOS, the locked OCR stack requires Apple Silicon and macOS 14 or newer.

```bash
# macOS
brew install ffmpeg deno

# Windows
winget install Gyan.FFmpeg DenoLand.Deno
```

Clone the release tag and install the locked dependencies. This release locks
`rapidocr-onnxruntime` at 1.4.4.

```bash
git clone --branch v2.0.0 https://github.com/rteoo/cattle-auction.git
cd cattle-auction
uv sync --frozen                  # base dependencies
uv sync --frozen --extra local    # optional: MLX Whisper, Apple Silicon only
```

## First use

1. Create a `.env` file in the repository root with the keys for the services
   you will use. The default run needs `OPENAI_API_KEY` and `GROQ_API_KEY`:

   ```bash
   OPENAI_API_KEY=sk-...
   GROQ_API_KEY=gsk-...
   OPENROUTER_API_KEY=sk-or-... # only with --provider openrouter
   ```

2. Run the pipeline on one auction:

   ```bash
   uv run --frozen python main.py "https://www.youtube.com/watch?v=..."
   ```

3. Read the results in `output/<video_id>/`: `result_<video_id>.json` holds the
   metadata and every lot, and the terminal shows the summary, the lot table, and
   the run's estimated cost.

Keys are loaded from `.env` on every run. Only set the keys you need.

## How it works

| Stage | What happens |
| --- | --- |
| Download audio | `yt-dlp` fetches audio only; `ffmpeg` converts it to 16 kHz mono WAV |
| Transcribe | PT-BR speech-to-text, then the hallucination and coverage gate |
| Download OCR video | A 480p video by default, or 720p on request |
| Screenshots | `ffmpeg` frames on a fixed interval or at lot-board changes |
| OCR | RapidOCR reads the on-screen text of every frame |
| Aggregate | Transcript and OCR merge into 10-minute windows with 1-minute overlap |
| Extract lots | Each window goes to the LLM with a PT-BR prompt; lots are merged and sanity-checked |
| Extract metadata | The opening windows plus the video title and description yield auction details |

On-screen text helps identify exact lot labels, head counts, and price units.
An OCR-only ID can be rejected when its source frame lacks a lot-board marker
and other evidence does not support it. Explicit audio or visual evidence can
confirm a sale. A board price may still be an open bid.

The extractor reviews uncertain or conflicting outcomes and missing, flagged,
or statistically unusual prices with nearby transcript and OCR. A revised
decision must cite a timestamped source line. This review can add LLM calls.

Frame indexes record fractional source timestamps. Scene mode also tries to
retain the outgoing board shortly before a detected change. OCR metadata
retains each frame's bounding boxes, confidence scores, and source path.

## Options

```bash
uv run --frozen python main.py <youtube_url> [OPTIONS]
uv run --frozen python main.py <url_1> <url_2> [OPTIONS]
uv run --frozen python main.py --batch-file links.txt --batch-name maio-2026 [OPTIONS]
```

| Option | Default | Effect |
| --- | --- | --- |
| `--provider` | `openai` | `openai` for GPT-6 Luna or `openrouter` for Gemini 2.5 Flash-Lite |
| `--transcriber` | `groq` | Transcription backend: `groq`, `mlx`, or `cpp` |
| `--whisper-model` | `medium` | Model size for `mlx` and `cpp`: `tiny`, `base`, `small`, `medium`, `large-v3` |
| `--cpp-model` | auto | Path to a ggml model file for whisper.cpp |
| `--frame-sampling` | `interval` | `interval` samples on a fixed clock; `scene` samples where the lot board changes |
| `--screenshot-interval` | `30` | Seconds between frames in interval sampling |
| `--safety-interval` | `60` | Scene sampling: seconds between safety-grid frames added to detections |
| `--ocr-video-height` | `480` | OCR video height: `480` or `720` |
| `--output-dir` | `output` | Base directory for all generated files |
| `--no-resume` | off | Recompute derived checkpoints; downloaded media is kept |
| `--metadata / --no-metadata` | on | Show auction metadata |
| `--summary / --no-summary` | on | Show summary statistics |
| `--table / --no-table` | on | Show the full lot table |
| `--batch-file` | none | Text file with one URL per line; blank lines and `#` comments are ignored |
| `--batch-name` | timestamp | Folder name for the batch report under `output/batches/` |
| `--stop-on-error` | off | Stop a batch at the first failed URL |

Use `--frame-sampling scene` for fewer frames placed closer to lot changes, and
`--ocr-video-height 720` when small on-screen text is misread at 480p.

## Batch runs

Pass several URLs, or a file of URLs with `--batch-file`. Videos run one after
another; each keeps its own checkpoint folder. When the batch finishes, the CLI
prints a comparison table and writes:

| File | Contents |
| --- | --- |
| `output/batches/<name>/batch_summary.json` | Totals, per-video rows, failure records, comparison winners, cost |
| `output/batches/<name>/comparison.md` | Human-readable summary and per-category price table |

A batch continues past a failed URL by default, records the error, and exits
non-zero if any item failed.

## Output

Every run writes to `output/<video_id>/`:

| File | Contents |
| --- | --- |
| `audio_source_<id>.<ext>`, `audio_<id>.wav` | Downloaded audio and the 16 kHz mono WAV |
| `video_ocr_<id>_480p.mp4` | Low-resolution video for screenshots (`_720p` on request) |
| `transcript_<id>.json` | Timestamped transcript segments |
| `screenshots_<id>/`, `screenshots_<id>.json` | JPEG frames and their index |
| `ocr_results_<id>.json` | Screen text per timestamp |
| `lots_<id>.json` | Extracted lots |
| `lots_<id>.quality.json` | Window validation statuses and a count of OCR-only lot IDs rejected by the board-marker check |
| `metadata_<id>.json` | Date, city, auctioneer, farm, and auction type |
| `result_<id>.json` | Final result: metadata, lots, and the run's estimated cost |

Each lot follows this schema:

```json
{
  "lot_number": "08B",
  "sex": "macho",
  "category": "garrote",
  "num_animals": 30,
  "age_months": 18,
  "breed": "Nelore",
  "unit_price": 3200.00,
  "total_price": 96000.00,
  "sold": true,
  "timestamp_start": "01:24:35",
  "notes": null,
  "sale_timestamp": "01:26:10",
  "raw_price": "R$ 3.200,00",
  "price_basis": "per_head",
  "price_needs_review": false,
  "price_timestamp": "01:26:10"
}
```

As of v2.0.0, `lot_number` is a string. Treat `"0"`, `"00"`, `"11A"`, and
`"08B"` as distinct labels. The lot parser accepts old integer IDs and converts
them to strings. It cannot recover leading zeros or suffixes that old files
did not store.

`sold` is `true` for *arrematado*, `false` for withdrawn or unsold, and `null`
when the outcome is unclear. The CLI clears `unit_price` and `total_price`
when there is no confirmed sale. When the per-head price and head count are known, it computes
`total_price` from `unit_price × num_animals`. For price analysis, check
`price_needs_review` and consult `raw_price`, `price_basis`, and the evidence
timestamps when present. The CLI excludes flagged prices from confirmed-price
statistics.

## Models and cost

| Transcriber | Flag | Runs on | Cost |
| --- | --- | --- | --- |
| Groq Whisper Large v3 Turbo | `--transcriber groq` | Groq cloud | about US$0.04 per audio hour |
| MLX Whisper | `--transcriber mlx` | Apple Silicon, local | free |
| whisper.cpp | `--transcriber cpp` | local `whisper-cli` | free |

For whisper.cpp, install it and download a model first:

```bash
brew install whisper-cpp
whisper-cpp-download-ggml-model medium
```

| Provider | Model | Standard input per 1M tokens | Standard output per 1M tokens |
| --- | --- | ---: | ---: |
| `openai` (default) | `gpt-6-luna` | US$0.10 | US$0.50 |
| `openrouter` | `google/gemini-2.5-flash-lite` | US$0.10 | US$0.40 |

Luna uses medium reasoning effort. The run estimates token costs at standard
uncached rates. See [OpenAI pricing](https://developers.openai.com/api/docs/pricing)
and [OpenRouter pricing](https://openrouter.ai/google/gemini-2.5-flash-lite).

Measured on a 5-hour video on an Apple Silicon M2, Groq transcribes in about 1–2
minutes and local medium-model backends in about 20–40. Screenshots take 2–3
minutes, OCR of ~600 frames 5–10, and extraction of ~30 windows 3–8. Downloads
depend on the network.

## Benchmarking

To score saved output, run `bench/analyze.py` on the result and reference files.
It runs offline.

```bash
uv run --frozen python bench/analyze.py --reference PATH_TO_REFERENCE_JSON --result PATH_TO_RESULT_OR_LOTS_JSON
```

To compare a fresh extraction, run `bench/run_single.py` on saved transcript and
OCR checkpoints. That command makes paid LLM calls.

```bash
uv run --frozen python bench/run_single.py VIDEO_ID openai gpt-6-luna --reasoning-effort medium --input-dir PATH_TO_VIDEO_CHECKPOINTS
```

The scoring command accepts a lot array or an object containing `lots`.
Use exact string IDs in new reference files. Numeric IDs in historical
references can match a unique corresponding label, preserving distinct
padded or suffixed labels in new data.

For a paid extraction-only benchmark on cached transcript and OCR, compare
GPT-6 Luna reasoning levels on the two included reference auctions:

```bash
uv run --frozen python bench/orchestrate.py --luna --video VDzRLEMUagA --video sKmUHExf464
uv run --frozen python bench/analyze.py
```

Read hammer-price coverage and its sample count alongside price error.

## Resume and checkpoints

Each stage saves its result under `output/<video_id>/`. Downloads are reused
once complete. The transcript, screenshot, OCR, lot, and metadata checkpoints
also record the inputs that produced them, so a rerun recomputes a stage when
its prompt, model, sampling setting, or source file changes and reuses it
otherwise.

- Upgrading to v2.0.0 can rerun lot extraction because the prompts and checkpoint
  provenance changed. Compatible upstream checkpoints and downloaded media can
  still be reused.
- JSON checkpoints are written atomically, and an unreadable checkpoint is
  recomputed instead of stopping the run.
- ffmpeg outputs are renamed into place only after a clean exit, so an
  interrupted conversion is never mistaken for a finished one.
- If an LLM window, sale-review batch, or Groq chunk fails, completed work
  is kept. A rerun with matching inputs resumes from the remaining work.
- `--no-resume` clears derived checkpoints but keeps downloaded media.

A stage served from its checkpoint costs nothing, so a fully resumed run reports
an estimated cost of about zero.

## Data safety and privacy

cattle-auction sends data only to the services a run uses. It has no telemetry.

- With `--transcriber groq`, the auction audio is uploaded to Groq as 32 kbps MP3.
  The `mlx` and `cpp` backends keep audio on the machine.
- Transcript and OCR text for each window, plus the video title and description
  for metadata, are sent to OpenRouter or OpenAI.
- API keys stay in `.env`, which is git-ignored. Do not commit it or pass keys on
  the command line.
- `output/` is git-ignored. It holds downloaded audio and video from YouTube;
  keep it private.

## Platform status and limitations

cattle-auction is a Python CLI built on portable tools (`yt-dlp`, `ffmpeg`,
RapidOCR). The default Groq path is used on macOS and Windows; CI runs the
offline test suite on Linux with Python 3.11.

- **Transcription:** MLX Whisper runs only on Apple Silicon. whisper.cpp needs
  `whisper-cli` on `PATH`; its model is auto-detected under `~/.cache/whisper/`
  or `/opt/homebrew/share/whisper-cpp/`, and otherwise must be passed with
  `--cpp-model`.
- **Broadcast layouts:** the lot-ID filter looks for lot markers across
  layouts. Some price and category checks require a recognizable per-head
  board. When the layout is unfamiliar, extraction may rely more on the audio.
- **Audio-only recaps:** a sale recap with no lot start or identifying details
  can be linked to the sole matching zero-padded board label. Check the video
  when padded and unpadded labels may refer to separate lots.
- **Scene sampling:** the adaptive threshold is calibrated on synthetic lot
  boards, not yet on real broadcasts with live camera feeds behind the overlay.
- **Arroba prices:** quotes may be retained in `raw_price` with
  `price_basis: "per_arroba"`. The pipeline does not convert them to a price per head.
- **Cost estimates:** per-token and per-hour prices are fixed in
  `pipeline/costs.py` and drift with provider pricing.
- **Live verification:** YouTube downloads, ffmpeg, OCR, and LLM or Groq calls
  are outside the offline test suite and are not exercised in CI.

## Develop and build

Run the offline test suite from the repository root:

```bash
uv run --frozen pytest tests/ -q
```

Always pass `--frozen`: it tests the committed `uv.lock` instead of re-resolving
it. The tests make no network calls, download no videos, and need no API keys.
[AGENTS.md](AGENTS.md) documents the architecture, extraction rules, and
checkpoint contract.

The benchmark harness in [`bench/`](bench) compares extraction models against a
human reference; `benchmark.py` runs a single-video comparison.

Releases use `release.py`. Run the dry run first to review the proposed
version and planned checks before creating a release commit or tag:

```bash
uv run --frozen python release.py --dry-run
```

A real release runs the tests, builds and verifies the wheel and source archive,
stages only allowlisted paths, and creates the release commit, tag, and GitHub
release. Release history is documented in [CHANGELOG.md](CHANGELOG.md).

## License

cattle-auction is released under the [MIT License](LICENSE).

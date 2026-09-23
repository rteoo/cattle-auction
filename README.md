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
- Defensive extraction: overlapping windows, duplicate merging, hallucination-burst
  limits, and a second LLM check for statistically implausible prices.
- Auction metadata: date, city, auctioneer, farm, and auction type.
- Batch mode over many videos with a comparison report.
- Every stage is checkpointed; interrupted runs resume where they stopped, down to
  the last finished LLM window or Groq audio chunk.
- A per-run USD cost estimate for LLM tokens and cloud transcription.

## Quick start

cattle-auction runs from source; there is no packaged release. It needs
Python 3.11 or later, [uv](https://docs.astral.sh/uv/), `ffmpeg`, and the `deno`
runtime that `yt-dlp` uses for YouTube downloads:

```bash
# macOS
brew install ffmpeg deno

# Windows
winget install Gyan.FFmpeg DenoLand.Deno
```

Clone the repository and install the locked dependencies:

```bash
git clone https://github.com/rteoo/cattle-auction.git
cd cattle-auction
uv sync --frozen                  # base dependencies
uv sync --frozen --extra local    # optional: MLX Whisper, Apple Silicon only
```

## First use

1. Create a `.env` file in the repository root with the keys for the services
   you will use. The default run needs `OPENROUTER_API_KEY` and `GROQ_API_KEY`:

   ```bash
   OPENROUTER_API_KEY=sk-or-...
   GROQ_API_KEY=gsk-...
   OPENAI_API_KEY=sk-...        # only with --provider openai
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

On-screen data takes priority over audio for lot number, head count, price, and
sale status. Prices outside the auction's own Tukey fence are sent back to the
LLM with their source evidence to be confirmed, corrected, or discarded.

## Options

```bash
uv run --frozen python main.py <youtube_url> [OPTIONS]
uv run --frozen python main.py <url_1> <url_2> [OPTIONS]
uv run --frozen python main.py --batch-file links.txt --batch-name maio-2026 [OPTIONS]
```

| Option | Default | Effect |
| --- | --- | --- |
| `--provider` | `openrouter` | LLM provider: `openrouter` or `openai` |
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
| `metadata_<id>.json` | Date, city, auctioneer, farm, and auction type |
| `result_<id>.json` | Final result: metadata, lots, and the run's estimated cost |

Each lot follows this schema:

```json
{
  "lot_number": 12,
  "sex": "macho",
  "category": "garrote",
  "num_animals": 30,
  "age_months": 18,
  "breed": "Nelore",
  "unit_price": 3200.00,
  "total_price": 96000.00,
  "sold": true,
  "timestamp_start": "01:24:35",
  "notes": null
}
```

`sold` is `true` for *arrematado*, `false` for withdrawn or unsold, and `null`
when the video does not settle it. `total_price` is recomputed from
`unit_price × num_animals` when the source does not state it.

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

The two extraction models were chosen by benchmarking against a human reference:

| Provider | Model | Cost per video | Speed | Lot coverage | Price error (MAPE) |
| --- | --- | ---: | ---: | ---: | ---: |
| `openrouter` (default) | `google/gemini-2.5-flash-lite-preview-09-2025` | ~US$0.05 | 13–24 s | 92–100% | 1.9–2.0% |
| `openai` | `gpt-4.1-mini` | ~US$0.13 | 31 s | 100% | 0.1% |

Measured on a 5-hour video on an Apple Silicon M2, Groq transcribes in about 1–2
minutes and local medium-model backends in about 20–40. Screenshots take 2–3
minutes, OCR of ~600 frames 5–10, and extraction of ~30 windows 3–8. Downloads
depend on the network.

## Resume and checkpoints

Each stage saves its result under `output/<video_id>/`. Downloads are reused
once complete. The transcript, screenshot, OCR, lot, and metadata checkpoints
also record the inputs that produced them, so a rerun recomputes a stage when
its prompt, model, sampling setting, or source file changes and reuses it
otherwise.

- JSON checkpoints are written atomically, and an unreadable checkpoint is
  recomputed instead of stopping the run.
- ffmpeg outputs are renamed into place only after a clean exit, so an
  interrupted conversion is never mistaken for a finished one.
- If an LLM window or a Groq chunk fails, the stage stops, but the finished
  windows and chunks are kept; the rerun only sends what did not finish.
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
- **Broadcast layouts:** the extraction prompt is tuned for the common lot-board
  overlay (`LOTE | fazenda | VALORPORANIMAL | lote | R$ | preço | ...`). Other
  layouts rely more on the audio and may extract less reliably.
- **Scene sampling:** the adaptive threshold is calibrated on synthetic lot
  boards, not yet on real broadcasts with live camera feeds behind the overlay.
- **Arroba prices:** prices quoted per arroba (@) are recorded in `notes`, not
  converted to a price per head.
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
version, tests, package contents, and staged paths without changing Git state:

```bash
uv run --frozen python release.py --dry-run
```

A real release runs the tests, builds and verifies the wheel and source archive,
stages only allowlisted paths, and creates the release commit, tag, and GitHub
release. Release history is documented in [CHANGELOG.md](CHANGELOG.md).

## License

cattle-auction is released under the [MIT License](LICENSE).

# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[1.1.0]: https://github.com/TeodoroRodrigo/cattle-auction/releases/tag/v1.1.0
[1.0.0]: https://github.com/TeodoroRodrigo/cattle-auction/releases/tag/v1.0.0

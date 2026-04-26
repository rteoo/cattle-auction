"""
Benchmark script: runs extraction with multiple LLM models on the same video
and compares results side-by-side.

Usage:
    uv run python benchmark.py <youtube_url>
"""

import json
import shutil
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from pipeline import aggregator, downloader, extractor, ocr, screenshotter
from pipeline import transcriber as transcriber_mod

PROMPTS_DIR = Path(__file__).parent / "prompts"

MODELS = [
    ("openrouter", "google/gemini-2.5-flash-lite-preview-09-2025"),
    ("openai", "gpt-4.1-mini"),
]

def download_small(url: str, output_dir: Path, video_id: str) -> tuple[Path, Path]:
    """Download video at 480p max (smaller/faster) and extract audio."""
    import subprocess

    video_path = output_dir / f"video_{video_id}.mp4"
    audio_path = output_dir / f"audio_{video_id}.wav"

    if not video_path.exists():
        import os
        env = os.environ.copy()
        # Ensure winget-installed tools are on PATH
        winget_links = os.path.expanduser("~") + "\\AppData\\Local\\Microsoft\\WinGet\\Links"
        env["PATH"] = winget_links + ";" + env.get("PATH", "")
        subprocess.run(
            [
                "yt-dlp",
                "--remote-components", "ejs:github",
                "-o", str(video_path),
                "-f", "bestvideo[height<=480][ext=mp4]+bestaudio[ext=m4a]/best[height<=480][ext=mp4]/best",
                "--merge-output-format", "mp4",
                "--ffmpeg-location", winget_links,
                url,
            ],
            check=True,
            env=env,
        )
    else:
        print(f"  Video already exists, skipping download.")

    if not audio_path.exists():
        ffmpeg_path = os.path.join(winget_links, "ffmpeg.exe")
        if not os.path.exists(ffmpeg_path):
            ffmpeg_path = "ffmpeg"
        subprocess.run(
            [
                ffmpeg_path,
                "-i", str(video_path),
                "-ar", "16000",
                "-ac", "1",
                "-c:a", "pcm_s16le",
                str(audio_path),
                "-y",
                "-hide_banner",
                "-loglevel", "error",
            ],
            check=True,
            env=env,
        )
    else:
        print(f"  Audio already extracted, skipping.")

    return video_path, audio_path


def run_shared_stages(url: str, output_dir: str):
    """Run download, transcription, screenshots, OCR — cached across models."""
    video_id = url.split("v=")[-1].split("&")[0]
    run_dir = Path(output_dir) / video_id
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  SHARED STAGES — video {video_id}")
    print(f"{'='*60}\n")

    # Stage 1: Download
    print("[1/4] Downloading video info...")
    video_info = downloader.get_video_info(url, run_dir, video_id)

    print("[1/4] Downloading video + audio (480p)...")
    video_path, audio_path = download_small(url, run_dir, video_id)

    # Stage 2: Transcribe
    print("[2/4] Transcribing audio (groq)...")
    transcript_path = run_dir / f"transcript_{video_id}.json"
    segments = transcriber_mod.transcribe(
        audio_path=audio_path,
        output_path=transcript_path,
        backend="groq",
    )

    # Stage 3: Screenshots
    print("[3/4] Taking screenshots...")
    screenshots = screenshotter.extract_screenshots(
        video_path=video_path,
        output_dir=run_dir,
        video_id=video_id,
        interval=30,
    )

    # Stage 4: OCR
    print("[4/4] Running OCR...")
    ocr_path = run_dir / f"ocr_results_{video_id}.json"
    ocr_results = ocr.run_ocr(screenshots, ocr_path)

    # Aggregate windows
    windows = aggregator.aggregate(segments, ocr_results)

    return video_id, run_dir, windows, video_info


def run_extraction(
    provider: str,
    model: str,
    windows: list,
    video_id: str,
    run_dir: Path,
    video_info: dict,
    benchmark_dir: Path,
):
    """Run extraction with a specific model and save results."""
    model_label = f"{provider}/{model}"
    safe_name = model.replace("/", "_").replace(":", "_")
    print(f"\n{'-'*60}")
    print(f"  EXTRACTING: {model_label}")
    print(f"{'-'*60}")

    # Remove cached extraction files so the model runs fresh
    lots_path = run_dir / f"lots_{video_id}.json"
    metadata_path = run_dir / f"metadata_{video_id}.json"
    result_path = run_dir / f"result_{video_id}.json"
    for p in [lots_path, metadata_path, result_path]:
        p.unlink(missing_ok=True)

    try:
        client = extractor.LLMClient(provider=provider, model=model)
    except Exception as e:
        print(f"  SKIP: could not create client — {e}")
        return None

    # Extract lots
    start_time = time.time()
    try:
        lots = extractor.extract_lots(
            windows=windows,
            client=client,
            prompt_path=PROMPTS_DIR / "extraction.txt",
            output_path=lots_path,
            verify_prompt_path=PROMPTS_DIR / "verify.txt",
        )
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"  FAILED after {elapsed:.1f}s — {e}")
        return {
            "model": model_label,
            "status": "error",
            "error": str(e),
            "time_seconds": round(elapsed, 1),
        }

    lots_time = time.time() - start_time

    # Extract metadata
    meta_start = time.time()
    try:
        metadata = extractor.extract_metadata(
            windows=windows,
            client=client,
            prompt_path=PROMPTS_DIR / "metadata.txt",
            output_path=metadata_path,
            video_info=video_info,
        )
    except Exception as e:
        metadata = {"error": str(e)}

    meta_time = time.time() - meta_start
    total_time = lots_time + meta_time

    # Build result summary
    lots_data = [lot.model_dump() for lot in lots]

    # Calculate stats
    total_lots = len(lots)
    total_animals = sum(l.num_animals for l in lots)
    sold_count = sum(1 for l in lots if l.sold is True)
    unsold_count = sum(1 for l in lots if l.sold is False)
    unknown_sold = sum(1 for l in lots if l.sold is None)
    prices = [l.unit_price for l in lots if l.unit_price and l.unit_price > 0]
    avg_price = round(sum(prices) / len(prices), 2) if prices else 0
    lots_with_price = len(prices)
    lots_without_price = total_lots - lots_with_price
    lots_with_breed = sum(1 for l in lots if l.breed and l.breed != "Sem Raça Definida")
    lots_with_age = sum(1 for l in lots if l.age_months is not None)
    lots_with_timestamp = sum(1 for l in lots if l.timestamp_start is not None)

    result = {
        "model": model_label,
        "status": "ok",
        "time_seconds": round(total_time, 1),
        "lots_extraction_time": round(lots_time, 1),
        "metadata_extraction_time": round(meta_time, 1),
        "total_lots": total_lots,
        "total_animals": total_animals,
        "sold": sold_count,
        "unsold": unsold_count,
        "unknown_sold": unknown_sold,
        "lots_with_price": lots_with_price,
        "lots_without_price": lots_without_price,
        "avg_unit_price": avg_price,
        "lots_with_breed": lots_with_breed,
        "lots_with_age": lots_with_age,
        "lots_with_timestamp": lots_with_timestamp,
        "metadata": metadata,
        "lot_numbers": sorted([l.lot_number for l in lots]),
    }

    # Save per-model details
    out = benchmark_dir / f"{safe_name}.json"
    out.write_text(json.dumps({"summary": result, "lots": lots_data}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  OK: {total_lots} lots, {total_animals} animals, {sold_count} sold, avg R${avg_price:.2f} — {total_time:.1f}s")

    return result


def print_comparison(results: list[dict]):
    """Print side-by-side comparison table."""
    ok_results = [r for r in results if r and r["status"] == "ok"]
    failed = [r for r in results if r and r["status"] != "ok"]

    if not ok_results:
        print("\nNo successful results to compare.")
        return

    print(f"\n{'='*100}")
    print(f"  BENCHMARK RESULTS")
    print(f"{'='*100}\n")

    # Header
    headers = ["Metric"] + [r["model"].split("/")[-1][:20] for r in ok_results]
    col_w = 22
    header_line = f"{'Metric':<25}" + "".join(f"{h:>{col_w}}" for h in headers[1:])
    print(header_line)
    print("-" * len(header_line))

    # Rows
    metrics = [
        ("Total lots", "total_lots"),
        ("Total animals", "total_animals"),
        ("Sold", "sold"),
        ("Unsold", "unsold"),
        ("Unknown sold", "unknown_sold"),
        ("Lots w/ price", "lots_with_price"),
        ("Lots w/o price", "lots_without_price"),
        ("Avg unit price (R$)", "avg_unit_price"),
        ("Lots w/ breed", "lots_with_breed"),
        ("Lots w/ age", "lots_with_age"),
        ("Lots w/ timestamp", "lots_with_timestamp"),
        ("Extraction time (s)", "lots_extraction_time"),
        ("Total time (s)", "time_seconds"),
    ]

    for label, key in metrics:
        vals = []
        for r in ok_results:
            v = r.get(key, "—")
            if isinstance(v, float):
                vals.append(f"{v:,.2f}")
            else:
                vals.append(str(v))
        row = f"{label:<25}" + "".join(f"{v:>{col_w}}" for v in vals)
        print(row)

    # Lot number coverage
    print(f"\n{'-'*60}")
    print("Lot numbers found per model:")
    for r in ok_results:
        nums = r.get("lot_numbers", [])
        name = r["model"].split("/")[-1][:30]
        print(f"  {name}: {nums}")

    # Metadata comparison
    print(f"\n{'-'*60}")
    print("Metadata extracted per model:")
    for r in ok_results:
        name = r["model"].split("/")[-1][:30]
        meta = r.get("metadata", {})
        print(f"  {name}:")
        for k in ["date", "city", "auctioneer", "farm", "auction_type"]:
            print(f"    {k}: {meta.get(k, '—')}")

    if failed:
        print(f"\n{'-'*60}")
        print("Failed models:")
        for r in failed:
            print(f"  {r['model']}: {r.get('error', 'unknown error')}")


def main():
    if len(sys.argv) < 2:
        print("Usage: uv run python benchmark.py <youtube_url>")
        sys.exit(1)

    url = sys.argv[1]
    output_dir = "output"

    # Run shared stages (download, transcribe, screenshot, OCR)
    video_id, run_dir, windows, video_info = run_shared_stages(url, output_dir)

    # Create benchmark output dir
    benchmark_dir = run_dir / "benchmark"
    benchmark_dir.mkdir(exist_ok=True)

    print(f"\n  {len(windows)} windows to process")
    print(f"  Testing {len(MODELS)} models\n")

    # Run each model
    results = []
    for provider, model in MODELS:
        result = run_extraction(provider, model, windows, video_id, run_dir, video_info, benchmark_dir)
        results.append(result)

    # Save combined results
    combined_path = benchmark_dir / "comparison.json"
    combined_path.write_text(
        json.dumps([r for r in results if r], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Print comparison
    print_comparison(results)

    print(f"\n  Results saved to: {benchmark_dir}")


if __name__ == "__main__":
    main()

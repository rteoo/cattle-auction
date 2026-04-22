"""Run extraction for one (video_id, provider, model) combo using cached
transcript + OCR. Writes lots JSON, summary JSON, and a log file into
bench_results/<video_id>/<model_safe>/."""
import json
import os
import sys
import time
import traceback
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)

# Load .env so API keys are available
for line in (ROOT / ".env").read_text().splitlines():
    if "=" in line and not line.strip().startswith("#"):
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

sys.path.insert(0, str(ROOT))


def main():
    if len(sys.argv) != 4:
        print("usage: run_single.py <video_id> <provider> <model>")
        sys.exit(2)

    video_id, provider, model = sys.argv[1], sys.argv[2], sys.argv[3]
    model_safe = model.replace("/", "_").replace(":", "_")
    out_dir = ROOT / "bench_results" / video_id / model_safe
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "log.txt"
    summary_path = out_dir / "summary.json"
    lots_path = out_dir / "lots.json"

    # If already complete, skip (makes resumable).
    if summary_path.exists():
        existing = json.loads(summary_path.read_text())
        if existing.get("status") == "ok":
            print(f"[{video_id}] {model}: SKIP (already complete)")
            return 0

    # Redirect all output to log file (prevents mangled interleaved prints
    # when many parallel runners are active).
    import contextlib
    log_f = open(log_path, "w")

    with contextlib.redirect_stdout(log_f), contextlib.redirect_stderr(log_f):
        print(f"=== Bench run: video={video_id} provider={provider} model={model} ===")

        from pipeline.transcriber import Segment
        from pipeline.aggregator import aggregate
        from pipeline.extractor import LLMClient, extract_lots

        vid_dir = ROOT / "output" / video_id
        transcript_data = json.loads((vid_dir / f"transcript_{video_id}.json").read_text())
        ocr_data = json.loads((vid_dir / f"ocr_results_{video_id}.json").read_text())
        segments = [Segment(**s) for s in transcript_data]
        windows = aggregate(segments, ocr_data)
        print(f"loaded {len(segments)} transcript segments, {len(ocr_data)} OCR rows, "
              f"{len(windows)} windows")

        t0 = time.time()
        try:
            client = LLMClient(provider=provider, model=model)
            lots = extract_lots(
                windows, client,
                prompt_path=ROOT / "prompts" / "extraction.txt",
                output_path=lots_path,
                verify_prompt_path=ROOT / "prompts" / "verify.txt",
            )
            elapsed = time.time() - t0
            summary = {
                "video_id": video_id,
                "provider": provider,
                "model": model,
                "elapsed_seconds": round(elapsed, 1),
                "lot_count": len(lots),
                "lots_with_price": sum(1 for l in lots if l.unit_price),
                "lots_without_price": sum(1 for l in lots if not l.unit_price),
                "llm_calls": client.n_calls,
                "input_tokens": client.input_tokens,
                "output_tokens": client.output_tokens,
                "status": "ok",
            }
        except Exception as e:
            elapsed = time.time() - t0
            summary = {
                "video_id": video_id,
                "provider": provider,
                "model": model,
                "elapsed_seconds": round(elapsed, 1),
                "status": "error",
                "error": str(e)[:400],
                "traceback": traceback.format_exc()[:2000],
            }
            traceback.print_exc()

    log_f.close()
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    # One terse line on the parent stderr so the orchestrator can watch
    print(
        f"[{video_id}] {model}: {summary.get('status')} "
        f"({elapsed:.0f}s, {summary.get('lot_count', '—')} lots)",
        file=sys.__stdout__, flush=True,
    )
    return 0 if summary["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())

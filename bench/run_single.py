"""Replay extraction with a content-addressed cache and explicit reasoning effort."""
import argparse
import contextlib
import hashlib
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if not __package__:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

from pipeline.aggregator import aggregate
from pipeline.checkpoint import write_json
from pipeline.extractor import LLMClient, extract_lots
from pipeline.transcriber import Segment
from pipeline.transcript_quality import check_transcript


def _run_fingerprint(configuration: dict, files: dict[str, Path]) -> tuple[str, dict]:
    manifest = {
        **configuration,
        "inputs": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in files.items()},
    }
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video_id")
    parser.add_argument("provider", choices=("openai", "openrouter"))
    parser.add_argument("model")
    parser.add_argument("--reasoning-effort", choices=("none", "low", "medium", "high", "xhigh", "max"))
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9_-]{11}", args.video_id):
        parser.error("video_id must be an 11-character YouTube video identifier")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_./:-]*", args.model):
        parser.error("invalid model identifier")
    if args.reasoning_effort and args.provider != "openai":
        parser.error("--reasoning-effort requires the OpenAI provider")

    load_dotenv(ROOT / ".env")
    video_id = args.video_id
    video_dir = ROOT / "output" / video_id
    files = {
        "transcript": video_dir / f"transcript_{video_id}.json",
        "ocr": video_dir / f"ocr_results_{video_id}.json",
        "prompt": ROOT / "prompts" / "extraction.txt",
        "verify_prompt": ROOT / "prompts" / "verify.txt",
        "parser": ROOT / "pipeline" / "extractor.py",
        "lot_schema": ROOT / "models" / "lot.py",
        "aggregator": ROOT / "pipeline" / "aggregator.py",
        "transcript_quality": ROOT / "pipeline" / "transcript_quality.py",
        "runner": Path(__file__),
        "lock": ROOT / "uv.lock",
    }
    video_info = video_dir / f"video_info_{video_id}.json"
    if video_info.is_file():
        files["video_info"] = video_info

    try:
        client = LLMClient(args.provider, args.model, reasoning_effort=args.reasoning_effort)
        options = client.completion_options()
        run_id, manifest = _run_fingerprint(
            {"video_id": video_id, "provider": args.provider, "model": args.model, "request_options": options},
            files,
        )
    except (OSError, ValueError) as exc:
        print(f"{video_id}: benchmark could not start ({exc})", file=sys.stderr)
        return 1

    model_safe = args.model.replace("/", "_").replace(":", "_")
    effort = options.get("reasoning_effort")
    variant = model_safe + (f"__{effort}" if effort else "")
    variant_dir = ROOT / "bench_results" / video_id / variant
    out_dir = variant_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    lots_path = out_dir / "lots.json"
    latest_path = variant_dir / "latest.json"

    try:
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
        lots = json.loads(lots_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        existing, lots = None, None
    if (isinstance(existing, dict) and existing.get("status") == "ok"
            and existing.get("run_id") == run_id and isinstance(lots, list)
            and existing.get("lot_count") == len(lots)):
        write_json(latest_path, {"run_id": run_id})
        print(f"[{video_id}] {variant}: SKIP ({run_id[:12]})")
        return 0

    started = time.monotonic()
    summary = {**manifest, "run_id": run_id}
    with (out_dir / "log.txt").open("w", encoding="utf-8") as log, contextlib.redirect_stdout(log):
        try:
            transcript = json.loads(files["transcript"].read_text(encoding="utf-8"))
            ocr = json.loads(files["ocr"].read_text(encoding="utf-8"))
            info = json.loads(video_info.read_text(encoding="utf-8")) if video_info.is_file() else {}
            quality = check_transcript([Segment(**row) for row in transcript], audio_duration=info.get("duration"))
            lots = extract_lots(
                aggregate(quality.segments, ocr),
                client,
                prompt_path=files["prompt"],
                output_path=lots_path,
                verify_prompt_path=files["verify_prompt"],
            )
            summary.update(status="ok", lot_count=len(lots))
        except Exception as exc:
            summary.update(status="error", error_type=type(exc).__name__)
            print(f"{type(exc).__name__}: {exc}")

    elapsed = time.monotonic() - started
    summary.update(
        elapsed_seconds=round(elapsed, 1), llm_calls=client.n_calls,
        input_tokens=client.input_tokens, output_tokens=client.output_tokens,
    )
    write_json(summary_path, summary)
    # Report failures for the current inputs instead of showing an older run.
    write_json(latest_path, {"run_id": run_id})
    print(f"[{video_id}] {variant}: {summary['status']} ({elapsed:.0f}s, {run_id[:12]})")
    return 0 if summary["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())

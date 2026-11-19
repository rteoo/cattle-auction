"""Orchestrate the benchmark matrix defined by VIDEOS and MODELS.
Runs up to N in parallel, each as a separate subprocess so failures in one
don't poison the others. Prints progress as each completes."""
import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

VIDEOS = [
    "TTNfx03Ubr8",
    "VDzRLEMUagA",
    "ZCwxnhUZjQM",
    "sKmUHExf464",
    "vWs74eYdyXo",
]

MODELS = [
    # The two options shipped in production (see pipeline/extractor.py
    # `_DEFAULT_MODELS`). Reference is Claude running this conversation —
    # produced manually, saved in bench_results/<video>/REFERENCE_claude/lots.json.
    ("openrouter", "google/gemini-2.5-flash-lite-preview-09-2025", None),
    ("openai", "gpt-4.1-mini", None),
]

MAX_PARALLEL = 4  # small matrix (10 runs) — low concurrency is fine


def run_one(video: str, provider: str, model: str, effort: str | None) -> tuple[str, str, str, int]:
    """Spawn run_single.py. Returns (video, model, outcome, returncode)."""
    cmd = [
        "uv", "run", "--frozen", "python", "bench/run_single.py",
        video, provider, model,
    ]
    if effort:
        cmd += ["--reasoning-effort", effort]
    try:
        result = subprocess.run(
            cmd, cwd=ROOT,
            capture_output=True, text=True, timeout=60 * 30,  # 30 min per run
        )
        output = (result.stdout or result.stderr).strip()
        outcome = output.splitlines()[-1] if output else "(no output)"
        return (video, model, outcome, result.returncode)
    except subprocess.TimeoutExpired:
        return (video, model, "TIMEOUT after 30m", -1)
    except Exception as e:
        return (video, model, f"spawn failed: {e}", -1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--luna", action="store_true", help="Compare GPT-6 Luna low and medium effort")
    parser.add_argument("--video", action="append", choices=VIDEOS, help="Restrict the replay to selected videos")
    args = parser.parse_args()
    videos = list(dict.fromkeys(args.video)) if args.video else VIDEOS
    models = [("openai", "gpt-6-luna", effort) for effort in ("low", "medium")] if args.luna else MODELS
    combos = [(v, p, m, e) for v in videos for (p, m, e) in models]
    print(f"Dispatching {len(combos)} combos with {MAX_PARALLEL}-way parallelism")
    print(f"  {len(videos)} videos × {len(models)} configurations")
    print()

    done = 0
    failed = []
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as pool:
        futures = {
            pool.submit(run_one, v, p, m, e): (v, m, e) for v, p, m, e in combos
        }
        for fut in as_completed(futures):
            v, m, e = futures[fut]
            video, model, outcome, rc = fut.result()
            done += 1
            status = "✓" if rc == 0 else "✗"
            print(f"[{done:>2}/{len(combos)}] {status} {outcome}")
            if rc != 0:
                failed.append((v, f"{m} ({e})" if e else m, outcome))

    print()
    print(f"Completed {done}, failures: {len(failed)}")
    for v, m, o in failed:
        print(f"  ✗ {v} / {m}: {o}")

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())

"""Orchestrate the full benchmark: 5 videos × 15 models = 75 combos.
Runs up to N in parallel, each as a separate subprocess so failures in one
don't poison the others. Prints progress as each completes."""
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
    ("openrouter", "google/gemini-2.5-flash-lite-preview-09-2025"),
    ("openai", "gpt-4.1-mini"),
]

MAX_PARALLEL = 4  # small matrix (10 runs) — low concurrency is fine


def run_one(video: str, provider: str, model: str) -> tuple[str, str, str, int]:
    """Spawn run_single.py. Returns (video, model, outcome, returncode)."""
    cmd = [
        "uv", "run", "python", "bench/run_single.py",
        video, provider, model,
    ]
    try:
        result = subprocess.run(
            cmd, cwd=ROOT,
            capture_output=True, text=True, timeout=60 * 30,  # 30 min per run
        )
        outcome = result.stdout.strip().splitlines()[-1] if result.stdout else "(no output)"
        return (video, model, outcome, result.returncode)
    except subprocess.TimeoutExpired:
        return (video, model, "TIMEOUT after 30m", -1)
    except Exception as e:
        return (video, model, f"spawn failed: {e}", -1)


def main():
    combos = [(v, p, m) for v in VIDEOS for (p, m) in MODELS]
    print(f"Dispatching {len(combos)} combos with {MAX_PARALLEL}-way parallelism")
    print(f"  {len(VIDEOS)} videos × {len(MODELS)} models")
    print()

    done = 0
    failed = []
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as pool:
        futures = {
            pool.submit(run_one, v, p, m): (v, p, m) for v, p, m in combos
        }
        for fut in as_completed(futures):
            v, p, m = futures[fut]
            video, model, outcome, rc = fut.result()
            done += 1
            status = "✓" if rc == 0 else "✗"
            print(f"[{done:>2}/{len(combos)}] {status} {outcome}")
            if rc != 0:
                failed.append((v, m, outcome))

    print()
    print(f"Completed {done}, failures: {len(failed)}")
    for v, m, o in failed:
        print(f"  ✗ {v} / {m}: {o}")

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())

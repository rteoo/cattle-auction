import json
import logging
import warnings
from pathlib import Path

from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

from pipeline.screenshotter import Screenshot

# Suppress PaddlePaddle / PaddleOCR verbose logging and connectivity banner
import os
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
logging.disable(logging.DEBUG)
warnings.filterwarnings("ignore", message=".*pin_memory.*")


def run_ocr(
    screenshots: list[Screenshot],
    output_path: Path,
    confidence_threshold: float = 0.4,
) -> dict[str, list[str]]:
    """
    Run PaddleOCR on all screenshots. Returns dict mapping timestamp_str → [text lines].
    Saves/loads ocr_results.json for checkpointing.
    """
    if output_path.exists():
        print(f"  OCR results already exist, loading from cache.")
        return json.loads(output_path.read_text())

    from paddleocr import PaddleOCR

    print("  Loading PaddleOCR (pt)...")
    # use_angle_cls=False: auction overlays are horizontal, skip rotation detection
    # (show_log was removed in PaddleOCR 2.8; logging is suppressed via logging.disable above)
    reader = PaddleOCR(lang="pt", use_angle_cls=False)

    results: dict[str, list[str]] = {}
    total = len(screenshots)

    with Progress(
        TextColumn("  [progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("[cyan]{task.fields[ts]}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("Running OCR", total=total, ts="")

        for shot in screenshots:
            progress.update(task, advance=1, ts=shot.timestamp_str)

            if not shot.path.exists():
                results[shot.timestamp_str] = []
                continue

            raw = reader.ocr(str(shot.path), cls=False)

            lines = []
            # raw is [[bbox, [text, confidence]], ...] or [None] when no text found
            if raw and raw[0]:
                for detection in raw[0]:
                    text, confidence = detection[1]
                    if confidence >= confidence_threshold:
                        lines.append(text)

            results[shot.timestamp_str] = lines

    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    return results

import json
import logging
import warnings
from pathlib import Path

from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

from pipeline.screenshotter import Screenshot


def run_ocr(
    screenshots: list[Screenshot],
    output_path: Path,
    confidence_threshold: float = 0.4,
) -> dict[str, list[str]]:
    """
    Run RapidOCR on all screenshots. Returns dict mapping timestamp_str → [text lines].
    Saves/loads ocr_results.json for checkpointing.
    """
    if output_path.exists():
        print(f"  OCR results already exist, loading from cache.")
        return json.loads(output_path.read_text())

    from rapidocr_onnxruntime import RapidOCR

    print("  Loading RapidOCR...")
    reader = RapidOCR()

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

            raw, _ = reader(str(shot.path))

            lines = []
            # raw is [[box, text, confidence], ...] or None when no text found
            if raw:
                for detection in raw:
                    text, confidence = detection[1], detection[2]
                    if confidence >= confidence_threshold:
                        lines.append(text)

            results[shot.timestamp_str] = lines

    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    return results

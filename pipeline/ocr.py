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
    provenance = _ocr_provenance(screenshots, confidence_threshold)
    if _valid_checkpoint(output_path):
        metadata_path = _provenance_path(output_path)
        cached_provenance = _load_provenance(metadata_path)
        if cached_provenance is None and not metadata_path.exists():
            result = json.loads(output_path.read_text(encoding="utf-8"))
            expected_timestamps = {shot.timestamp_str for shot in screenshots}
            if isinstance(result, dict) and set(result) == expected_timestamps:
                print("  OCR results already exist, adopting legacy cache.")
                _save_provenance(provenance, metadata_path)
                return result
            print("  Legacy OCR cache does not match screenshots; re-running OCR.")
        if cached_provenance == provenance:
            print(f"  OCR results already exist, loading from cache.")
            return json.loads(output_path.read_text(encoding="utf-8"))
        print("  OCR provenance changed, re-running OCR.")

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

    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    _save_provenance(provenance, _provenance_path(output_path))
    return results


def _provenance_path(path: Path) -> Path:
    return path.with_suffix(".meta.json")


def _valid_checkpoint(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _file_identity(path: Path) -> dict:
    identity = {"path": str(path.resolve())}
    try:
        stat = path.stat()
    except OSError:
        identity.update({"size": None, "mtime_ns": None})
    else:
        identity.update({"size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    return identity


def _ocr_provenance(screenshots: list[Screenshot], confidence_threshold: float) -> dict:
    return {
        "confidence_threshold": confidence_threshold,
        "screenshots": [
            {
                "seconds": shot.seconds,
                "timestamp_str": shot.timestamp_str,
                **_file_identity(shot.path),
            }
            for shot in screenshots
        ],
    }


def _load_provenance(path: Path) -> dict | None:
    if not _valid_checkpoint(path):
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _save_provenance(provenance: dict, path: Path) -> None:
    path.write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")

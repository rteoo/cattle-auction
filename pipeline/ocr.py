import hashlib
import json
from pathlib import Path

from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

from pipeline.checkpoint import write_json
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
        cached_results = _load_results(output_path)
        if cached_results is None:
            print("  OCR checkpoint is unreadable, re-running OCR.")
        elif (cached_provenance and cached_provenance.get("provenance") == provenance
              and _project_text(cached_provenance.get("frames")) == cached_results):
            print("  OCR results already exist, loading from cache.")
            return cached_results
        else:
            print("  OCR provenance changed, re-running OCR.")

    from rapidocr_onnxruntime import RapidOCR

    print("  Loading RapidOCR...")
    reader = RapidOCR()

    frames = []
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

            exists = shot.path.is_file()
            raw, _ = reader(str(shot.path)) if exists else (None, None)
            frames.append({
                "path": str(shot.path),
                "seconds": shot.seconds,
                "timestamp": shot.timestamp_str,
                "missing": not exists,
                "detections": [
                    {
                        "box": [[float(x), float(y)] for x, y in box] if box is not None else None,
                        "text": text,
                        "confidence": float(confidence),
                        "included": bool(confidence >= confidence_threshold),
                    }
                    for box, text, confidence in raw or []
                ],
            })

    results = _project_text(frames)
    write_json(output_path, results)
    write_json(_provenance_path(output_path), {"provenance": provenance, "frames": frames})
    return results


def _project_text(frames: list | None) -> dict[str, list[str]] | None:
    if not isinstance(frames, list):
        return None
    results: dict[str, list[str]] = {}
    try:
        for frame in frames:
            texts = results.setdefault(frame["timestamp"], [])
            for detection in frame["detections"]:
                if detection["included"]:
                    texts.append(detection["text"])
    except (KeyError, TypeError):
        return None
    return results


def _provenance_path(path: Path) -> Path:
    return path.with_suffix(".meta.json")


def _valid_checkpoint(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _file_identity(path: Path) -> dict:
    identity = {"path": str(path.resolve())}
    try:
        with path.open("rb") as frame:
            digest = hashlib.file_digest(frame, "sha256").hexdigest()
    except OSError:
        digest = None
    identity["sha256"] = digest
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


def _load_results(path: Path) -> dict[str, list[str]] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None

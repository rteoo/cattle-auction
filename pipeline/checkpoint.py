"""Crash-safe JSON checkpoint writes shared by every pipeline stage."""
import json
import os
from pathlib import Path


def write_json(path: Path, data) -> None:
    """Write `data` as JSON so readers see the old file or the complete new one.

    Resume trusts whatever is on disk, so a run killed mid-write must never
    leave truncated JSON at the checkpoint path. The data is written to a
    sibling temp file and renamed over the target, which is atomic on the
    same filesystem.
    """
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp_path, path)

"""Checkpoint writes are atomic: a crash leaves the old file or the new one."""
import json

import pytest

from pipeline import checkpoint


def test_interrupted_write_keeps_previous_checkpoint(monkeypatch, tmp_path):
    path = tmp_path / "lots.json"
    checkpoint.write_json(path, [{"lot_number": 1}])

    def killed(src, dst):
        raise KeyboardInterrupt

    monkeypatch.setattr(checkpoint.os, "replace", killed)
    with pytest.raises(KeyboardInterrupt):
        checkpoint.write_json(path, [{"lot_number": 1}, {"lot_number": 2}])

    assert json.loads(path.read_text(encoding="utf-8")) == [{"lot_number": 1}]


def test_write_replaces_existing_file_and_leaves_no_temp(tmp_path):
    path = tmp_path / "result.json"
    path.write_text("old", encoding="utf-8")

    checkpoint.write_json(path, {"cidade": "Goiânia"})

    assert json.loads(path.read_text(encoding="utf-8")) == {"cidade": "Goiânia"}
    assert "Goiânia" in path.read_text(encoding="utf-8")
    assert [p.name for p in tmp_path.iterdir()] == ["result.json"]

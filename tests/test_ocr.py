import json
import sys
import types

from pipeline import ocr
from pipeline.screenshotter import Screenshot


class _Reader:
    def __init__(self):
        self.calls = []

    def __call__(self, path):
        self.calls.append(path)
        return ([[None, "LOTE 1", 0.9]], None)


def test_ocr_records_screenshot_provenance_and_invalidates_changed_file(monkeypatch, tmp_path):
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"one")
    shot = Screenshot(0, "00:00:00", image)
    output = tmp_path / "ocr.json"
    reader = _Reader()
    monkeypatch.setitem(sys.modules, "rapidocr_onnxruntime", types.SimpleNamespace(RapidOCR=lambda: reader))

    ocr.run_ocr([shot], output)
    image.write_bytes(b"two")
    ocr.run_ocr([shot], output)

    assert len(reader.calls) == 2
    metadata = json.loads(output.with_suffix(".meta.json").read_text(encoding="utf-8"))
    assert metadata["screenshots"][0]["path"] == str(image.resolve())


def test_legacy_ocr_is_adopted_once(monkeypatch, tmp_path):
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"one")
    shot = Screenshot(0, "00:00:00", image)
    output = tmp_path / "ocr.json"
    output.write_text(json.dumps({"00:00:00": ["LEGACY"]}), encoding="utf-8")
    reader = _Reader()
    monkeypatch.setitem(sys.modules, "rapidocr_onnxruntime", types.SimpleNamespace(RapidOCR=lambda: reader))

    result = ocr.run_ocr([shot], output)

    assert result == {"00:00:00": ["LEGACY"]}
    assert reader.calls == []
    assert output.with_suffix(".meta.json").exists()


def test_legacy_ocr_with_different_timestamps_is_recomputed(monkeypatch, tmp_path):
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"one")
    shot = Screenshot(30, "00:00:30", image)
    output = tmp_path / "ocr.json"
    output.write_text(json.dumps({"00:00:00": ["STALE"]}), encoding="utf-8")
    reader = _Reader()
    monkeypatch.setitem(sys.modules, "rapidocr_onnxruntime", types.SimpleNamespace(RapidOCR=lambda: reader))

    result = ocr.run_ocr([shot], output)

    assert result == {"00:00:30": ["LOTE 1"]}
    assert reader.calls == [str(image)]

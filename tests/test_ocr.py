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
    assert metadata["provenance"]["screenshots"][0]["path"] == str(image.resolve())
    assert metadata["frames"][0]["detections"][0]["confidence"] == 0.9


def test_ocr_retains_boxes_and_multiple_frames_in_the_same_second(monkeypatch, tmp_path):
    first, second = tmp_path / "first.jpg", tmp_path / "second.jpg"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    box = [[1, 2], [3, 2], [3, 4], [1, 4]]

    class Reader:
        def __call__(self, image):
            text = "LOTE 1" if image == str(first) else "VENDIDO"
            return [[box, text, 0.9], [box, "incerto", 0.1]], None

    monkeypatch.setitem(sys.modules, "rapidocr_onnxruntime", types.SimpleNamespace(RapidOCR=Reader))
    output = tmp_path / "ocr.json"
    result = ocr.run_ocr([Screenshot(5.1, "00:00:05", first), Screenshot(5.8, "00:00:05", second)], output)
    assert result == {"00:00:05": ["LOTE 1", "VENDIDO"]}
    frames = json.loads(output.with_suffix(".meta.json").read_text(encoding="utf-8"))["frames"]
    assert [f["seconds"] for f in frames] == [5.1, 5.8]
    assert frames[0]["detections"][1] == {
        "box": box, "text": "incerto", "confidence": 0.1, "included": False,
    }


def test_legacy_ocr_is_rebuilt_with_evidence(monkeypatch, tmp_path):
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"one")
    shot = Screenshot(0, "00:00:00", image)
    output = tmp_path / "ocr.json"
    output.write_text(json.dumps({"00:00:00": ["LEGACY"]}), encoding="utf-8")
    reader = _Reader()
    monkeypatch.setitem(sys.modules, "rapidocr_onnxruntime", types.SimpleNamespace(RapidOCR=lambda: reader))

    result = ocr.run_ocr([shot], output)

    assert result == {"00:00:00": ["LOTE 1"]}
    assert reader.calls == [str(image)]
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


def test_corrupt_ocr_checkpoint_is_recomputed(monkeypatch, tmp_path):
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"one")
    shot = Screenshot(0, "00:00:00", image)
    output = tmp_path / "ocr.json"
    reader = _Reader()
    monkeypatch.setitem(sys.modules, "rapidocr_onnxruntime", types.SimpleNamespace(RapidOCR=lambda: reader))
    ocr.run_ocr([shot], output)
    output.write_text('{"00:00:00": ["LO', encoding="utf-8")

    assert ocr.run_ocr([shot], output) == {"00:00:00": ["LOTE 1"]}
    assert len(reader.calls) == 2

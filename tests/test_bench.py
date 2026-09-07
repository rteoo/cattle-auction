"""Benchmark reports must tolerate incomplete and unusable synthetic results."""

import json

import pytest

from bench import analyze


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=True), encoding="utf-8")


def test_analysis_handles_lots_without_summary(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(analyze, "RESULTS", tmp_path)
    monkeypatch.setattr(analyze, "VIDEOS", ["synthetic"])
    monkeypatch.setattr(analyze, "REFERENCE_VIDEOS", {"synthetic"})
    monkeypatch.setattr(analyze, "MODELS", [("test model", "model", "test", 0, 0)])

    _write_json(
        tmp_path / "synthetic" / "REFERENCE_claude" / "lots.json",
        [{"lot_number": 1, "unit_price": 3000}],
    )
    _write_json(
        tmp_path / "synthetic" / "model" / "lots.json",
        [{"lot_number": 1, "unit_price": 3000}],
    )

    analyze.main()

    output = capsys.readouterr().out
    assert "SECTION 1" in output
    assert "SECTION 3" in output
    assert "test model" in output


def test_compare_ignores_nonfinite_and_nonpositive_prices():
    ref = [
        {"lot_number": 1, "unit_price": 100},
        {"lot_number": 2, "unit_price": 0},
        {"lot_number": 3, "unit_price": float("nan")},
        {"lot_number": 4, "unit_price": 1e-308},
    ]
    model = [
        {"lot_number": 1, "unit_price": 110},
        {"lot_number": 2, "unit_price": float("inf")},
        {"lot_number": 3, "unit_price": 100},
        {"lot_number": 4, "unit_price": 1e308},
    ]

    result = analyze.compare(ref, model)

    assert result["mape"] == pytest.approx(0.1)


def test_analysis_skips_reference_without_usable_prices(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(analyze, "RESULTS", tmp_path)
    monkeypatch.setattr(analyze, "VIDEOS", ["synthetic"])
    monkeypatch.setattr(analyze, "REFERENCE_VIDEOS", {"synthetic"})
    monkeypatch.setattr(analyze, "MODELS", [])

    _write_json(
        tmp_path / "synthetic" / "REFERENCE_claude" / "lots.json",
        [
            {"lot_number": 1, "unit_price": None},
            {"lot_number": 2, "unit_price": 0},
            {"lot_number": 3, "unit_price": float("nan")},
        ],
    )

    analyze.main()

    assert "R$" not in capsys.readouterr().out

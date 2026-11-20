"""Tests for the single-URL stage wiring in main._run_single_url.

Every external stage is stubbed; what is under test is the glue: that the
transcript gate is applied, that sampling flags reach the screenshot stage,
and that the reported cost reflects what this run actually spent.
"""
import json
import sqlite3
import sys
from contextlib import closing
from pathlib import Path

import pytest
from click.testing import CliRunner

sys.path.insert(0, str(Path(__file__).parent.parent))

import main
from models.lot import AuctionResult, Lot
from pipeline.screenshotter import Screenshot
from pipeline.transcriber import Segment


class _FakeClient:
    def __init__(self, *args, **kwargs):
        self.input_tokens = 1_000_000
        self.output_tokens = 500_000
        self.n_calls = 3


def _install_stubs(monkeypatch, tmp_path, *, segments, duration=3600.0):
    """Stub every pipeline stage so only main's wiring is exercised."""
    captured = {}

    monkeypatch.setattr(main.downloader, "get_video_id", lambda url: "vid")
    monkeypatch.setattr(
        main.downloader, "get_video_info",
        lambda url, run_dir, video_id: {"title": "t", "description": "d", "duration": duration},
    )
    monkeypatch.setattr(
        main.downloader, "download_audio",
        lambda url, run_dir, video_id: run_dir / "audio.wav",
    )
    monkeypatch.setattr(
        main.downloader, "download_ocr_video",
        lambda url, run_dir, video_id, height: run_dir / "video.mp4",
    )

    def fake_transcribe(audio_path, output_path, **kwargs):
        output_path.write_text("[]", encoding="utf-8")
        return segments

    monkeypatch.setattr(main.transcriber_mod, "transcribe", fake_transcribe)

    def fake_screenshots(video_path, output_dir, video_id, **kwargs):
        captured.update(kwargs)
        return [Screenshot(seconds=0, timestamp_str="00:00:00", path=output_dir / "f.jpg")]

    monkeypatch.setattr(main.screenshotter, "extract_screenshots", fake_screenshots)
    monkeypatch.setattr(main.ocr, "run_ocr", lambda shots, output_path: {"00:00:00": ["LOTE 1"]})

    def fake_aggregate(segs, ocr_results, **kwargs):
        captured["aggregated_segments"] = segs
        return []

    monkeypatch.setattr(main.aggregator, "aggregate", fake_aggregate)
    monkeypatch.setattr(main.extractor, "LLMClient", _FakeClient)
    monkeypatch.setattr(main.extractor, "extract_lots", lambda *a, **k: [])
    monkeypatch.setattr(main.extractor, "extract_metadata", lambda *a, **k: {})

    return captured


def _run(tmp_path, **overrides):
    kwargs = dict(
        provider="openai",
        output_dir=str(tmp_path),
        transcriber="groq",
        whisper_model="medium",
        cpp_model=None,
        screenshot_interval=30,
        ocr_video_height=480,
        no_resume=False,
        show_metadata=False,
        show_summary=False,
        show_table=False,
    )
    kwargs.update(overrides)
    return main._run_single_url("https://youtu.be/vid", **kwargs)


def test_hallucinated_transcript_loop_is_stripped_before_aggregation(monkeypatch, tmp_path):
    """A repeated-phrase loop must not reach the LLM as if it were real speech."""
    loop = [Segment(start=float(i), end=float(i) + 1, text="obrigado") for i in range(10)]
    real = [Segment(start=20.0, end=25.0, text="lote quarenta e dois, doze garrotes")]
    captured = _install_stubs(monkeypatch, tmp_path, segments=loop + real)

    _run(tmp_path)

    passed = captured["aggregated_segments"]
    assert [s.text for s in passed] == ["obrigado", "lote quarenta e dois, doze garrotes"]


def test_frame_sampling_flags_reach_the_screenshot_stage(monkeypatch, tmp_path):
    captured = _install_stubs(
        monkeypatch, tmp_path,
        segments=[Segment(start=0.0, end=600.0, text="fala real do leiloeiro")],
    )

    _run(tmp_path, frame_sampling="scene", safety_interval=45)

    assert captured["sampling"] == "scene"
    assert captured["safety_interval"] == 45


def test_cost_bills_transcription_when_whisper_actually_ran(monkeypatch, tmp_path):
    _install_stubs(
        monkeypatch, tmp_path,
        segments=[Segment(start=0.0, end=3000.0, text="fala real do leiloeiro")],
        duration=3600.0,
    )

    result = _run(tmp_path)

    # 1h of Groq audio at $0.04/h, plus 1M in + 0.5M out on GPT-6 Luna.
    assert result.cost_usd == round(0.04 + 0.10 + 0.25, 6)


def test_cached_transcript_is_not_billed_again(monkeypatch, tmp_path):
    """A resumed run did not call Groq, so it must not report Groq's cost."""
    _install_stubs(
        monkeypatch, tmp_path,
        segments=[Segment(start=0.0, end=3000.0, text="fala real do leiloeiro")],
        duration=3600.0,
    )
    # Pre-create the checkpoint the way an earlier run would have left it.
    run_dir = tmp_path / "vid"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "transcript_vid.json").write_text("[]", encoding="utf-8")

    result = _run(tmp_path)

    assert result.cost_usd == round(0.10 + 0.25, 6)


def test_legacy_video_info_duration_is_derived_for_fresh_groq_transcription(monkeypatch, tmp_path):
    _install_stubs(
        monkeypatch,
        tmp_path,
        segments=[Segment(start=0.0, end=1200.0, text="fala real do leiloeiro")],
    )
    monkeypatch.setattr(
        main.downloader,
        "get_video_info",
        lambda url, run_dir, video_id: {"title": "t", "description": "d"},
    )
    monkeypatch.setattr(main.transcriber_mod, "_audio_duration", lambda path: 1800.0)

    result = _run(tmp_path)

    # 30 minutes of fresh Groq transcription costs $0.02.
    assert result.cost_usd == round(0.02 + 0.10 + 0.25, 6)


def test_stale_transcript_provenance_is_billed_as_a_fresh_groq_run(monkeypatch, tmp_path):
    _install_stubs(
        monkeypatch,
        tmp_path,
        segments=[Segment(start=0.0, end=3000.0, text="fala real do leiloeiro")],
        duration=3600.0,
    )
    run_dir = tmp_path / "vid"
    run_dir.mkdir(parents=True, exist_ok=True)
    transcript_path = run_dir / "transcript_vid.json"
    transcript_path.write_text("[]", encoding="utf-8")
    transcript_path.with_suffix(".meta.json").write_text(
        json.dumps({"backend": "mlx", "effective_model": "medium", "audio": {}}),
        encoding="utf-8",
    )

    result = _run(tmp_path)

    assert result.cost_usd == round(0.04 + 0.10 + 0.25, 6)


def test_cli_database_is_opt_in_and_keeps_json_output(monkeypatch, tmp_path):
    _install_stubs(monkeypatch, tmp_path, segments=[])
    monkeypatch.setattr(
        main.extractor, "extract_lots",
        lambda *a, **k: [Lot(lot_number="08B", sold=True, unit_price="3.100,50")],
    )
    output = tmp_path / "output"
    db_path = output / "history.sqlite3"
    args = ["https://youtu.be/vid", "--output-dir", str(output), "--no-table", "--no-summary"]
    runner = CliRunner()

    result = runner.invoke(main.cli, args)
    assert result.exit_code == 0, result.output
    assert not db_path.exists()

    result = runner.invoke(main.cli, [*args, "--sqlite-db", str(db_path)])
    assert result.exit_code == 0, result.output
    assert json.loads((output / "vid" / "result_vid.json").read_text())["lots"][0]["lot_number"] == "08B"
    with closing(sqlite3.connect(db_path)) as db:
        assert db.execute("SELECT lot_number, unit_price_cents FROM confirmed_prices").fetchall() == [
            ("08B", 310050),
        ]


def test_batch_stores_successful_results_and_reports_failed_writes(monkeypatch, tmp_path):
    def fake_run(url, **kwargs):
        video_id = url.rsplit("/", 1)[-1]
        lot = Lot(lot_number="00", sold=True, unit_price=3000)
        lots = [lot, lot] if video_id == "bad" else [lot]
        return AuctionResult(video_id=video_id, video_url=url, total_lots=len(lots), lots=lots)

    monkeypatch.setattr(main, "_run_single_url", fake_run)
    db_path = tmp_path / "history.sqlite3"
    result = CliRunner().invoke(main.cli, [
        "https://youtu.be/before", "https://youtu.be/bad", "https://youtu.be/after",
        "--sqlite-db", str(db_path), "--output-dir", str(tmp_path / "output"),
        "--batch-name", "history",
    ])

    assert result.exit_code == 1
    report = json.loads((tmp_path / "output" / "batches" / "history" / "batch_summary.json").read_text())
    assert report["totals"]["successful"] == 2
    assert report["totals"]["failed"] == 1
    with closing(sqlite3.connect(db_path)) as db:
        assert db.execute("SELECT video_id FROM auctions ORDER BY video_id").fetchall() == [
            ("after",), ("before",),
        ]


def test_invalid_history_path_fails_before_a_paid_run(monkeypatch, tmp_path):
    db_path = tmp_path / "invalid.sqlite3"
    db_path.write_text("not a database")

    def unexpected_run(*args, **kwargs):
        pytest.fail("The pipeline should not run with an invalid history database")

    monkeypatch.setattr(main, "_run_single_url", unexpected_run)
    result = CliRunner().invoke(main.cli, ["https://youtu.be/vid", "--sqlite-db", str(db_path)])

    assert result.exit_code == 1
    assert "Could not open SQLite history" in result.output

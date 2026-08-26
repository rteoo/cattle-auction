"""Tests for the single-URL stage wiring in main._run_single_url.

Every external stage is stubbed; what is under test is the glue: that the
transcript gate is applied, that sampling flags reach the screenshot stage,
and that the reported cost reflects what this run actually spent.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import main
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
    monkeypatch.setattr(main.extractor, "default_model", lambda provider: "gpt-4.1-mini")
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

    # 1h of Groq audio at $0.04/h, plus 1M in + 0.5M out on gpt-4.1-mini
    # ($0.40 + $0.80).
    assert result.cost_usd == round(0.04 + 0.40 + 0.80, 6)


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

    assert result.cost_usd == round(0.40 + 0.80, 6)

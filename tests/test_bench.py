"""Benchmark reports must tolerate incomplete and unusable synthetic results."""

import json

import pytest

from bench import analyze, run_single


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


def test_missing_or_unconfirmed_prices_cannot_inflate_coverage():
    ref = [
        {"lot_number": 1, "sold": True, "unit_price": 3000},
        {"lot_number": 2, "sold": True, "unit_price": 4000},
        {"lot_number": 3, "sold": False},
    ]
    model = [
        {"lot_number": 1, "sold": True, "unit_price": 3000},
        {"lot_number": 2, "sold": None, "unit_price": 4000},
        {"lot_number": 4, "sold": True, "unit_price": 2000},
    ]
    score = analyze.compare(ref, model)
    assert score["precision"] == pytest.approx(2 / 3)
    assert score["coverage"] == pytest.approx(2 / 3)
    assert score["sold_status_accuracy"] == pytest.approx(1 / 3)
    assert (score["hammer_price_count"], score["hammer_price_expected"]) == (1, 2)
    assert (score["mape"], score["mape_n"]) == (0, 1)


def test_run_identity_depends_on_effort_and_artifact_contents(tmp_path):
    evidence = tmp_path / "transcript.json"
    evidence.write_text("audio first", encoding="utf-8")
    files = {"transcript": evidence}
    low, _ = run_single._run_fingerprint({"reasoning_effort": "low"}, files)
    medium, _ = run_single._run_fingerprint({"reasoning_effort": "medium"}, files)
    assert low != medium
    evidence.write_text("audio changed", encoding="utf-8")
    changed, _ = run_single._run_fingerprint({"reasoning_effort": "medium"}, files)
    assert medium != changed


def test_analyzer_loads_the_pointed_run_instead_of_legacy_results(tmp_path):
    fingerprint = "a" * 64
    _write_json(tmp_path / "latest.json", {"run_id": fingerprint})
    _write_json(tmp_path / "summary.json", {"status": "ok"})
    _write_json(tmp_path / "lots.json", [{"lot_number": 99}])
    _write_json(tmp_path / fingerprint / "summary.json", {"run_id": fingerprint, "status": "ok"})
    _write_json(tmp_path / fingerprint / "lots.json", [])
    summary, lots = analyze._load_run(tmp_path)
    assert summary["run_id"] == fingerprint
    assert lots == []


def test_replay_resumes_only_the_same_effort_and_inputs(tmp_path, monkeypatch):
    from pipeline.extractor import LLMClient

    video_id = "fixture0001"
    video_dir = tmp_path / "output" / video_id
    transcript = video_dir / f"transcript_{video_id}.json"
    _write_json(transcript, [{"start": 0, "end": 30, "text": "Lote 42"}])
    _write_json(video_dir / f"ocr_results_{video_id}.json", {})
    for name in ("prompts/extraction.txt", "prompts/verify.txt", "pipeline/extractor.py",
                 "models/lot.py", "pipeline/aggregator.py", "pipeline/transcript_quality.py", "uv.lock"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture", encoding="utf-8")

    class OfflineClient(LLMClient):
        def __init__(self, provider, model, *, reasoning_effort):
            self.provider, self.model, self.reasoning_effort = provider, model, reasoning_effort
            self.n_calls = self.input_tokens = self.output_tokens = 0

    calls = []

    def fake_extract(windows, client, *, output_path, **kwargs):
        calls.append(client.completion_options()["reasoning_effort"])
        _write_json(output_path, [])
        return []

    monkeypatch.setattr(run_single, "ROOT", tmp_path)
    monkeypatch.setattr(run_single, "LLMClient", OfflineClient)
    monkeypatch.setattr(run_single, "extract_lots", fake_extract)
    args = [video_id, "openai", "gpt-6-luna", "--reasoning-effort"]
    assert run_single.main([*args, "low"]) == 0
    assert run_single.main([*args, "low"]) == 0
    assert run_single.main([*args, "medium"]) == 0
    _write_json(transcript, [{"start": 0, "end": 30, "text": "Lote 43"}])
    assert run_single.main([*args, "medium"]) == 0
    assert calls == ["low", "medium", "medium"]
    directory = tmp_path / "bench_results" / video_id / "gpt-6-luna__medium"
    assert len(list(directory.glob("*/summary.json"))) == 2


def test_price_median_stays_finite_at_float_limits():
    assert analyze._median_price([1e308, 1e308]) == 1e308
    assert analyze._median_price([5e-324, 5e-324]) == 5e-324
    assert analyze._median_price([100, 200]) == 150


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


def test_benchmark_converts_audio_when_only_the_video_is_cached(tmp_path, monkeypatch):
    import subprocess

    import benchmark

    (tmp_path / "video_vid.mp4").write_bytes(b"cached video")
    commands = []

    def fake_run(cmd, check, env):
        commands.append(cmd)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    benchmark.download_small("https://www.youtube.com/watch?v=vid", tmp_path, "vid")

    assert len(commands) == 1
    assert commands[0][commands[0].index("-c:a") + 2] == str(tmp_path / "audio_vid.wav")

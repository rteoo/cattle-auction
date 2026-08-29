import json
import os
import sys
import types
from pathlib import Path

from pipeline import transcriber


def test_transcribe_defaults_to_groq_and_records_provenance(monkeypatch, tmp_path):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"audio")
    output = tmp_path / "transcript.json"
    calls = []

    monkeypatch.setattr(
        transcriber,
        "_transcribe_groq",
        lambda path: calls.append("groq") or [transcriber.Segment(0.0, 1.0, "fala")],
    )

    result = transcriber.transcribe(audio, output)

    assert calls == ["groq"]
    assert result[0].text == "fala"
    metadata = json.loads(output.with_suffix(".meta.json").read_text(encoding="utf-8"))
    assert metadata["backend"] == "groq"
    assert metadata["effective_model"] == transcriber._GROQ_MODEL
    assert metadata["audio"]["size"] == audio.stat().st_size


def test_transcribe_recomputes_when_model_provenance_changes(monkeypatch, tmp_path):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"audio")
    output = tmp_path / "transcript.json"
    calls = []
    monkeypatch.setattr(
        transcriber,
        "_transcribe_mlx",
        lambda path, model: calls.append(("mlx", model)) or [],
    )
    transcriber.transcribe(audio, output, backend="mlx", whisper_model="medium")
    transcriber.transcribe(audio, output, backend="mlx", whisper_model="small")

    assert calls == [("mlx", "medium"), ("mlx", "small")]


def test_cpp_model_file_is_part_of_transcript_provenance(tmp_path):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"audio")
    first_model = tmp_path / "first.bin"
    first_model.write_bytes(b"one")
    second_model = tmp_path / "second.bin"
    second_model.write_bytes(b"two")

    first = transcriber._transcript_provenance(audio, "cpp", "medium", cpp_model_path=first_model)
    second = transcriber._transcript_provenance(audio, "cpp", "medium", cpp_model_path=second_model)

    assert first != second


def test_auto_detected_cpp_model_is_part_of_checkpoint_provenance(monkeypatch, tmp_path):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"audio")
    output = tmp_path / "transcript.json"
    first_model = tmp_path / "first.bin"
    first_model.write_bytes(b"one")
    second_model = tmp_path / "second.bin"
    second_model.write_bytes(b"two")
    detected_model = first_model
    calls = []

    monkeypatch.setattr(transcriber, "_find_cpp_model", lambda model: detected_model)
    monkeypatch.setattr(
        transcriber,
        "_transcribe_cpp",
        lambda path, model, model_path: calls.append(model_path) or [],
    )

    transcriber.transcribe(audio, output, backend="cpp", whisper_model="medium")
    detected_model = second_model

    assert calls == [first_model]
    assert not transcriber.transcript_checkpoint_matches(
        audio, output, backend="cpp", whisper_model="medium",
    )


def test_cached_cpp_transcript_loads_when_auto_detected_model_is_unavailable(monkeypatch, tmp_path):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"audio")
    output = tmp_path / "transcript.json"
    model = tmp_path / "model.bin"
    model.write_bytes(b"model")
    calls = []

    monkeypatch.setattr(transcriber, "_find_cpp_model", lambda name: model)
    monkeypatch.setattr(
        transcriber,
        "_transcribe_cpp",
        lambda path, name, model_path: calls.append(model_path)
        or [transcriber.Segment(0.0, 1.0, "cached")],
    )
    transcriber.transcribe(audio, output, backend="cpp", whisper_model="medium")
    monkeypatch.setattr(
        transcriber,
        "_find_cpp_model",
        lambda name: (_ for _ in ()).throw(FileNotFoundError("model missing")),
    )

    assert transcriber.transcript_checkpoint_matches(
        audio, output, backend="cpp", whisper_model="medium",
    )
    result = transcriber.transcribe(audio, output, backend="cpp", whisper_model="medium")

    assert calls == [model]
    assert result[0].text == "cached"


def test_legacy_transcript_is_adopted_once(monkeypatch, tmp_path):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"audio")
    output = tmp_path / "transcript.json"
    output.write_text(json.dumps([{"start": 0.0, "end": 1.0, "text": "legacy"}]), encoding="utf-8")
    calls = []
    monkeypatch.setattr(transcriber, "_transcribe_groq", lambda path: calls.append(True) or [])

    result = transcriber.transcribe(audio, output, backend="groq")

    assert result[0].text == "legacy"
    assert calls == []
    assert output.with_suffix(".meta.json").exists()


def test_groq_chunking_includes_fractional_tail(monkeypatch, tmp_path):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"audio")
    mp3 = audio.with_suffix(".mp3")
    with mp3.open("wb") as handle:
        handle.seek(transcriber._GROQ_MAX_BYTES + 1)
        handle.write(b"x")

    class FakeGroq:
        def __init__(self, **kwargs):
            pass

    monkeypatch.setenv("GROQ_API_KEY", "test")
    monkeypatch.setitem(sys.modules, "groq", types.SimpleNamespace(Groq=FakeGroq))
    monkeypatch.setattr(transcriber, "_audio_duration", lambda path: 1800.5)
    offsets = []
    monkeypatch.setattr(
        transcriber,
        "_groq_call",
        lambda client, path, offset: offsets.append(offset) or [],
    )

    def fake_run(cmd, check):
        output = Path(cmd[cmd.index("-codec:a") + 2])
        output.write_bytes(b"chunk")

    monkeypatch.setattr(transcriber.subprocess, "run", fake_run)
    transcriber._transcribe_groq(audio)

    assert offsets == [0, 900, 1800]


def test_derived_audio_cache_must_not_predate_its_source(tmp_path):
    source = tmp_path / "audio.wav"
    derived = tmp_path / "audio.mp3"
    derived.write_bytes(b"old")
    source.write_bytes(b"newer source")
    os.utime(derived, ns=(1_000_000_000, 1_000_000_000))
    os.utime(source, ns=(2_000_000_000, 2_000_000_000))

    assert not transcriber._is_fresh_nonempty_file(derived, source)

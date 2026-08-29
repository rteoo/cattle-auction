import sys
import types
from pathlib import Path

from pipeline import transcriber


def test_transcribe_defaults_to_groq(monkeypatch, tmp_path):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"audio")
    output = tmp_path / "transcript.json"
    calls = []

    monkeypatch.setattr(
        transcriber,
        "_transcribe_groq",
        lambda path: calls.append(path) or [transcriber.Segment(0.0, 1.0, "fala")],
    )

    result = transcriber.transcribe(audio, output)

    assert calls == [audio]
    assert result[0].text == "fala"


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

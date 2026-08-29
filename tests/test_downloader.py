"""Tests for YouTube download command selection."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline import downloader


def test_get_video_id_rejects_untrusted_host_without_fallback(monkeypatch):
    class FailYDL:
        def __init__(self, *args, **kwargs):
            raise AssertionError("yt-dlp must not resolve an untrusted host")

    monkeypatch.setattr(downloader.yt_dlp, "YoutubeDL", FailYDL)

    with pytest.raises(ValueError, match="YouTube"):
        downloader.get_video_id("https://youtube.com.evil.test/watch?v=dQw4w9WgXcQ")


def test_get_video_id_rejects_invalid_video_id():
    with pytest.raises(ValueError, match="video ID"):
        downloader.get_video_id("https://www.youtube.com/watch?v=../../secret")


@pytest.mark.parametrize("video_id", ["../secret", "a:b", "vid*", ""])
def test_output_path_video_id_rejects_unsafe_filename_characters(video_id, tmp_path):
    with pytest.raises(ValueError, match="URL-safe"):
        downloader.download_ocr_video("https://www.youtube.com/", tmp_path, video_id)


def test_download_audio_uses_audio_only_format_before_wav_conversion(monkeypatch, tmp_path):
    calls = []

    def fake_run(cmd, check):
        calls.append(cmd)
        if cmd[0] == "yt-dlp":
            (tmp_path / "audio_source_vid.m4a").write_text("audio", encoding="utf-8")
        elif cmd[0] == "ffmpeg":
            assert Path(cmd[cmd.index("-i") + 1]).exists()
            Path(cmd[cmd.index("-c:a") + 2]).write_text("wav", encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(downloader.subprocess, "run", fake_run)

    audio_path = downloader.download_audio(
        "https://www.youtube.com/watch?v=vid",
        tmp_path,
        "vid",
    )

    assert audio_path == tmp_path / "audio_vid.wav"
    assert calls[0][0] == "yt-dlp"
    assert calls[0][calls[0].index("-f") + 1] == "ba[ext=m4a]/ba"
    assert "bestvideo" not in calls[0]
    assert calls[1][0] == "ffmpeg"
    assert calls[1][calls[1].index("-i") + 1] == str(tmp_path / "audio_source_vid.m4a")


def test_download_ocr_video_defaults_to_480p(monkeypatch, tmp_path):
    calls = []

    def fake_run(cmd, check):
        calls.append(cmd)
        Path(cmd[cmd.index("-o") + 1]).write_text("video", encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(downloader.subprocess, "run", fake_run)

    video_path = downloader.download_ocr_video(
        "https://www.youtube.com/watch?v=vid",
        tmp_path,
        "vid",
    )

    assert video_path == tmp_path / "video_ocr_vid_480p.mp4"
    assert calls[0][0] == "yt-dlp"
    assert "height<=480" in calls[0][calls[0].index("-f") + 1]


def test_download_ocr_video_supports_720p_alternative(monkeypatch, tmp_path):
    calls = []

    def fake_run(cmd, check):
        calls.append(cmd)
        Path(cmd[cmd.index("-o") + 1]).write_text("video", encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(downloader.subprocess, "run", fake_run)

    video_path = downloader.download_ocr_video(
        "https://www.youtube.com/watch?v=vid",
        tmp_path,
        "vid",
        height=720,
    )

    assert video_path == tmp_path / "video_ocr_vid_720p.mp4"
    assert "height<=720" in calls[0][calls[0].index("-f") + 1]


def test_zero_byte_ocr_video_is_redownloaded(monkeypatch, tmp_path):
    video_path = tmp_path / "video_ocr_vid_480p.mp4"
    video_path.write_bytes(b"")
    calls = []

    def fake_run(cmd, check):
        calls.append(cmd)
        Path(cmd[cmd.index("-o") + 1]).write_bytes(b"video")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(downloader.subprocess, "run", fake_run)

    assert downloader.download_ocr_video("https://www.youtube.com/watch?v=vid", tmp_path, "vid") == video_path
    assert len(calls) == 1
    assert video_path.read_bytes() == b"video"


def test_zero_byte_audio_checkpoint_is_rebuilt(monkeypatch, tmp_path):
    source = tmp_path / "audio_source_vid.m4a"
    source.write_bytes(b"source")
    audio_path = tmp_path / "audio_vid.wav"
    audio_path.write_bytes(b"")
    calls = []

    def fake_run(cmd, check):
        calls.append(cmd)
        output = Path(cmd[cmd.index("-c:a") + 2])
        output.write_bytes(b"wav")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(downloader.subprocess, "run", fake_run)

    assert downloader.download_audio("https://www.youtube.com/watch?v=vid", tmp_path, "vid") == audio_path
    assert len(calls) == 1
    assert audio_path.read_bytes() == b"wav"


def test_audio_checkpoint_older_than_source_is_rebuilt(monkeypatch, tmp_path):
    source = tmp_path / "audio_source_vid.m4a"
    source.write_bytes(b"new source")
    audio_path = tmp_path / "audio_vid.wav"
    audio_path.write_bytes(b"stale wav")
    os.utime(audio_path, ns=(1_000_000_000, 1_000_000_000))
    os.utime(source, ns=(2_000_000_000, 2_000_000_000))
    calls = []

    def fake_run(cmd, check):
        calls.append(cmd)
        output = Path(cmd[cmd.index("-c:a") + 2])
        output.write_bytes(b"fresh wav")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(downloader.subprocess, "run", fake_run)

    assert downloader.download_audio("https://www.youtube.com/watch?v=vid", tmp_path, "vid") == audio_path
    assert len(calls) == 1
    assert audio_path.read_bytes() == b"fresh wav"

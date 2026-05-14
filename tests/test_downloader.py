"""Tests for YouTube download command selection."""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline import downloader


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

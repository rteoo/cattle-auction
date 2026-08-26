"""Tests for the Whisper transcript quality gate."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.transcriber import Segment
from pipeline.transcript_quality import check_transcript


def _seg(start, end, text):
    return Segment(start=start, end=end, text=text)


class TestCleanTranscript:
    def test_clean_transcript_passes_through_as_ok(self):
        segments = [
            _seg(0.0, 5.0, "Bom dia, seja bem-vindo ao leilão."),
            _seg(5.0, 10.0, "Lote número um, cinco cabeças de bezerro Nelore."),
            _seg(10.0, 15.0, "Fechado por três mil reais por cabeça."),
        ]
        result = check_transcript(segments)
        assert result.status == "ok"
        assert result.warning == ""
        assert result.dropped_segments == 0
        assert result.segments == segments


class TestEmptyTranscript:
    def test_no_segments_is_empty(self):
        result = check_transcript([])
        assert result.status == "empty"
        assert result.segments == []

    def test_all_blank_segments_is_empty(self):
        segments = [_seg(0.0, 1.0, ""), _seg(1.0, 2.0, "   ")]
        result = check_transcript(segments)
        assert result.status == "empty"
        assert result.segments == []


class TestCaptionCredit:
    def test_caption_credit_only_is_empty(self):
        segments = [_seg(0.0, 3.0, "Legendas pela comunidade Amara.org")]
        result = check_transcript(segments)
        assert result.status == "empty"
        assert result.segments == []

    def test_caption_credit_mixed_with_real_speech_drops_only_credit(self):
        segments = [
            _seg(0.0, 3.0, "Legendas pela comunidade Amara.org"),
            _seg(3.0, 8.0, "Lote número dois, três cabeças de garrote."),
        ]
        result = check_transcript(segments)
        assert result.status == "degraded"
        assert result.dropped_segments == 1
        assert len(result.segments) == 1
        assert result.segments[0].text == "Lote número dois, três cabeças de garrote."
        assert "caption-credit" in result.warning
        assert "1" in result.warning


class TestRepetitionLoop:
    def test_ten_segment_repetition_loop_collapsed_to_one(self):
        loop = [_seg(float(i), float(i + 1), "vamos, vamos, vamos") for i in range(10)]
        result = check_transcript(loop)
        assert result.status == "degraded"
        assert len(result.segments) == 1
        assert result.dropped_segments == 9
        assert "repeated-phrase" in result.warning

    def test_three_segment_repeat_not_collapsed(self):
        segments = [_seg(float(i), float(i + 1), "vamos, vamos, vamos") for i in range(3)]
        result = check_transcript(segments)
        assert result.status == "ok"
        assert result.dropped_segments == 0
        assert len(result.segments) == 3


class TestSparseCoverage:
    def test_sparse_coverage_flagged_as_degraded(self):
        # 3 seconds of speech spans out of a 60-second audio track (5% coverage).
        segments = [_seg(0.0, 3.0, "Bom dia a todos.")]
        result = check_transcript(segments, audio_duration=60.0)
        assert result.status == "degraded"
        assert "5.0%" in result.warning
        assert "OCR" in result.warning

    def test_sparse_rule_skipped_without_audio_duration(self):
        segments = [_seg(0.0, 3.0, "Bom dia a todos.")]
        result = check_transcript(segments, audio_duration=None)
        assert result.status == "ok"
        assert result.warning == ""

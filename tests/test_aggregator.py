"""Tests for pipeline/aggregator.py — window creation and overlap logic."""
import pytest

from pipeline.aggregator import aggregate, _fmt, _parse_ts
from pipeline.transcriber import Segment


# ── helpers ──────────────────────────────────────────────────────────────────

def seg(start: float, end: float, text: str) -> Segment:
    return Segment(start=start, end=end, text=text)


# ── _fmt / _parse_ts ─────────────────────────────────────────────────────────

class TestHelpers:
    def test_fmt_zero(self):
        assert _fmt(0) == "00:00:00"

    def test_fmt_one_hour(self):
        assert _fmt(3600) == "01:00:00"

    def test_fmt_mixed(self):
        assert _fmt(3661) == "01:01:01"

    def test_parse_ts_zero(self):
        assert _parse_ts("00:00:00") == 0

    def test_parse_ts_one_hour(self):
        assert _parse_ts("01:00:00") == 3600

    def test_parse_ts_mixed(self):
        assert _parse_ts("01:01:01") == 3661

    def test_fmt_parse_roundtrip(self):
        for secs in [0, 30, 600, 3600, 7265]:
            assert _parse_ts(_fmt(secs)) == secs


# ── aggregate ────────────────────────────────────────────────────────────────

class TestAggregate:
    def test_empty_segments_returns_empty(self):
        assert aggregate([], {}) == []

    def test_single_short_segment_one_window(self):
        segments = [seg(0, 30, "Lote 1")]
        windows = aggregate(segments, {})
        assert len(windows) == 1

    def test_window_label_format(self):
        segments = [seg(0, 30, "text")]
        windows = aggregate(segments, {})
        assert windows[0].label == "00:00:00 - 00:10:00"

    def test_transcript_included_in_window(self):
        segments = [seg(0, 30, "Lote 1 bezerro")]
        windows = aggregate(segments, {})
        assert "Lote 1 bezerro" in windows[0].combined_text
        assert "ÁUDIO:" in windows[0].combined_text

    def test_ocr_included_in_window(self):
        segments = [seg(0, 30, "text")]
        ocr = {"00:00:30": ["LOTE 1", "R$ 3.100"]}
        windows = aggregate(segments, ocr)
        assert "LOTE 1" in windows[0].combined_text
        assert "TELA:" in windows[0].combined_text

    def test_empty_ocr_texts_excluded(self):
        segments = [seg(0, 30, "text")]
        ocr = {"00:00:30": []}  # empty text list
        windows = aggregate(segments, ocr)
        assert "TELA:" not in windows[0].combined_text

    def test_overlap_segment_appears_in_two_windows(self):
        """A segment right at the boundary should appear in both windows due to overlap."""
        # window 1: 0–600, window 2: 540–1140 (540 = 600 - 60 overlap)
        segments = [
            seg(0, 10, "early"),
            seg(570, 580, "boundary lot"),  # falls in overlap zone
            seg(700, 710, "late"),
        ]
        windows = aggregate(segments, {})
        assert len(windows) == 2
        # boundary lot should appear in both windows
        assert "boundary lot" in windows[0].combined_text
        assert "boundary lot" in windows[1].combined_text

    def test_segment_not_in_window_excluded(self):
        segments = [seg(0, 10, "early"), seg(700, 710, "late")]
        windows = aggregate(segments, {})
        # "late" should only be in window 2 (starts at 540)
        assert "late" not in windows[0].combined_text
        assert "late" in windows[1].combined_text

    def test_empty_window_gets_placeholder(self):
        # Segment at start, then a long gap with no content in middle window
        segments = [seg(0, 1, "start"), seg(1800, 1801, "end")]
        windows = aggregate(segments, {})
        # Middle window (540-1140) should be empty
        middle = windows[1]
        assert "(sem conteúdo)" in middle.combined_text

    def test_multiple_ocr_entries_joined_with_pipe(self):
        segments = [seg(0, 30, "text")]
        ocr = {"00:00:30": ["LOTE 5", "Nelore", "R$ 2.800"]}
        windows = aggregate(segments, ocr)
        assert "LOTE 5 | Nelore | R$ 2.800" in windows[0].combined_text

    def test_window_count_matches_duration(self):
        """20-minute video with 10-min windows and 1-min overlap → 3 windows."""
        segments = [seg(0, 1199, "long auction")]
        windows = aggregate(segments, {})
        # window 1: 0–600, window 2: 540–1140, window 3: 1080–1680
        assert len(windows) == 3

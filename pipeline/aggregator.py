import re
from dataclasses import dataclass

from pipeline.transcriber import Segment

# Matches broadcast clock overlays like "19:59:41" or "20:05:08"
# (real-world wall-clock times baked into the stream graphics — not video positions)
_CLOCK_RE = re.compile(r"^\d{2}:\d{2}:\d{2}$")


@dataclass
class Window:
    window_start: int  # seconds
    window_end: int
    label: str  # "00:00:00 - 00:10:00"
    combined_text: str


def aggregate(
    segments: list[Segment],
    ocr_results: dict[str, list[str]],
    window_size: int = 600,   # 10 minutes
    overlap: int = 60,         # 1 minute overlap between windows
) -> list[Window]:
    """
    Merge transcript segments and OCR results into time windows for LLM processing.
    Windows overlap by `overlap` seconds so lots that span a boundary aren't split.
    """
    if window_size <= 0 or overlap < 0 or overlap >= window_size:
        raise ValueError(
            "window_size must be positive and greater than a non-negative overlap"
        )

    if not segments and not ocr_results:
        return []

    # Transcriber backends normally return chronological segments, but cached
    # or externally supplied transcripts are not guaranteed to preserve that
    # order. The early-break optimization below is correct only after sorting.
    segments = sorted(segments, key=lambda segment: (segment.start, segment.end))

    transcript_duration = max((seg.end for seg in segments), default=0)
    ocr_duration = max((_parse_ts(ts) for ts in ocr_results), default=0)
    total_duration = int(max(transcript_duration, ocr_duration)) + 1
    windows: list[Window] = []

    start = 0
    while start < total_duration:
        end = start + window_size
        label = f"{_fmt(start)} - {_fmt(end)}"

        # Collect transcript segments that overlap this window
        trans_lines = []
        for seg in segments:
            if seg.end < start:
                continue
            if seg.start > end:
                break
            ts = _fmt(int(seg.start))
            trans_lines.append(f"[{ts}] ÁUDIO: {seg.text}")

        # Collect OCR results for screenshots within this window
        ocr_lines = []
        for ts_str, texts in ocr_results.items():
            secs = _parse_ts(ts_str)
            if start <= secs <= end and texts:
                # Filter broadcast clock overlays (HH:MM:SS wall-clock time shown
                # in stream graphics) — these are NOT video positions and confuse the LLM
                filtered = [t for t in texts if not _CLOCK_RE.match(t)]
                if filtered:
                    joined = " | ".join(filtered)
                    ocr_lines.append(f"[{ts_str}] TELA: {joined}")

        # Interleave by timestamp
        all_lines = trans_lines + ocr_lines
        all_lines.sort(key=lambda ln: ln[1:9])  # sort by HH:MM:SS prefix

        combined_text = "\n".join(all_lines) if all_lines else "(sem conteúdo)"

        windows.append(Window(
            window_start=start,
            window_end=end,
            label=label,
            combined_text=combined_text,
        ))

        start += window_size - overlap

    return windows


def _fmt(seconds: int) -> str:
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def _parse_ts(ts: str) -> int:
    parts = ts.split(":")
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])

"""
Transcript quality gate for Whisper output.

Whisper on long auction audio with music/silence stretches produces two
well-known failure modes that would otherwise be fed straight to the LLM
extractor as if they were real speech — a hallucinated sentence can invent
a price:

  1. Caption-credit boilerplate — Whisper sometimes returns only a generic
     subtitle-community credit instead of actual transcription.
  2. Repetition loops — the same short phrase repeated for dozens of
     consecutive segments, a classic Whisper hallucination on silence.

`check_transcript()` detects both, plus overall sparse coverage (mostly
silence/noise), and returns a cleaned segment list with a status the
caller can act on (e.g. lean harder on OCR evidence for that window).
"""

import re
import unicodedata
from dataclasses import dataclass

from pipeline.transcriber import Segment

# A hallucinated loop needs at least this many consecutive identical
# segments before we treat it as noise. Below this, an auctioneer
# genuinely repeating a call ("dando... dando...") is plausible.
_REPETITION_RUN_THRESHOLD = 6

# Below this fraction of audio_duration actually covered by segment spans,
# the transcript is mostly silence/noise and extraction should lean on OCR.
_MIN_COVERAGE_RATIO = 0.15

# A real caption credit is ~6-10 words; this gives slack for punctuation
# variants without matching a long real sentence that happens to mention it.
_CAPTION_CREDIT_MAX_WORDS = 15

_CAPTION_CREDIT_PATTERNS = (
    "legendas pela comunidade amara org",
    "transcricao e legendas pela comunidade amara org",
    "transcricao e legenda pela comunidade amara org",
    "legenda pela comunidade amara org",
    "subtitles by the amara org community",
    "captions by the amara org community",
)


@dataclass(frozen=True)
class TranscriptQuality:
    segments: list[Segment]   # cleaned segments (may be shorter than input)
    status: str               # "ok" | "empty" | "degraded"
    warning: str = ""         # human-readable, empty when status == "ok"
    dropped_segments: int = 0


def check_transcript(segments: list[Segment], *, audio_duration: float | None = None) -> TranscriptQuality:
    """Run the quality gate over a raw transcript and return a cleaned view.

    Status precedence: "empty" > "degraded" > "ok". Any rule that removes
    segments but leaves a usable transcript yields "degraded" with a
    warning naming what was removed and how many.
    """
    non_blank = [s for s in segments if s.text.strip()]
    if not non_blank:
        return TranscriptQuality(segments=[], status="empty")

    kept, credit_dropped = _drop_caption_credits(non_blank)
    if not kept:
        return TranscriptQuality(segments=[], status="empty")

    kept, loop_dropped = _collapse_repetition_loops(kept)

    warnings = []
    if credit_dropped:
        warnings.append(
            f"Dropped {credit_dropped} caption-credit segment(s) with no real speech."
        )
    if loop_dropped:
        warnings.append(
            f"Collapsed a repeated-phrase hallucination, removing {loop_dropped} "
            f"duplicate segment(s)."
        )

    total_dropped = credit_dropped + loop_dropped
    status = "degraded" if total_dropped else "ok"
    warning = " ".join(warnings)

    if audio_duration is not None and audio_duration > 0:
        spoken = sum(s.end - s.start for s in kept)
        coverage = spoken / audio_duration
        if coverage < _MIN_COVERAGE_RATIO:
            status = "degraded"
            coverage_warning = (
                f"Transcript covers only {coverage * 100:.1f}% of the audio duration; "
                f"extraction should lean on OCR evidence instead of transcript text."
            )
            warning = f"{warning} {coverage_warning}".strip()

    return TranscriptQuality(
        segments=kept,
        status=status,
        warning=warning,
        dropped_segments=total_dropped,
    )


def _drop_caption_credits(segments: list[Segment]) -> tuple[list[Segment], int]:
    kept = [s for s in segments if not _is_caption_credit(s.text)]
    return kept, len(segments) - len(kept)


def _is_caption_credit(text: str) -> bool:
    normalized = _normalize_text(text)
    if not normalized:
        return False
    if len(normalized.split()) > _CAPTION_CREDIT_MAX_WORDS:
        return False
    return any(pattern in normalized for pattern in _CAPTION_CREDIT_PATTERNS)


def _collapse_repetition_loops(segments: list[Segment]) -> tuple[list[Segment], int]:
    """Collapse runs of >= _REPETITION_RUN_THRESHOLD identical segments to their first."""
    result: list[Segment] = []
    dropped = 0
    i, n = 0, len(segments)
    while i < n:
        normalized = _normalize_text(segments[i].text)
        j = i + 1
        while j < n and _normalize_text(segments[j].text) == normalized:
            j += 1
        run_length = j - i
        if run_length >= _REPETITION_RUN_THRESHOLD:
            result.append(segments[i])
            dropped += run_length - 1
        else:
            result.extend(segments[i:j])
        i = j
    return result, dropped


def _normalize_text(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    ascii_text = "".join(c for c in decomposed if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", " ", ascii_text.lower()).strip()

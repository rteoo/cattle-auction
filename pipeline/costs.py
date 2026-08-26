"""
Cost estimation for LLM extraction calls and Groq audio transcription.

`LLMClient` in pipeline/extractor.py already counts input/output tokens and
call counts, but nothing converts them to money — batch runs over dozens of
videos have invisible spend otherwise. `estimate_cost()` fills that gap.

Prices are published per-1M-token USD rates. They are estimates and will
drift; verify against each provider's pricing page before relying on them
for budgeting.
"""

# Keyed by the model id passed to LLMClient (see _DEFAULT_MODELS in
# pipeline/extractor.py for the two supported extraction models).
PRICE_TABLE: dict[str, dict[str, float]] = {
    "google/gemini-2.5-flash-lite-preview-09-2025": {"input": 0.10, "output": 0.40},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
}

# Used when `model` isn't in PRICE_TABLE (e.g. a new/renamed model). Set to
# the more expensive of the two known models so an unrecognized model biases
# toward overestimating cost rather than silently underestimating it.
_FALLBACK_PRICES: dict[str, float] = {"input": 0.40, "output": 1.60}

# Groq Whisper Large v3 Turbo pricing (see pipeline/transcriber.py docstring).
GROQ_WHISPER_USD_PER_HOUR: float = 0.04


def estimate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    *,
    transcriber: str = "groq",
    audio_seconds: float = 0.0,
) -> dict[str, float]:
    """Estimate USD cost for one video's LLM extraction + transcription.

    Never raises over a pricing lookup — an unrecognized model id falls back
    to `_FALLBACK_PRICES` rather than blowing up a finished pipeline run.
    """
    prices = PRICE_TABLE.get(model, _FALLBACK_PRICES)
    llm_usd = (
        (input_tokens / 1_000_000) * prices["input"]
        + (output_tokens / 1_000_000) * prices["output"]
    )

    if transcriber == "groq":
        transcription_usd = (audio_seconds / 3600.0) * GROQ_WHISPER_USD_PER_HOUR
    else:
        # mlx and cpp run locally and cost nothing to run; any other/unknown
        # transcriber name is treated the same way (no billable API involved).
        transcription_usd = 0.0

    total_usd = llm_usd + transcription_usd

    return {
        "llm_usd": round(llm_usd, 6),
        "transcription_usd": round(transcription_usd, 6),
        "total_usd": round(total_usd, 6),
    }


def format_cost(usd: float) -> str:
    """Format a USD amount for display, e.g. "$0.0512".

    A single video typically costs cents, so 4 decimal places are used.
    Anything positive but too small to show at that precision renders as
    "<$0.0001" rather than the misleading "$0.0000".
    """
    if usd == 0:
        return "$0.0000"
    if usd < 0.0001:
        return "<$0.0001"
    return f"${usd:.4f}"

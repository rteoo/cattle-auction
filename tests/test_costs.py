"""Tests for LLM + transcription cost estimation."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.costs import estimate_cost, format_cost, GROQ_WHISPER_USD_PER_HOUR, _FALLBACK_PRICES


class TestEstimateCostKnownModels:
    def test_gemini_flash_lite_arithmetic(self):
        result = estimate_cost(
            "google/gemini-2.5-flash-lite-preview-09-2025",
            input_tokens=500_000,
            output_tokens=200_000,
            transcriber="mlx",
        )
        # 500_000/1e6 * 0.10 + 200_000/1e6 * 0.40 = 0.05 + 0.08
        assert result["llm_usd"] == 0.13
        assert result["transcription_usd"] == 0.0
        assert result["total_usd"] == 0.13

    def test_gpt41_mini_arithmetic(self):
        result = estimate_cost(
            "gpt-4.1-mini",
            input_tokens=1_000_000,
            output_tokens=500_000,
            transcriber="mlx",
        )
        # 1_000_000/1e6 * 0.40 + 500_000/1e6 * 1.60 = 0.40 + 0.80
        assert result["llm_usd"] == 1.20
        assert result["total_usd"] == 1.20


class TestUnknownModel:
    def test_unknown_model_uses_fallback_without_raising(self):
        result = estimate_cost(
            "some/unreleased-model-v9",
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            transcriber="mlx",
        )
        expected_llm = (
            (1_000_000 / 1_000_000) * _FALLBACK_PRICES["input"]
            + (1_000_000 / 1_000_000) * _FALLBACK_PRICES["output"]
        )
        assert result["llm_usd"] == round(expected_llm, 6)


class TestTranscriptionCost:
    def test_groq_bills_audio_seconds(self):
        result = estimate_cost(
            "gpt-4.1-mini", input_tokens=0, output_tokens=0,
            transcriber="groq", audio_seconds=3600.0,
        )
        assert result["transcription_usd"] == round(GROQ_WHISPER_USD_PER_HOUR, 6)

    def test_groq_half_hour(self):
        result = estimate_cost(
            "gpt-4.1-mini", input_tokens=0, output_tokens=0,
            transcriber="groq", audio_seconds=1800.0,
        )
        assert result["transcription_usd"] == round(GROQ_WHISPER_USD_PER_HOUR / 2, 6)

    def test_mlx_is_free(self):
        result = estimate_cost(
            "gpt-4.1-mini", input_tokens=0, output_tokens=0,
            transcriber="mlx", audio_seconds=99999.0,
        )
        assert result["transcription_usd"] == 0.0

    def test_cpp_is_free(self):
        result = estimate_cost(
            "gpt-4.1-mini", input_tokens=0, output_tokens=0,
            transcriber="cpp", audio_seconds=99999.0,
        )
        assert result["transcription_usd"] == 0.0


class TestZeroTokens:
    def test_zero_tokens_and_zero_audio_is_zero(self):
        result = estimate_cost(
            "gpt-4.1-mini", input_tokens=0, output_tokens=0, transcriber="mlx",
        )
        assert result == {"llm_usd": 0.0, "transcription_usd": 0.0, "total_usd": 0.0}


class TestFormatCost:
    def test_typical_amount(self):
        assert format_cost(0.0512) == "$0.0512"

    def test_exact_zero(self):
        assert format_cost(0.0) == "$0.0000"

    def test_tiny_positive_amount_below_display_precision(self):
        assert format_cost(0.00003) == "<$0.0001"

    def test_boundary_at_display_precision(self):
        assert format_cost(0.0001) == "$0.0001"

"""Tests for Lot model validation — especially price coercion."""
import pytest
from pydantic import ValidationError

from models.lot import Lot, AuctionResult


# ── coerce_price ────────────────────────────────────────────────────────────

class TestCoercePrice:
    def _lot(self, unit_price=None, total_price=None):
        return Lot(
            lot_number=1, sex="macho", category="bezerro",
            num_animals=5, breed="Nelore",
            unit_price=unit_price, total_price=total_price,
        )

    # Plain numbers
    def test_float_passthrough(self):
        assert self._lot(unit_price=3200.0).unit_price == 3200.0

    def test_int_coerced_to_float(self):
        assert self._lot(unit_price=3200).unit_price == 3200.0

    def test_none_stays_none(self):
        assert self._lot(unit_price=None).unit_price is None

    # Brazilian thousand-separator format (the 3.100 → 3.10 bug)
    def test_br_thousand_dot_no_decimal(self):
        """3.100 means three thousand one hundred, NOT 3.10."""
        assert self._lot(unit_price="3.100").unit_price == 3100.0

    def test_br_thousand_dot_with_comma_decimal(self):
        assert self._lot(unit_price="3.100,00").unit_price == 3100.0

    def test_br_large_with_cents(self):
        assert self._lot(unit_price="1.800,50").unit_price == 1800.50

    def test_br_two_thousand_separators(self):
        assert self._lot(unit_price="1.234.567,89").unit_price == 1234567.89

    # R$ prefix stripping
    def test_rs_prefix_stripped(self):
        assert self._lot(unit_price="R$ 2.500,00").unit_price == 2500.0

    def test_rs_prefix_no_space(self):
        assert self._lot(unit_price="R$2.500").unit_price == 2500.0

    # BR float mis-parse guard: LLM outputs "5.160" in JSON → Python float 5.16
    def test_float_under_100_multiplied_by_1000(self):
        """5.16 (from JSON "5.160") must be corrected to 5160."""
        assert self._lot(unit_price=5.16).unit_price == pytest.approx(5160.0)

    def test_float_under_100_no_false_positive_above_100(self):
        """Values ≥ 100 are not touched by the guard."""
        assert self._lot(unit_price=100.0).unit_price == pytest.approx(100.0)
        assert self._lot(unit_price=2750.0).unit_price == pytest.approx(2750.0)

    def test_zero_not_multiplied(self):
        """Zero is treated as null (no bid), not multiplied."""
        assert self._lot(unit_price=0).unit_price == pytest.approx(0.0)

    # Edge cases
    def test_empty_string_returns_none(self):
        assert self._lot(unit_price="").unit_price is None

    def test_whitespace_only_returns_none(self):
        assert self._lot(unit_price="   ").unit_price is None

    def test_total_price_also_coerced(self):
        lot = self._lot(total_price="15.000,00")
        assert lot.total_price == 15000.0

    def test_both_prices_coerced(self):
        lot = self._lot(unit_price="3.100,00", total_price="15.500,00")
        assert lot.unit_price == 3100.0
        assert lot.total_price == 15500.0


# ── normalize_category ──────────────────────────────────────────────────────

class TestNormalizeCategory:
    def _lot(self, category):
        return Lot(lot_number=1, sex="macho", category=category, num_animals=5, breed="Nelore")

    def test_plural_garrotes_normalized(self):
        assert self._lot("garrotes").category == "garrote"

    def test_plural_novilhas_normalized(self):
        assert self._lot("novilhas").category == "novilha"

    def test_plural_bezerros_normalized(self):
        assert self._lot("bezerros").category == "bezerro"

    def test_singular_unchanged(self):
        assert self._lot("garrote").category == "garrote"

    def test_unknown_category_unchanged(self):
        assert self._lot("tourinho").category == "tourinho"

    def test_whitespace_stripped(self):
        assert self._lot("  garrotes  ").category == "garrote"

    def test_case_insensitive(self):
        assert self._lot("Garrotes").category == "garrote"


# ── sold field ───────────────────────────────────────────────────────────────

class TestSoldField:
    def _lot(self, sold=None):
        return Lot(
            lot_number=1, sex="macho", category="bezerro",
            num_animals=5, breed="Nelore", sold=sold,
        )

    def test_sold_true(self):
        assert self._lot(sold=True).sold is True

    def test_sold_false(self):
        assert self._lot(sold=False).sold is False

    def test_sold_none_default(self):
        assert self._lot().sold is None

    def test_sold_none_explicit(self):
        assert self._lot(sold=None).sold is None


# ── required fields ──────────────────────────────────────────────────────────

class TestRequiredFields:
    def test_missing_lot_number_raises(self):
        with pytest.raises(ValidationError):
            Lot(sex="macho", category="bezerro", num_animals=5, breed="Nelore")

    def test_missing_sex_raises(self):
        with pytest.raises(ValidationError):
            Lot(lot_number=1, category="bezerro", num_animals=5, breed="Nelore")

    def test_missing_num_animals_raises(self):
        with pytest.raises(ValidationError):
            Lot(lot_number=1, sex="macho", category="bezerro", breed="Nelore")

    def test_optional_fields_default_to_none(self):
        lot = Lot(lot_number=1, sex="macho", category="bezerro", num_animals=5, breed="Nelore")
        assert lot.age_months is None
        assert lot.unit_price is None
        assert lot.total_price is None
        assert lot.sold is None
        assert lot.timestamp_start is None
        assert lot.notes is None


# ── AuctionResult ─────────────────────────────────────────────────────────────

class TestAuctionResult:
    def test_metadata_fields_optional(self):
        result = AuctionResult(
            video_url="https://youtube.com/watch?v=abc",
            video_id="abc",
            total_lots=0,
            lots=[],
        )
        assert result.date is None
        assert result.city is None
        assert result.auctioneer is None
        assert result.farm is None
        assert result.auction_type is None

    def test_metadata_fields_stored(self):
        result = AuctionResult(
            video_url="https://youtube.com/watch?v=abc",
            video_id="abc",
            date="28/03/2026",
            city="Araguaína",
            auctioneer="Leilões Abreu",
            farm="Fazenda Santa Clara",
            auction_type="corte",
            total_lots=1,
            lots=[],
        )
        assert result.date == "28/03/2026"
        assert result.city == "Araguaína"
        assert result.auctioneer == "Leilões Abreu"

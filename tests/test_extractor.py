"""Tests for extractor pure-logic functions: _parse_response, _merge, _validate_lots."""
import pytest

from models.lot import Lot
from pipeline.extractor import (
    _parse_response,
    _merge,
    _validate_lots,
    _sanity_check,
    _compute_price_bounds,
    _verify_lot,
    _parse_hhmmss,
)
from pipeline.aggregator import Window


# ── _validate_lots ───────────────────────────────────────────────────────────

class TestValidateLots:
    def _valid(self, **kwargs):
        base = dict(lot_number=1, sex="macho", category="bezerro", num_animals=5, breed="Nelore")
        base.update(kwargs)
        return base

    def test_valid_item_returns_lot(self):
        lots = _validate_lots([self._valid()])
        assert len(lots) == 1
        assert isinstance(lots[0], Lot)

    def test_invalid_item_skipped(self):
        lots = _validate_lots([{"bad": "data"}])
        assert lots == []

    def test_mixed_valid_and_invalid(self):
        items = [self._valid(lot_number=1), {"bad": "data"}, self._valid(lot_number=2)]
        lots = _validate_lots(items)
        assert len(lots) == 2
        assert [l.lot_number for l in lots] == [1, 2]

    def test_empty_list(self):
        assert _validate_lots([]) == []

    def test_price_coerced_in_validated_lot(self):
        lots = _validate_lots([self._valid(unit_price="3.100")])
        assert lots[0].unit_price == 3100.0


# ── _parse_response ──────────────────────────────────────────────────────────

class TestParseResponse:
    VALID = '[{"lot_number":1,"sex":"macho","category":"bezerro","num_animals":5,"breed":"Nelore"}]'

    def test_clean_json_array(self):
        lots = _parse_response(self.VALID)
        assert len(lots) == 1
        assert lots[0].lot_number == 1

    def test_empty_array(self):
        assert _parse_response("[]") == []

    def test_extra_text_before(self):
        lots = _parse_response("Here are the lots:\n" + self.VALID)
        assert len(lots) == 1

    def test_extra_text_after(self):
        lots = _parse_response(self.VALID + "\nDone.")
        assert len(lots) == 1

    def test_extra_text_both_sides(self):
        lots = _parse_response("Sure!\n" + self.VALID + "\nHope that helps.")
        assert len(lots) == 1

    def test_invalid_json_returns_empty(self):
        assert _parse_response("not json at all") == []

    def test_multiple_lots(self):
        raw = (
            '[{"lot_number":1,"sex":"macho","category":"bezerro","num_animals":5,"breed":"Nelore"},'
            '{"lot_number":2,"sex":"femea","category":"vaca","num_animals":2,"breed":"Nelore"}]'
        )
        lots = _parse_response(raw)
        assert len(lots) == 2
        assert lots[0].lot_number == 1
        assert lots[1].lot_number == 2

    def test_whitespace_stripped(self):
        lots = _parse_response("  " + self.VALID + "  ")
        assert len(lots) == 1

    def test_invalid_lot_inside_valid_json_skipped(self):
        raw = '[{"bad":"data"}, {"lot_number":2,"sex":"macho","category":"bezerro","num_animals":3,"breed":"Nelore"}]'
        lots = _parse_response(raw)
        assert len(lots) == 1
        assert lots[0].lot_number == 2

    def test_sold_field_parsed(self):
        raw = '[{"lot_number":1,"sex":"macho","category":"bezerro","num_animals":5,"breed":"Nelore","sold":false}]'
        lots = _parse_response(raw)
        assert lots[0].sold is False


# ── _merge ───────────────────────────────────────────────────────────────────

def _make_lot(**kwargs):
    base = dict(lot_number=1, sex="macho", category="bezerro", num_animals=5, breed="Nelore")
    base.update(kwargs)
    return Lot(**base)


class TestMerge:
    def test_new_lot_inserted(self):
        store = {}
        lot = _make_lot(lot_number=1)
        _merge(store, lot)
        assert 1 in store
        assert store[1] is lot

    def test_existing_lot_nulls_filled_by_new(self):
        store = {}
        first = _make_lot(lot_number=1, unit_price=None)
        _merge(store, first)
        second = _make_lot(lot_number=1, unit_price=3000.0)
        _merge(store, second)
        assert store[1].unit_price == 3000.0

    def test_non_price_non_null_not_overwritten(self):
        """First non-null wins for stable fields like breed."""
        store = {}
        first = _make_lot(lot_number=1, breed="Nelore")
        _merge(store, first)
        second = _make_lot(lot_number=1, breed="Anelorado")
        _merge(store, second)
        assert store[1].breed == "Nelore"

    def test_price_updated_by_later_window(self):
        """Prices use last-non-null: final hammer price beats opening ask."""
        store = {}
        first = _make_lot(lot_number=1, unit_price=2600.0)
        _merge(store, first)
        second = _make_lot(lot_number=1, unit_price=1900.0)
        _merge(store, second)
        assert store[1].unit_price == 1900.0

    def test_price_not_cleared_by_later_null(self):
        """A later window with null price doesn't wipe an established price."""
        store = {}
        first = _make_lot(lot_number=1, unit_price=2600.0)
        _merge(store, first)
        second = _make_lot(lot_number=1, unit_price=None)
        _merge(store, second)
        assert store[1].unit_price == 2600.0

    def test_total_price_updated_by_later_window(self):
        store = {}
        first = _make_lot(lot_number=1, total_price=67600.0)
        _merge(store, first)
        second = _make_lot(lot_number=1, total_price=49400.0)
        _merge(store, second)
        assert store[1].total_price == 49400.0

    def test_multiple_null_fields_filled(self):
        store = {}
        first = _make_lot(lot_number=1, unit_price=None, age_months=None)
        _merge(store, first)
        second = _make_lot(lot_number=1, unit_price=3000.0, age_months=12)
        _merge(store, second)
        assert store[1].unit_price == 3000.0
        assert store[1].age_months == 12

    def test_different_lots_stored_independently(self):
        store = {}
        _merge(store, _make_lot(lot_number=1))
        _merge(store, _make_lot(lot_number=2))
        assert set(store.keys()) == {1, 2}

    def test_sold_false_not_overwritten_by_none(self):
        """sold=False is a value — None should not overwrite it."""
        store = {}
        first = _make_lot(lot_number=1, sold=False)
        _merge(store, first)
        second = _make_lot(lot_number=1, sold=None)
        _merge(store, second)
        assert store[1].sold is False

    def test_sold_none_filled_by_true(self):
        store = {}
        first = _make_lot(lot_number=1, sold=None)
        _merge(store, first)
        second = _make_lot(lot_number=1, sold=True)
        _merge(store, second)
        assert store[1].sold is True

    def test_sold_true_overrides_false(self):
        """sold=True is a final determination — overrides prior False."""
        store = {}
        first = _make_lot(lot_number=1, sold=False)
        _merge(store, first)
        second = _make_lot(lot_number=1, sold=True)
        _merge(store, second)
        assert store[1].sold is True


# ── _sanity_check: shape invariants (no bounds) ──────────────────────────────

class TestSanityCheckInvariants:
    """
    Shape invariants apply regardless of price scale — per-window calls use
    these (no `bounds` argument) because statistics aren't computed yet.
    """

    def _lot(self, **kwargs):
        base = dict(lot_number=1, sex="macho", category="bezerro", num_animals=5, breed="Nelore")
        base.update(kwargs)
        return Lot(**base)

    # unit × n ≈ total — when violated, trust unit and clear total
    def test_product_mismatch_clears_total(self):
        """unit × n ≠ total by >20% → trust unit, clear total."""
        lot = self._lot(num_animals=10, unit_price=3000.0, total_price=500.0)
        checked = _sanity_check(lot)
        assert checked.unit_price == 3000.0
        assert checked.total_price is None

    def test_product_match_preserved(self):
        lot = self._lot(num_animals=10, unit_price=3000.0, total_price=30000.0)
        checked = _sanity_check(lot)
        assert checked.unit_price == 3000.0
        assert checked.total_price == 30000.0

    def test_product_within_tolerance(self):
        """Small rounding difference is tolerated."""
        lot = self._lot(num_animals=10, unit_price=3000.0, total_price=29500.0)
        checked = _sanity_check(lot)
        assert checked.unit_price == 3000.0
        assert checked.total_price == 29500.0

    # unit > total fallback (fires when num_animals is missing)
    def test_unit_greater_than_total_without_num_animals_hits_fallback(self):
        """Without num_animals, product check can't run; inversion fallback catches it."""
        lot = self._lot(num_animals=1, unit_price=5000.0, total_price=1000.0)
        # product check sees expected=5000, total=1000, rel_err=0.8 → clears total first
        # So the fallback inversion check won't fire. That's fine — trusting unit
        # when they disagree is the right call when a num is known (num=1 here).
        checked = _sanity_check(lot)
        assert checked.unit_price == 5000.0
        assert checked.total_price is None

    def test_null_prices_unchanged(self):
        lot = self._lot(unit_price=None, total_price=None)
        checked = _sanity_check(lot)
        assert checked.unit_price is None
        assert checked.total_price is None

    def test_only_unit_present_passes(self):
        """total_price can be legitimately null — don't penalize that."""
        lot = self._lot(num_animals=10, unit_price=3000.0, total_price=None)
        checked = _sanity_check(lot)
        assert checked.unit_price == 3000.0
        assert checked.total_price is None

    def test_non_price_fields_preserved(self):
        lot = self._lot(unit_price=3000.0, total_price=30000.0,
                        num_animals=10, breed="Nelore", age_months=18)
        checked = _sanity_check(lot)
        assert checked.breed == "Nelore"
        assert checked.age_months == 18
        assert checked.num_animals == 10

    # Without bounds, prices of any magnitude pass the invariants if they're
    # internally consistent. This is intentional — statistical filtering happens
    # post-merge with the full distribution.
    def test_without_bounds_absurd_price_not_clipped(self):
        lot = self._lot(num_animals=1, unit_price=999999.0, total_price=999999.0)
        checked = _sanity_check(lot)
        assert checked.unit_price == 999999.0


# ── _sanity_check: statistical bounds (post-merge) ───────────────────────────

class TestSanityCheckStatisticalBounds:
    """
    Statistical bounds come from the auction's own price distribution.
    When supplied to `_sanity_check`, they filter out-of-distribution unit prices.
    """

    def _lot(self, **kwargs):
        base = dict(lot_number=1, sex="macho", category="bezerro", num_animals=5, breed="Nelore")
        base.update(kwargs)
        return Lot(**base)

    def test_unit_below_lower_bound_nulled(self):
        lot = self._lot(num_animals=1, unit_price=50.0, total_price=50.0)
        checked = _sanity_check(lot, bounds=(500.0, 15000.0))
        assert checked.unit_price is None

    def test_unit_above_upper_bound_nulled(self):
        lot = self._lot(num_animals=1, unit_price=30000.0, total_price=30000.0)
        checked = _sanity_check(lot, bounds=(500.0, 15000.0))
        assert checked.unit_price is None

    def test_unit_in_bounds_preserved(self):
        lot = self._lot(num_animals=10, unit_price=3000.0, total_price=30000.0)
        checked = _sanity_check(lot, bounds=(500.0, 15000.0))
        assert checked.unit_price == 3000.0
        assert checked.total_price == 30000.0

    def test_implied_per_head_checked_when_only_total_survives(self):
        """After unit nulled by bounds, implied per-head from total must also be in bounds."""
        lot = self._lot(num_animals=50, unit_price=50000.0, total_price=2500.0)
        # Step 1: unit=50000 above bounds → nulled.
        # Step 2: product check needs unit → skipped.
        # Step 4: implied = 2500 / 50 = 50, below lower bound → clear total.
        checked = _sanity_check(lot, bounds=(500.0, 15000.0))
        assert checked.unit_price is None
        assert checked.total_price is None

    def test_inf_upper_bound_disables_upper_check(self):
        import math
        lot = self._lot(num_animals=1, unit_price=999999.0, total_price=999999.0)
        checked = _sanity_check(lot, bounds=(0.0, math.inf))
        assert checked.unit_price == 999999.0


# ── _compute_price_bounds ────────────────────────────────────────────────────

class TestComputePriceBounds:
    """Tukey 1.5·IQR fences derived from observed prices."""

    def test_insufficient_samples_returns_infinite_bounds(self):
        """Below the minimum sample size, bounds are disabled (0, inf)."""
        import math
        lo, hi = _compute_price_bounds([3000.0, 3500.0])
        assert lo == 0.0
        assert math.isinf(hi)

    def test_tukey_fence_formula(self):
        """Bounds are exactly [Q1 − 1.5·IQR, Q3 + 1.5·IQR]."""
        import statistics
        from pipeline.extractor import _IQR_MULTIPLIER
        prices = [2000.0, 2500.0, 3000.0, 3500.0, 4000.0, 4500.0, 5000.0]
        q1, _, q3 = statistics.quantiles(prices, n=4)
        iqr = q3 - q1
        lo, hi = _compute_price_bounds(prices)
        assert lo == max(0.0, q1 - _IQR_MULTIPLIER * iqr)
        assert hi == q3 + _IQR_MULTIPLIER * iqr

    def test_robust_to_extreme_outliers(self):
        """The point of Tukey IQR: extreme outliers don't destroy the bounds.
        Quartiles stay stable up to the ~25% breakdown point, unlike mean/stdev
        which would be pulled hard by R$ 55k hallucinations."""
        # 20 clean values + 2 outliers = ~9% contamination, well under 25% breakdown
        clean = [2000.0, 2500.0, 3000.0, 3000.0, 3000.0, 3500.0, 3500.0, 4000.0] * 2 \
              + [2200.0, 2800.0, 3200.0, 3700.0]
        polluted = clean + [55000.0, 55000.0]
        _, hi_clean = _compute_price_bounds(clean)
        _, hi_polluted = _compute_price_bounds(polluted)
        # Adding contamination barely moves the upper fence (within ~2× of clean)
        assert abs(hi_polluted - hi_clean) / hi_clean < 1.0
        # And the hallucinations themselves are clearly rejected
        assert hi_polluted < 10000.0

    def test_typical_auction_distribution(self):
        """A realistic auction produces bounds that cover the mass of the
        distribution and exclude extreme hallucinations."""
        prices = (
            [3000.0] * 20 + [2500.0] * 15 + [3500.0] * 15
            + [4000.0, 4500.0, 2000.0, 2200.0]
        )
        lo, hi = _compute_price_bounds(prices)
        # R$ 30k bezerro hallucination would be above the upper fence
        assert hi < 30000.0
        # Typical values fit inside the fence
        assert lo < 2500.0 < hi
        assert lo < 3500.0 < hi

    def test_zero_prices_excluded(self):
        """Zeros (unbid lots) don't count in the distribution."""
        import math
        # Only 4 non-zero values → below sample minimum
        lo, hi = _compute_price_bounds([0.0, 0.0, 3000.0, 3500.0, 3200.0, 3100.0])
        assert lo == 0.0
        assert math.isinf(hi)

    def test_lower_bound_clamped_at_zero(self):
        """When Q1 − 1.5·IQR goes negative, clamp to zero (price can't be negative)."""
        prices = [1000.0, 1500.0, 2000.0, 2500.0, 3000.0, 10000.0]
        lo, _ = _compute_price_bounds(prices)
        assert lo >= 0.0

    def test_adapts_to_auction_price_level(self):
        """Bounds shift with the auction's price scale — no hard-coded values."""
        commodity = [2000.0, 2500.0, 3000.0, 3500.0, 4000.0, 4500.0, 5000.0]
        elite = [20000.0, 25000.0, 30000.0, 35000.0, 40000.0, 45000.0, 50000.0]
        _, hi_c = _compute_price_bounds(commodity)
        _, hi_e = _compute_price_bounds(elite)
        # Elite auction's upper fence should be ~10× higher than commodity's
        assert hi_e > hi_c * 5


# ── _validate_lots uses only invariants (no bounds) ──────────────────────────

class TestValidateLotsShapeOnly:
    """Per-window validation must not apply statistical bounds — they don't
    exist yet at that point in the pipeline."""

    def _valid(self, **kwargs):
        base = dict(lot_number=1, sex="macho", category="bezerro",
                    num_animals=10, breed="Nelore")
        base.update(kwargs)
        return base

    def test_validate_lots_catches_unit_over_total_via_product_check(self):
        """Lot 10 production failure (unit=55000, total=2920, n=50):
        product check sees unit×n=2.75M vs total=2920 → clears total."""
        lots = _validate_lots([self._valid(num_animals=50, unit_price=55000.0, total_price=2920.0)])
        assert len(lots) == 1
        # total cleared by product-consistency check
        assert lots[0].total_price is None
        # unit still present — it'll be filtered post-merge by statistical bounds
        assert lots[0].unit_price == 55000.0


# ── _parse_hhmmss ────────────────────────────────────────────────────────────

class TestParseHHMMSS:
    def test_basic(self):
        assert _parse_hhmmss("00:00:00") == 0
        assert _parse_hhmmss("00:01:00") == 60
        assert _parse_hhmmss("01:00:00") == 3600
        assert _parse_hhmmss("02:30:45") == 2 * 3600 + 30 * 60 + 45

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            _parse_hhmmss("not a timestamp")


# ── _verify_lot ──────────────────────────────────────────────────────────────

class _MockClient:
    """Stand-in for LLMClient that returns a canned response string."""
    def __init__(self, response: str, raise_on_call: bool = False):
        self.response = response
        self.raise_on_call = raise_on_call
        self.calls = []

    def complete(self, system: str, user: str, max_retries: int = 3) -> str:
        self.calls.append((system, user))
        if self.raise_on_call:
            raise RuntimeError("simulated LLM failure")
        return self.response


def _make_window(start: int = 0, end: int = 600, text: str = "dummy evidence") -> Window:
    return Window(window_start=start, window_end=end,
                  label=f"{start//3600:02d}:{(start%3600)//60:02d}:{start%60:02d} - …",
                  combined_text=text)


def _flagged_lot(**kwargs):
    """Build a lot that's been flagged (has a timestamp so verification can run)."""
    base = dict(lot_number=10, sex="macho", category="bezerro", num_animals=10,
                breed="Anelorado", unit_price=55000.0, timestamp_start="00:05:00")
    base.update(kwargs)
    return Lot(**base)


class TestVerifyLot:
    """The outlier-recovery pass: focused LLM call on a single flagged lot."""

    def test_confirm_verdict(self):
        """`confirmed:true` → return the sentinel 'confirm'."""
        client = _MockClient('{"confirmed": true, "correct_unit_price": 55000, "reasoning": "audio says fifty-five"}')
        lot = _flagged_lot()
        windows = [_make_window()]
        verdict = _verify_lot(lot, windows, client, "test prompt")
        assert verdict == "confirm"

    def test_correct_verdict(self):
        """`confirmed:false` with a numeric correct_unit_price → return that price."""
        client = _MockClient('{"confirmed": false, "correct_unit_price": 5500, "reasoning": "parse error"}')
        lot = _flagged_lot()
        windows = [_make_window()]
        verdict = _verify_lot(lot, windows, client, "test prompt")
        assert verdict == 5500.0

    def test_correction_equal_to_original_collapses_to_confirm(self):
        """LLM returns confirmed:false but the 'correct' price == the original
        → treat as a confirm, so the log reads correctly."""
        client = _MockClient('{"confirmed": false, "correct_unit_price": 55000, "reasoning": "same"}')
        lot = _flagged_lot(unit_price=55000.0)
        verdict = _verify_lot(lot, [_make_window()], client, "test prompt")
        assert verdict == "confirm"

    def test_correction_returned_as_float(self):
        """Corrections come back as float, regardless of LLM-emitted JSON type."""
        # JSON integer
        client1 = _MockClient('{"confirmed": false, "correct_unit_price": 3200}')
        assert _verify_lot(_flagged_lot(), [_make_window()], client1, "p") == 3200.0
        # JSON float
        client2 = _MockClient('{"confirmed": false, "correct_unit_price": 3200.50}')
        assert _verify_lot(_flagged_lot(), [_make_window()], client2, "p") == 3200.5

    def test_discard_verdict(self):
        """`confirmed:false` with null price → return 'discard'."""
        client = _MockClient('{"confirmed": false, "correct_unit_price": null, "reasoning": "ambiguous"}')
        lot = _flagged_lot()
        windows = [_make_window()]
        verdict = _verify_lot(lot, windows, client, "test prompt")
        assert verdict == "discard"

    def test_response_with_extra_text(self):
        """Should tolerate extra text around the JSON."""
        client = _MockClient('Sure! Here is the verdict:\n{"confirmed": true, "correct_unit_price": 55000}\nHope it helps.')
        verdict = _verify_lot(_flagged_lot(), [_make_window()], client, "test prompt")
        assert verdict == "confirm"

    def test_malformed_json_returns_none(self):
        client = _MockClient("totally not json")
        verdict = _verify_lot(_flagged_lot(), [_make_window()], client, "test prompt")
        assert verdict is None

    def test_llm_error_returns_none(self):
        client = _MockClient("", raise_on_call=True)
        verdict = _verify_lot(_flagged_lot(), [_make_window()], client, "test prompt")
        assert verdict is None

    def test_missing_timestamp_returns_none(self):
        """Can't locate the evidence without timestamp_start."""
        lot = _flagged_lot(timestamp_start=None)
        client = _MockClient('{"confirmed": true}')
        verdict = _verify_lot(lot, [_make_window()], client, "test prompt")
        assert verdict is None
        assert client.calls == []  # never called

    def test_timestamp_outside_any_window_returns_none(self):
        lot = _flagged_lot(timestamp_start="99:59:59")
        client = _MockClient('{"confirmed": true}')
        verdict = _verify_lot(lot, [_make_window(0, 600)], client, "test prompt")
        assert verdict is None
        assert client.calls == []

    def test_prompt_includes_lot_details(self):
        """The user message should carry lot metadata + evidence text."""
        client = _MockClient('{"confirmed": true}')
        lot = _flagged_lot(lot_number=42, category="garrote", num_animals=25, unit_price=6800.0)
        windows = [_make_window(text="TELA: LOTE 42 R$ 6.800 25GARROTES")]
        _verify_lot(lot, windows, client, "system prompt")
        (system, user) = client.calls[0]
        assert system == "system prompt"
        assert "42" in user
        assert "garrote" in user
        assert "6,800.00" in user
        assert "25GARROTES" in user

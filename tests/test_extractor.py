"""Tests for extractor pure-logic functions: _parse_response, _merge, _validate_lots."""
import pytest

from models.lot import Lot
from pipeline.extractor import _parse_response, _merge, _validate_lots


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

    def test_existing_non_null_not_overwritten(self):
        """First non-null value wins."""
        store = {}
        first = _make_lot(lot_number=1, unit_price=3000.0)
        _merge(store, first)
        second = _make_lot(lot_number=1, unit_price=5000.0)
        _merge(store, second)
        assert store[1].unit_price == 3000.0

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

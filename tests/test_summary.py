"""Tests for main._calculate_summary statistics."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from models.lot import Lot
from main import _calculate_summary


def lot(lot_number, sex, category, num_animals, unit_price=None, sold=None, breed="Nelore"):
    return Lot(
        lot_number=lot_number, sex=sex, category=category,
        num_animals=num_animals, breed=breed,
        unit_price=unit_price, sold=sold,
    )


class TestCalculateSummary:
    def test_empty_returns_empty_dict(self):
        assert _calculate_summary([]) == {}

    def test_total_lots(self):
        lots = [lot(1, "macho", "bezerro", 5), lot(2, "femea", "vaca", 3)]
        assert _calculate_summary(lots)["total_lots"] == 2

    def test_total_animals(self):
        lots = [lot(1, "macho", "bezerro", 5), lot(2, "femea", "vaca", 3)]
        assert _calculate_summary(lots)["total_animals"] == 8

    def test_sex_animals_counts(self):
        lots = [
            lot(1, "macho", "bezerro", 10),
            lot(2, "macho", "garrote", 5),
            lot(3, "fêmea", "vaca", 3),
            lot(4, "misto", "bezerro", 8),
        ]
        s = _calculate_summary(lots)
        assert s["sex_animals"]["macho"] == 15
        assert s["sex_animals"]["fêmea"] == 3
        assert s["sex_animals"]["misto"] == 8

    def test_sex_with_zero_animals_excluded(self):
        lots = [lot(1, "macho", "bezerro", 5)]
        s = _calculate_summary(lots)
        assert "fêmea" not in s["sex_animals"]
        assert "misto" not in s["sex_animals"]

    def test_category_animals_by_count(self):
        lots = [
            lot(1, "macho", "bezerro", 20),
            lot(2, "macho", "bezerro", 15),
            lot(3, "fêmea", "vaca", 10),
            lot(4, "misto", "garrote", 5),
            lot(5, "misto", "novilha", 3),
            lot(6, "fêmea", "bezerra", 2),
        ]
        s = _calculate_summary(lots)
        cats = s["category_animals"]
        # bezerro has most (35), should be first
        assert list(cats.keys())[0] == "bezerro"
        assert cats["bezerro"] == 35
        # all categories present
        assert "vaca" in cats
        assert "garrote" in cats

    def test_average_price_excludes_nulls(self):
        lots = [
            lot(1, "macho", "bezerro", 5, unit_price=3000.0),
            lot(2, "fêmea", "vaca", 3, unit_price=5000.0),
            lot(3, "misto", "garrote", 4, unit_price=None),  # excluded
        ]
        s = _calculate_summary(lots)
        assert s["avg_price"] == pytest.approx(4000.0)

    def test_average_price_excludes_zero(self):
        lots = [
            lot(1, "macho", "bezerro", 5, unit_price=3000.0),
            lot(2, "fêmea", "vaca", 3, unit_price=0.0),  # excluded (zero price)
        ]
        s = _calculate_summary(lots)
        assert s["avg_price"] == pytest.approx(3000.0)

    def test_average_price_zero_when_no_prices(self):
        lots = [lot(1, "macho", "bezerro", 5, unit_price=None)]
        assert _calculate_summary(lots)["avg_price"] == 0

    def test_avg_price_by_category(self):
        lots = [
            lot(1, "macho", "bezerro", 5, unit_price=2000.0),
            lot(2, "macho", "bezerro", 3, unit_price=4000.0),
            lot(3, "fêmea", "vaca", 2, unit_price=6000.0),
        ]
        s = _calculate_summary(lots)
        assert s["category_prices"]["bezerro"] == pytest.approx(3000.0)
        assert s["category_prices"]["vaca"] == pytest.approx(6000.0)

    def test_category_excluded_from_price_if_no_price(self):
        lots = [
            lot(1, "macho", "bezerro", 5, unit_price=3000.0),
            lot(2, "fêmea", "vaca", 3, unit_price=None),
        ]
        s = _calculate_summary(lots)
        assert "bezerro" in s["category_prices"]
        assert "vaca" not in s["category_prices"]

    def test_sold_count(self):
        lots = [
            lot(1, "macho", "bezerro", 5, sold=True),
            lot(2, "fêmea", "vaca", 3, sold=True),
            lot(3, "misto", "garrote", 4, sold=False),
            lot(4, "misto", "novilha", 2, sold=None),  # not counted either way
        ]
        s = _calculate_summary(lots)
        assert s["sold"] == 2
        assert s["not_sold"] == 1

    def test_sold_zero_when_all_none(self):
        lots = [lot(1, "macho", "bezerro", 5, sold=None)]
        s = _calculate_summary(lots)
        assert s["sold"] == 0
        assert s["not_sold"] == 0

    def test_single_lot(self):
        lots = [lot(1, "macho", "bezerro", 7, unit_price=3500.0, sold=True)]
        s = _calculate_summary(lots)
        assert s["total_lots"] == 1
        assert s["total_animals"] == 7
        assert s["avg_price"] == pytest.approx(3500.0)
        assert s["sold"] == 1

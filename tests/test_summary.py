"""Tests for main._calculate_summary statistics."""
import sys
from io import StringIO
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from click.testing import CliRunner
from rich.console import Console
import main
from models.lot import Lot
from main import _calculate_summary, cli


def lot(lot_number, sex, category, num_animals, unit_price=None, sold=None, breed="Nelore"):
    return Lot(
        lot_number=lot_number, sex=sex, category=category,
        num_animals=num_animals, breed=breed,
        unit_price=unit_price, sold=sold,
    )


def test_lot_table_labels_disputed_and_unconfirmed_prices(monkeypatch):
    output = StringIO()
    monkeypatch.setattr(main, "console", Console(file=output, width=160))
    main._print_table([
        lot(1, "macho", "bezerro", 5, unit_price=2400, sold=True).model_copy(
            update={"price_needs_review": True}
        ),
        lot(2, "macho", "bezerro", 5, unit_price=3100),
    ])
    assert "2,400.00 (em revisão)" in output.getvalue()
    assert "3,100.00 (lance)" in output.getvalue()


class TestCalculateSummary:
    def test_empty_returns_empty_dict(self):
        assert _calculate_summary([]) == {}

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
            lot(1, "macho", "bezerro", 5, unit_price=3000.0, sold=True),
            lot(2, "fêmea", "vaca", 3, unit_price=5000.0, sold=True),
            lot(3, "misto", "garrote", 4, unit_price=None, sold=True),
        ]
        s = _calculate_summary(lots)
        assert s["avg_price"] == pytest.approx(3750.0)

    def test_average_price_excludes_zero(self):
        lots = [
            lot(1, "macho", "bezerro", 5, unit_price=3000.0, sold=True),
            lot(2, "fêmea", "vaca", 3, unit_price=0.0, sold=True),
        ]
        s = _calculate_summary(lots)
        assert s["avg_price"] == pytest.approx(3000.0)

    def test_average_price_zero_when_no_prices(self):
        lots = [lot(1, "macho", "bezerro", 5, unit_price=None)]
        assert _calculate_summary(lots)["avg_price"] == 0

    def test_category_excluded_from_price_if_no_price(self):
        lots = [
            lot(1, "macho", "bezerro", 5, unit_price=3000.0, sold=True),
            lot(2, "fêmea", "vaca", 3, unit_price=None),
        ]
        s = _calculate_summary(lots)
        assert "bezerro" in s["category_prices"]
        assert "vaca" not in s["category_prices"]

    def test_price_averages_are_weighted_by_animals(self):
        lots = [
            lot(1, "macho", "bezerro", 1, unit_price=1000.0, sold=True),
            lot(2, "macho", "bezerro", 9, unit_price=9000.0, sold=True),
            lot(3, "fêmea", "vaca", 2, unit_price=5000.0, sold=True),
        ]

        summary = _calculate_summary(lots)

        # (1×1,000 + 9×9,000 + 2×5,000) / 12 = 7,666.67...
        assert summary["avg_price"] == pytest.approx(7666.666666666667)
        assert summary["category_prices"]["bezerro"] == pytest.approx(8200.0)

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
        assert s["unknown_sale_status"] == 1

    def test_only_confirmed_uncontested_sales_contribute_to_price(self):
        summary = _calculate_summary([
            lot(1, "macho", "bezerro", 5, unit_price=2400, sold=True),
            lot(2, "macho", "bezerro", 10, unit_price=3100),
            lot(3, "macho", "bezerro", 10, unit_price=5000, sold=False),
            lot(4, "macho", "bezerro", 10, unit_price=6000, sold=True).model_copy(
                update={"price_needs_review": True}
            ),
        ])
        assert summary["avg_price"] == 2400
        assert summary["priced_lots"] == 1
        assert summary["priced_animals"] == 5

    def test_single_lot(self):
        lots = [lot(1, "macho", "bezerro", 7, unit_price=3500.0, sold=True)]
        s = _calculate_summary(lots)
        assert s["total_lots"] == 1
        assert s["total_animals"] == 7
        assert s["avg_price"] == pytest.approx(3500.0)
        assert s["sold"] == 1


class TestCliValidation:
    def test_screenshot_interval_rejects_zero(self):
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["https://www.youtube.com/watch?v=test", "--screenshot-interval", "0"],
        )
        assert result.exit_code != 0
        assert "Invalid value for '--screenshot-interval'" in result.output

    def test_ocr_video_height_rejects_unsupported_resolution(self):
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["https://www.youtube.com/watch?v=test", "--ocr-video-height", "1080"],
        )
        assert result.exit_code != 0
        assert "Invalid value for '--ocr-video-height'" in result.output

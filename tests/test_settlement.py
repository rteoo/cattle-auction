import json

import pytest

from models.lot import Lot
from pipeline.aggregator import Window
from pipeline.extractor import _lot_has_window_support, extract_lots
from pipeline.settlement import _review_update


def test_distinct_display_labels_survive_extraction_and_checkpoint(tmp_path):
    class Client:
        calls = 0

        def complete(self, system, user):
            self.calls += 1
            return json.dumps([{"lot_number": label} for label in ("0", "00", "11A", "08B", "B-19")])

    prompt = tmp_path / "prompt.txt"
    prompt.write_text("extract", encoding="utf-8")
    output = tmp_path / "lots.json"
    windows = [Window(0, 600, "00:00:00 - 00:10:00", "[00:02:00] TELA: LOTE | 08B | 03NOVILHAS")]
    client = Client()
    for _ in range(2):
        lots = extract_lots(windows, client, prompt, output)
        assert {lot.lot_number for lot in lots} == {"0", "00", "11A", "08B", "B-19"}
    assert client.calls == 1
    assert Lot(lot_number=8).lot_number == "8"
    assert _lot_has_window_support("08B", windows[0])
    assert not _lot_has_window_support("08", windows[0])
    assert not _lot_has_window_support("8", windows[0])


def test_review_requires_real_citation_and_matching_screen_amount():
    lot = Lot(lot_number="041A", price_needs_review=True, sold=None)
    evidence = {
        "00:27:08": [("ÁUDIO", "Comprou a fazenda de Rio Claro.")],
        "00:27:01": [("TELA", "LOTE | 041A | VALORPORANIMAL | R$ 3.150")],
    }
    answer = {
        "sold": True, "sale_timestamp": "00:27:08", "sale_quote": "comprou a fazenda",
        "unit_price": 3150, "price_timestamp": "00:27:01", "price_quote": "R$ 3.150",
    }
    revised = _review_update(lot, answer, evidence, (0, 3000))
    assert revised.sold is True
    assert revised.unit_price == 3150  # Source-confirmed values may be statistical outliers.
    assert revised.price_needs_review is False
    wrong = _review_update(lot, {**answer, "unit_price": 3155}, evidence, (0, 3000))
    assert wrong.sold is True and wrong.unit_price is None and wrong.price_needs_review
    assert _review_update(lot, {**answer, "sale_quote": "venda inexistente"}, evidence, (0, 3000)) is None


def test_settlement_recovers_adjacent_closing_and_recomputes_total(tmp_path):
    class Client:
        def complete(self, system, user):
            if system == "extract":
                if user.startswith("Segmento [00:00:00"):
                    return '[{"lot_number":"07C","timestamp_start":"00:09:50","num_animals":3}]'
                return "[]"
            assert "[00:10:12] ÁUDIO:" in user
            return json.dumps({"outcomes": [{
                "lot_number": "07C", "sold": True, "sale_timestamp": "00:10:12",
                "sale_quote": "Levou três cabeças", "unit_price": 2700,
                "price_timestamp": "00:10:12", "price_quote": "dois mil e setecentos por cabeça",
            }]})

    extract_prompt = tmp_path / "extract.txt"
    review_prompt = tmp_path / "settlement.txt"
    extract_prompt.write_text("extract", encoding="utf-8")
    review_prompt.write_text("reconcile", encoding="utf-8")
    windows = [
        Window(0, 600, "00:00:00 - 00:10:00", "[00:09:50] TELA: LOTE | 07C"),
        Window(600, 1200, "00:10:00 - 00:20:00",
               "[00:10:12] ÁUDIO: Levou três cabeças a dois mil e setecentos por cabeça."),
    ]
    (lot,) = extract_lots(windows, Client(), extract_prompt, tmp_path / "result.json",
                          settlement_prompt_path=review_prompt)
    assert (lot.lot_number, lot.sold, lot.sale_timestamp) == ("07C", True, "00:10:12")
    assert (lot.unit_price, lot.total_price, lot.price_needs_review) == (2700, 8100, False)


def test_sale_review_resume_reuses_finished_batches(tmp_path):
    from pipeline.settlement import reconcile_settlements

    lots = {
        "19": Lot(lot_number="19", timestamp_start="00:01:00"),
        "37B": Lot(lot_number="37B", timestamp_start="00:17:00"),
    }
    windows = [
        Window(0, 600, "first", "[00:01:20] ÁUDIO: Retirou o lote dezenove."),
        Window(900, 1500, "second", "[00:17:20] ÁUDIO: Retirou o lote 37B."),
    ]
    path = tmp_path / "settlement-progress.json"

    class Client:
        def __init__(self, fail):
            self.fail = fail
            self.calls = []

        def complete(self, system, user):
            label = "19" if '"lot_number": "19"' in user else "37B"
            self.calls.append(label)
            if label == "37B" and self.fail:
                raise RuntimeError("temporarily unavailable")
            when, quote = {
                "19": ("00:01:20", "Retirou o lote dezenove."),
                "37B": ("00:17:20", "Retirou o lote 37B."),
            }[label]
            return json.dumps({"outcomes": [{
                "lot_number": label, "sold": False, "sale_timestamp": when, "sale_quote": quote,
            }]})

    with pytest.raises(RuntimeError, match="temporarily unavailable"):
        reconcile_settlements(lots, windows, Client(fail=True), "review", (0, 5000), path, {"v": 1}, set())

    second = Client(fail=False)
    revised = reconcile_settlements(lots, windows, second, "review", (0, 5000), path, {"v": 1}, set())
    assert second.calls == ["37B"]
    assert {label: lot.sold for label, lot in revised.items()} == {"19": False, "37B": False}

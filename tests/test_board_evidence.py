from models.lot import Lot
from pipeline.aggregator import Window
from pipeline.board_evidence import filter_unmarked_lot_ids, finalize_board_evidence


def test_board_cross_check_flags_conflicting_amount_only_on_same_lot():
    lots = {
        "12A": Lot(lot_number="12A", sold=True, unit_price=3800, sale_timestamp="00:01:04"),
        "12": Lot(lot_number="12", sold=True, unit_price=3800, sale_timestamp="00:01:38"),
    }
    window = Window(0, 600, "test", "\n".join([
        "[00:01:00] TELA: LOTE | VALOR POR ANIMAL | 12A | R$ | 3.700",
        "[00:01:30] TELA: LOTE | VALOR POR ANIMAL | 12 | R$ | 3.800",
    ]))
    checked = finalize_board_evidence(lots, [window])
    assert checked["12A"].price_needs_review
    assert checked["12A"].unit_price == 3800
    assert not checked["12"].price_needs_review


def test_undated_recap_maps_only_to_single_supported_padded_label():
    lots = {
        "007": Lot(lot_number="007", num_animals=3, sold=False, sale_timestamp="00:01:12"),
        "7": Lot(lot_number="7", sold=True, unit_price=3200, sale_timestamp="00:05:30"),
        "07B": Lot(lot_number="07B", num_animals=1),
        "44": Lot(lot_number="44"),
    }
    window = Window(0, 600, "test", "\n".join([
        "[00:01:00] TELA: LOTE | VALOR POR CABEÇA | 007 | 03NOVILHAS-14M-NELORE",
        "[00:05:30] ÁUDIO: Lote 7, três mil e duzentos, comprou.",
    ]))
    checked = finalize_board_evidence(lots, [window])
    assert set(checked) == {"007", "07B"}
    assert (checked["007"].sold, checked["007"].unit_price, checked["007"].total_price) == (True, 3200, 9600)
    assert checked["007"].category == "novilha"


def test_recap_cannot_merge_distinct_board_labels_or_choose_between_them():
    lots = {
        "0": Lot(lot_number="0", sold=True, unit_price=2100, sale_timestamp="00:05:30"),
        "00": Lot(lot_number="00", num_animals=3),
        "000": Lot(lot_number="000", num_animals=7),
    }
    window = Window(0, 600, "test", "\n".join([
        "[00:01:00] TELA: LOTE | VALOR POR ANIMAL | 00 | 03NOVILHAS-14M-NELORE",
        "[00:02:00] TELA: LOTE | VALOR POR ANIMAL | 000 | 07NOVILHAS-12M-NELORE",
        "[00:05:30] ÁUDIO: Lote 0, dois mil e cem, comprou.",
    ]))
    assert set(finalize_board_evidence(lots, [window])) == {"0", "00", "000"}



def test_marker_guard_keeps_board_ids_and_audio_only_lots():
    lots = {
        label: Lot(lot_number=label, timestamp_start=timestamp)
        for label, timestamp in {
            "0": "00:01:00",
            "00": "00:01:30",
            "11A": "00:02:00",
            "08B": "00:02:30",
            "105": "00:03:00",
            "41C": "00:03:30",
            "4321-8765": "00:04:30",
        }.items()
    }
    window = Window(0, 600, "test", "\n".join([
        "[00:01:00] TELA: LOTE | Fazenda | VALOR POR ANIMAL | 0",
        "[00:01:30] TELA: Lote00 | Valor: 2.500",
        "[00:02:00] TELA: Lote | 11A",
        "[00:02:30] TELA: Fundo sem cartela | 08B",
        "[00:02:55] ÁUDIO: Lote cento e cinco, duas novilhas.",
        "[00:03:00] TELA: Aviso | 105",
        "[00:03:30] TELA: Aviso | 41C",
        "[00:03:38] ÁUDIO: Lote 41C, pronto para venda.",
        "[00:04:00] TELA: Lote08B",
        "[00:04:30] TELA: Apoio | Contato: (00) 4321-8765",
    ]))
    assert set(filter_unmarked_lot_ids(lots, [window])) == {
        "0", "00", "11A", "08B", "105", "41C",
    }


def test_marker_guard_preserves_lots_with_no_matching_ocr_candidate():
    lot = Lot(lot_number="24", timestamp_start="00:00:30")
    window = Window(0, 600, "test", "[00:00:30] TELA: Patrocinador | 240 | 24B")
    assert filter_unmarked_lot_ids({"24": lot}, [window]) == {"24": lot}

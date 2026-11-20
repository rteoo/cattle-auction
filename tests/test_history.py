import sqlite3
from contextlib import closing

import pytest

from models.lot import AuctionResult, Lot
from pipeline import history


def auction(video_id, lots, *, date="17/11/2026", city="Rio Verde"):
    return AuctionResult(
        video_id=video_id,
        video_url=f"https://youtu.be/{video_id}",
        date=date,
        city=city,
        auctioneer="Leilões do Cerrado",
        farm="Fazenda Boa Vista",
        auction_type="corte",
        total_lots=len(lots),
        lots=lots,
    )


def test_history_filters_prices_and_replaces_only_the_rerun_video(tmp_path):
    path = tmp_path / "history.sqlite3"
    history.initialize(path)
    history.save(path, auction("first", [
        Lot(lot_number="0", category="novilha", unit_price="2.500,75", sold=True),
        Lot(lot_number="00", unit_price=2600, sold=True, price_needs_review=True),
        Lot(lot_number="11A", unit_price=2700, sold=False),
        Lot(lot_number="08B", unit_price=2900, sold=None),
    ]))
    history.save(path, auction("other", [], date="11/2026", city="Mineiros"))
    history.initialize(path)

    with closing(sqlite3.connect(path)) as db:
        assert db.execute("SELECT lot_number FROM lots ORDER BY lot_number").fetchall() == [
            ("0",), ("00",), ("08B",), ("11A",)
        ]
        assert db.execute(
            "SELECT auction_date, city, auctioneer, farm, auction_type, category, "
            "lot_number, unit_price_cents FROM confirmed_prices"
        ).fetchall() == [
            ("2026-11-17", "Rio Verde", "Leilões do Cerrado", "Fazenda Boa Vista",
             "corte", "novilha", "0", 250075)
        ]
        assert db.execute(
            "SELECT auction_date, date_reported FROM auctions WHERE video_id = 'other'"
        ).fetchone() == (None, "11/2026")

        correction = Lot(lot_number="00", unit_price=2800, sold=True)
        history.save(path, auction("first", [correction], city="Jataí"))
        assert db.execute("SELECT video_id, city FROM auctions ORDER BY video_id").fetchall() == [
            ("first", "Jataí"), ("other", "Mineiros")
        ]
        assert db.execute(
            "SELECT lot_number, unit_price_cents, city FROM confirmed_prices"
        ).fetchall() == [("00", 280000, "Jataí")]


def test_failed_write_rolls_back_auction_and_lots(tmp_path):
    path = tmp_path / "history.sqlite3"
    history.initialize(path)
    history.save(path, auction("first", [Lot(lot_number="09")]))
    duplicate = Lot(lot_number="00")
    with pytest.raises(history.HistoryError, match="Could not save video first"):
        history.save(path, auction("first", [duplicate, duplicate], city="Jataí"))

    with closing(sqlite3.connect(path)) as db:
        assert db.execute("SELECT city FROM auctions").fetchall() == [("Rio Verde",)]
        assert db.execute("SELECT lot_number FROM lots").fetchall() == [("09",)]


def test_initialize_rejects_unrelated_database(tmp_path):
    path = tmp_path / "another.sqlite3"
    with closing(sqlite3.connect(path)) as db:
        db.execute("CREATE TABLE auctions (unrelated TEXT)")
        db.execute("INSERT INTO auctions VALUES ('preserve')")
        db.commit()

    with pytest.raises(history.HistoryError, match="dedicated auction history"):
        history.initialize(path)

    with closing(sqlite3.connect(path)) as db:
        assert db.execute("SELECT unrelated FROM auctions").fetchall() == [("preserve",)]
        assert db.execute("PRAGMA user_version").fetchone() == (0,)

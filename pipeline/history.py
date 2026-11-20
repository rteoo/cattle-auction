"""Persist the latest completed extraction per video in a local SQLite database."""

import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path

from models.lot import AuctionResult, price_to_cents

_APPLICATION_ID = int.from_bytes(b"CAUC", "big")
_SCHEMA_VERSION = 1

_SCHEMA = (
    """
    CREATE TABLE auctions (
        video_id TEXT PRIMARY KEY NOT NULL,
        video_url TEXT NOT NULL,
        auction_date TEXT,
        date_reported TEXT,
        city TEXT,
        auctioneer TEXT,
        farm TEXT,
        auction_type TEXT,
        notes TEXT,
        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE lots (
        video_id TEXT NOT NULL REFERENCES auctions(video_id) ON DELETE CASCADE,
        lot_number TEXT NOT NULL,
        sex TEXT,
        category TEXT,
        num_animals INTEGER,
        age_months INTEGER,
        breed TEXT,
        unit_price_cents INTEGER,
        total_price_cents INTEGER,
        sold INTEGER CHECK (sold IN (0, 1)),
        timestamp_start TEXT,
        notes TEXT,
        sale_timestamp TEXT,
        raw_price TEXT,
        price_basis TEXT,
        price_needs_review INTEGER NOT NULL CHECK (price_needs_review IN (0, 1)),
        price_timestamp TEXT,
        PRIMARY KEY (video_id, lot_number)
    )
    """,
    "CREATE INDEX auctions_city_date ON auctions (city, auction_date)",
    "CREATE INDEX auctions_auctioneer_date ON auctions (auctioneer, auction_date)",
    """
    CREATE VIEW confirmed_prices AS
    SELECT a.video_id, a.video_url, a.auction_date, a.date_reported,
           a.city, a.auctioneer, a.farm, a.auction_type,
           l.lot_number, l.category, l.sex, l.breed, l.age_months, l.num_animals,
           l.unit_price_cents, l.total_price_cents, l.sale_timestamp, l.price_timestamp
    FROM auctions AS a JOIN lots AS l USING (video_id)
    WHERE l.sold = 1 AND l.price_needs_review = 0 AND l.unit_price_cents > 0
    """,
)


class HistoryError(RuntimeError):
    """The requested history database could not be initialized or written."""


def initialize(path: Path) -> None:
    """Create the schema before the pipeline spends time or API credits."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(path, timeout=10)) as db, db:
            # Check the file and create the schema under the same write lock.
            db.execute("BEGIN IMMEDIATE")
            application_id = db.execute("PRAGMA application_id").fetchone()[0]
            version = db.execute("PRAGMA user_version").fetchone()[0]
            if (application_id, version) == (_APPLICATION_ID, _SCHEMA_VERSION):
                return
            if application_id == _APPLICATION_ID:
                raise HistoryError(f"Unsupported auction history schema version {version} in {path}.")
            existing_schema = db.execute(
                "SELECT 1 FROM sqlite_schema WHERE name NOT GLOB 'sqlite_*' LIMIT 1"
            ).fetchone()
            if application_id or version or existing_schema:
                raise HistoryError(f"Use a dedicated auction history database. {path} already has another schema.")
            for statement in _SCHEMA:
                db.execute(statement)
            db.execute(f"PRAGMA application_id = {_APPLICATION_ID}")
            db.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
    except (OSError, sqlite3.Error) as exc:
        raise HistoryError(f"Could not open SQLite history at {path}: {exc}") from exc


def _iso_auction_date(value: str | None) -> str | None:
    if value:
        for pattern in ("%d/%m/%Y", "%Y-%m-%d"):
            try:
                return datetime.strptime(value.strip(), pattern).date().isoformat()
            except ValueError:
                continue
    return None


def save(path: Path, result: AuctionResult) -> None:
    """Replace one completed auction snapshot atomically, preserving other videos."""
    try:
        with closing(sqlite3.connect(path, timeout=10)) as db, db:
            db.execute("PRAGMA foreign_keys = ON")
            db.execute("BEGIN IMMEDIATE")
            if (db.execute("PRAGMA application_id").fetchone()[0] != _APPLICATION_ID
                    or db.execute("PRAGMA user_version").fetchone()[0] != _SCHEMA_VERSION):
                raise HistoryError(f"Initialize the auction history database at {path} before writing.")
            db.execute(
                """
                INSERT INTO auctions (
                    video_id, video_url, auction_date, date_reported, city,
                    auctioneer, farm, auction_type, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (video_id) DO UPDATE SET
                    video_url = excluded.video_url, auction_date = excluded.auction_date,
                    date_reported = excluded.date_reported, city = excluded.city,
                    auctioneer = excluded.auctioneer, farm = excluded.farm,
                    auction_type = excluded.auction_type, notes = excluded.notes,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (result.video_id, result.video_url, _iso_auction_date(result.date), result.date,
                 result.city, result.auctioneer, result.farm, result.auction_type, result.notes),
            )
            db.execute("DELETE FROM lots WHERE video_id = ?", (result.video_id,))
            db.executemany(
                """
                INSERT INTO lots (
                    video_id, lot_number, sex, category, num_animals, age_months, breed,
                    unit_price_cents, total_price_cents, sold, timestamp_start, notes,
                    sale_timestamp, raw_price, price_basis, price_needs_review, price_timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    (result.video_id, lot.lot_number, lot.sex, lot.category, lot.num_animals,
                     lot.age_months, lot.breed, price_to_cents(lot.unit_price),
                     price_to_cents(lot.total_price), lot.sold, lot.timestamp_start, lot.notes,
                     lot.sale_timestamp, lot.raw_price, lot.price_basis, lot.price_needs_review,
                     lot.price_timestamp)
                    for lot in result.lots
                ),
            )
    except (sqlite3.Error, ValueError, OverflowError) as exc:
        raise HistoryError(f"Could not save video {result.video_id} to SQLite at {path}: {exc}") from exc

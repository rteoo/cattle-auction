import re
from decimal import Decimal, InvalidOperation
from typing import Literal

from pydantic import BaseModel, Field, field_validator


_CATEGORY_PLURAL_MAP = {
    "bezerros": "bezerro",
    "bezerras": "bezerra",
    "garrotes": "garrote",
    "novilhos": "novilho",
    "novilhas": "novilha",
    "bois": "boi",
    "touros": "touro",
    "vacas": "vaca",
    "machos": "macho",
    "fêmeas": "fêmea",
    "mistos": "misto",
}


_BR_PRICE = re.compile(r"(?:[0-9]+|[0-9]{1,3}(?:\.[0-9]{3})+)(?:,[0-9]{1,2})?\Z")
_DECIMAL_PRICE = re.compile(r"[0-9]+(?:\.[0-9]{1,2})?\Z")


def price_to_cents(value: object) -> int | None:
    """Parse a BRL amount without guessing its magnitude or price basis."""
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip().removeprefix("R$").strip()
        if not text:
            return None
        if _BR_PRICE.fullmatch(text):
            value = text.replace(".", "").replace(",", ".")
        elif _DECIMAL_PRICE.fullmatch(text):
            value = text
        else:
            raise ValueError("invalid BRL amount")
    elif isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        raise ValueError("invalid BRL amount")

    try:
        amount = Decimal(str(value))
        if not amount.is_finite() or amount < 0:
            raise ValueError("price must be finite and non-negative")
        cents = amount * 100
        if cents != cents.to_integral_value():
            raise ValueError("price must have at most two decimal places")
        return int(cents)
    except InvalidOperation as exc:
        raise ValueError("invalid BRL amount") from exc


def coerce_price_value(value: object) -> float | None:
    cents = price_to_cents(value)
    if cents is None:
        return None
    try:
        return cents / 100
    except OverflowError as exc:
        raise ValueError("BRL amount cannot be represented") from exc


class Lot(BaseModel):
    lot_number: int = Field(ge=0)
    sex: str | None = None  # macho, fêmea, misto
    category: str | None = None
    num_animals: int | None = Field(default=None, gt=0)
    age_months: int | None = None
    breed: str | None = None
    unit_price: float | None = None
    total_price: float | None = None
    sold: bool | None = None  # True = arrematado, False = não vendido/retirado, None = indefinido
    timestamp_start: str | None = None  # HH:MM:SS
    notes: str | None = None
    raw_price: str | None = None
    price_basis: Literal["per_head", "per_arroba", "lot_total", "unknown"] | None = None
    price_needs_review: bool = False
    price_timestamp: str | None = None  # position of the quote or confirmed hammer price

    @field_validator("category", mode="before")
    @classmethod
    def normalize_category(cls, v):
        if isinstance(v, str):
            normalized = v.strip().lower()
            return _CATEGORY_PLURAL_MAP.get(normalized, normalized)
        return v

    @field_validator("unit_price", "total_price", mode="before")
    @classmethod
    def coerce_price(cls, v):
        return coerce_price_value(v)


class AuctionResult(BaseModel):
    video_url: str
    video_id: str
    date: str | None = None        # Data do leilão (DD/MM/YYYY)
    city: str | None = None        # Cidade / município do leilão
    auctioneer: str | None = None  # Casa leiloeira ou leiloeiro
    farm: str | None = None        # Nome da fazenda ou expositor principal
    auction_type: str | None = None  # Tipo: "corte", "reprodução", "misto", etc.
    notes: str | None = None       # Outras informações relevantes do evento
    total_lots: int
    lots: list[Lot]
    # Estimated USD spent by THIS run (LLM tokens + cloud transcription).
    # Stages served from checkpoints contribute nothing, so a fully resumed
    # run reports ~0. None only when the field predates this schema.
    cost_usd: float | None = None

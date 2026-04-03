from pydantic import BaseModel, field_validator
from typing import Literal


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


class Lot(BaseModel):
    lot_number: int
    sex: str  # macho, fêmea, misto
    category: str  # bezerro, bezerra, garrote, novilha, boi, vaca, touro, etc.
    num_animals: int
    age_months: int | None = None
    breed: str  # Nelore, Anelorado, Mestiço, Cruzado, etc.
    unit_price: float | None = None
    total_price: float | None = None
    sold: bool | None = None  # True = arrematado, False = não vendido/retirado, None = indefinido
    timestamp_start: str | None = None  # HH:MM:SS
    notes: str | None = None

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
        if v is None:
            return None
        if isinstance(v, str):
            # Strip R$, dots (BR thousand separator), convert comma to dot
            v = v.replace("R$", "").replace(".", "").replace(",", ".").strip()
            return float(v) if v else None
        f = float(v)
        # Guard against LLM outputting BR thousand-separator prices as JSON floats.
        # e.g. "5.160" in JSON → Python float 5.16, but intended price is R$5,160.
        # In Brazilian cattle auctions a per-head price below R$100 is not realistic,
        # so values like 5.16 / 2.75 are almost certainly 5160 / 2750 mis-parsed.
        if 0 < f < 100:
            f = f * 1000
        return f


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

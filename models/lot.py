from pydantic import BaseModel, field_validator


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


def coerce_price_value(v):
    """Coerce Brazilian auction price formats to float.

    Accepts strings such as "R$ 3.100,00" and numeric values. Numeric values
    below 100 are treated as likely JSON parses of BR thousand-separated prices
    such as 5.160 -> 5.16, then repaired to 5160.
    """
    if v is None:
        return None
    if isinstance(v, str):
        v = v.replace("R$", "").replace(".", "").replace(",", ".").strip()
        return float(v) if v else None
    f = float(v)
    if 0 < f < 100:
        f = f * 1000
    return f


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

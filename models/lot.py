from pydantic import BaseModel, field_validator
from typing import Literal


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

    @field_validator("unit_price", "total_price", mode="before")
    @classmethod
    def coerce_price(cls, v):
        if v is None:
            return None
        if isinstance(v, str):
            # Strip R$, dots, convert comma to dot
            v = v.replace("R$", "").replace(".", "").replace(",", ".").strip()
            return float(v) if v else None
        return float(v)


class AuctionResult(BaseModel):
    video_url: str
    video_id: str
    total_lots: int
    lots: list[Lot]

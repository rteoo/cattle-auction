"""Use explicit lot-board evidence to qualify reconciled extraction output."""

import re
import statistics
import unicodedata
from collections import defaultdict
from dataclasses import dataclass

from models.lot import Lot, price_to_cents
from pipeline.aggregator import Window
from pipeline.settlement import _seconds, _source_lines

_MONEY = re.compile(r"\d{1,3}(?:\.\d{3})+(?:,\d{1,2})?|\d+(?:,\d{1,2})?")
_PER_HEAD_HEADERS = {"VALORPORANIMAL", "PRECOPORANIMAL", "VALORPORCABECA", "PRECOPORCABECA"}
_BOARD_DESCRIPTION = re.compile(r"^(\d+)\s*([A-Za-zÀ-ÿ ]+?)(?:\s+P\.?)?\s*[-–]\s*\d+\s*M\b", re.I)
_CATEGORIES = {
    "bezerras": "bezerra", "bezerros": "bezerro", "garrotes": "garrote", "novilhas": "novilha",
    "novilhos": "novilho", "vacas": "vaca", "bois": "boi", "touros": "touro",
}
_FEMALE = {"bezerra", "novilha", "vaca"}
_MALE = {"bezerro", "garrote", "novilho", "boi", "touro"}


# The field name may be a separate OCR token or printed inline as "Lote08".
_LOT_MARKER = re.compile(r"(?<!\w)(?i:lote|lot)(?![^\W\d_])")
_LOT_FIELD = re.compile(
    rf"{_LOT_MARKER.pattern}(?:\s+(?i:n)[º°.])?\s*[:#]?\s*"
    r"([A-Za-z0-9]+(?:[./-][A-Za-z0-9]+)*)(?![\w/-])"
)
_ID_TOKEN = re.compile(r"[A-Za-z0-9]+(?:[./-][A-Za-z0-9]+)*")
_SPOKEN_NUMBER = re.compile(
    r"\b(?:lote|loto|lota)\s+(?:(?:do|de|número|numero)\s+)*"
    r"(?:zero|um|uma|dois|duas|tr[eê]s|quatro|cinco|seis|meia|sete|oito|nove|"
    r"dez|onze|doze|treze|catorze|quatorze|quinze|dez\w+|vinte|trinta|quarenta|"
    r"cinquenta|sessenta|setenta|oitenta|noventa|cem|cento|duzentos|trezentos|"
    r"quatrocentos|quinhentos|seiscentos|setecentos|oitocentos|novecentos|mil)\b",
    re.IGNORECASE,
)


def filter_unmarked_lot_ids(lots: dict[str, Lot], windows: list[Window]) -> dict[str, Lot]:
    """Reject exact OCR-only IDs anchored to a frame with no lot-board marker."""
    board_ids = set()
    audio_ids = set()
    spoken_number_times = []
    unmarked_ids = defaultdict(set)

    for timestamp, entries in _source_lines(windows).items():
        when = _seconds(timestamp)
        if when is None:
            continue
        for medium, text in entries:
            if medium == "ÁUDIO":
                audio_ids.update(_LOT_FIELD.findall(text))
                if _SPOKEN_NUMBER.search(text):
                    spoken_number_times.append(when)
            elif _LOT_MARKER.search(text):
                board_ids.update(_LOT_FIELD.findall(text))
                board_ids.update(token.strip() for token in text.split("|"))
            else:
                unmarked_ids[when].update(_ID_TOKEN.findall(text))

    kept = {}
    for label, lot in lots.items():
        start = _seconds(lot.timestamp_start)
        # Nearby spoken numbers can identify audio-only lots even when ASR spells them out.
        spoken = (label.isdecimal() and start is not None and
                  any(abs(start - when) <= 60 for when in spoken_number_times))
        if (start is not None and label in unmarked_ids.get(start, ())
                and label not in board_ids and label not in audio_ids and not spoken):
            continue
        kept[label] = lot
    return kept


@dataclass(frozen=True)
class Board:
    time: int
    label: str
    price: int | None
    quantity: int | None
    category: str | None


def _is_per_head_header(token: str) -> bool:
    plain = unicodedata.normalize("NFKD", token.upper())
    return "".join(char for char in plain if char.isalpha()) in _PER_HEAD_HEADERS


def _board(time: int, text: str, labels) -> Board | None:
    tokens = [token.strip() for token in text.split("|")]
    header = next((i for i, token in enumerate(tokens) if token.upper() == "LOTE"), None)
    if header is None:
        return None
    fields = tokens[header + 1:]
    per_head = next((i for i, token in enumerate(fields) if _is_per_head_header(token)), None)
    if per_head is None:
        return None
    fields = fields[per_head + 1:]
    ids = set(fields) & labels
    if len(ids) != 1:
        return None
    label = ids.pop()

    amounts = set()
    for i, token in enumerate(fields):
        amount = token[2:].strip() if token.startswith("R$") else ""
        if token == "R$" and i + 1 < len(fields):
            amount = fields[i + 1]
        if amount != label and _MONEY.fullmatch(amount):
            value = price_to_cents(amount)
            if value:
                amounts.add(value)
    # OCR can report the amount before the currency symbol when fields are ordered by y-position.
    if not amounts and any(token == "R$" for token in fields):
        amounts = {price_to_cents(token) for token in fields
                   if token != label and "." in token and _MONEY.fullmatch(token)}
    price = next(iter(amounts)) if len(amounts) == 1 else None

    details = next((m for token in fields if (m := _BOARD_DESCRIPTION.match(token))), None)
    quantity = int(details[1]) if details else None
    word = details[2].strip().casefold() if details else None
    category = _CATEGORIES.get(word, word)
    return Board(time, label, price, quantity, category)


def _explicit_recap(label: str, timestamp: str, sources: dict) -> bool:
    mentions = re.compile(rf"\blot[eo]\s+(?:n[º°.]?\s*)?{re.escape(label)}(?![\w/-]|\.[A-Za-z0-9])", re.I)
    closure = re.compile(r"\b(?:comprou|levou|arrematou|arrematado|vendeu|vendido|comercializado)\b", re.I)
    return any(kind == "ÁUDIO" and mentions.search(text) and closure.search(text)
               for kind, text in sources.get(timestamp, ()))


def finalize_board_evidence(lots: dict[str, Lot], windows: list[Window]) -> dict[str, Lot]:
    """Disambiguate orphan recaps and flag conflicts against a matching close board."""
    sources = _source_lines(windows)
    boards = defaultdict(list)
    screen_times = []
    for timestamp, entries in sources.items():
        when = _seconds(timestamp)
        if when is None:
            continue
        for medium, text in entries:
            if medium != "TELA":
                continue
            screen_times.append(when)
            if board := _board(when, text, lots.keys()):
                boards[board.label].append(board)
    # One typical capture gap accounts for the lag between narration and the board.
    timestamps = sorted(set(screen_times))
    gaps = [b - a for a, b in zip(timestamps, timestamps[1:])]
    tolerance = min(60, statistics.median(gaps)) if gaps else 0

    revised = dict(lots)
    for label, lot in lots.items():
        if label in boards or lot.timestamp_start or any((lot.category, lot.num_animals, lot.breed)):
            continue
        if lot.sold is None and not lot.sale_timestamp:
            revised.pop(label)
        elif (label.isdecimal() and lot.sold is True and lot.sale_timestamp
              and _explicit_recap(label, lot.sale_timestamp, sources)):
            candidates = [key for key in boards if key.isdecimal() and int(key) == int(label)]
            if len(candidates) != 1:
                continue
            target = revised[candidates[0]]
            recap_time = _seconds(lot.sale_timestamp)
            prior_time = _seconds(target.sale_timestamp)
            if recap_time is None or (prior_time is not None and recap_time <= prior_time):
                continue
            fields = {"sold", "sale_timestamp", "unit_price", "price_basis", "price_timestamp",
                      "raw_price", "price_needs_review"}
            changes = lot.model_dump(include=fields)
            changes["total_price"] = (price_to_cents(lot.unit_price) * target.num_animals / 100
                                      if lot.unit_price and target.num_animals else None)
            revised[target.lot_number] = target.model_copy(update=changes)
            revised.pop(label)

    for label, observations in boards.items():
        if label not in revised:
            continue
        lot = revised[label]
        changes = {}
        categories = {board.category for board in observations if board.quantity == lot.num_animals}
        if len(categories) == 1:
            category = categories.pop()
            if category in _MALE | _FEMALE and (lot.category is None or lot.category in {"macho", "fêmea"}):
                changes["category"] = category
                if lot.sex in {None, "misto"}:
                    changes["sex"] = "fêmea" if category in _FEMALE else "macho"

        close = _seconds(lot.sale_timestamp)
        if lot.sold is True and lot.unit_price and close is not None:
            local_prices = {board.price for board in observations
                            if board.price and abs(board.time - close) <= tolerance}
            if local_prices and price_to_cents(lot.unit_price) not in local_prices:
                changes["price_needs_review"] = True
        if changes:
            revised[label] = lot.model_copy(update=changes)
    return revised

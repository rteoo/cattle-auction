"""Reconcile uncertain outcomes with source lines from the surrounding windows."""

import json
import re
from collections import defaultdict
from pathlib import Path

from models.lot import Lot, coerce_price_value, price_to_cents
from pipeline.aggregator import Window
from pipeline.checkpoint import write_json

_SOURCE_LINE = re.compile(r"^\[(\d{2,}:\d{2}:\d{2})\] (ÁUDIO|TELA): (.+)$")
_CURRENCY_TOKEN = re.compile(r"(?<![\d.,])\d+(?:\.\d{3})*(?:,\d{1,2})?(?![\d.,])")
_MAX_TARGETS = 8


def _seconds(timestamp: str | None) -> int | None:
    if not isinstance(timestamp, str) or not re.fullmatch(r"\d{2,}:\d{2}:\d{2}", timestamp):
        return None
    h, m, s = map(int, timestamp.split(":"))
    return h * 3600 + m * 60 + s if m < 60 and s < 60 else None


def _source_lines(windows: list[Window]) -> dict[str, list[tuple[str, str]]]:
    lines = dict.fromkeys(line for window in windows for line in window.combined_text.splitlines())
    sources = defaultdict(list)
    for line in lines:
        if match := _SOURCE_LINE.fullmatch(line):
            timestamp, medium, body = match.groups()
            sources[timestamp].append((medium, body))
    return dict(sorted(sources.items(), key=lambda pair: _seconds(pair[0]) or 0))


def _citation(sources, timestamp, quote):
    if not isinstance(timestamp, str) or not isinstance(quote, str):
        return []
    needle = " ".join(quote.casefold().split())
    if not needle:
        return []
    return [(medium, body) for medium, body in sources.get(timestamp, ())
            if needle in " ".join(body.casefold().split())]


def _review_update(lot: Lot, answer: dict, sources, bounds: tuple[float, float]) -> Lot | None:
    sold = answer.get("sold")
    sale_time, sale_quote = answer.get("sale_timestamp"), answer.get("sale_quote")
    if type(sold) is not bool or not _citation(sources, sale_time, sale_quote):
        return None
    update = {"sold": sold, "sale_timestamp": sale_time}
    if sold is False:
        update.update(unit_price=None, total_price=None, raw_price=None, price_basis=None,
                      price_timestamp=None, price_needs_review=False)
        return lot.model_copy(update=update)

    price_time, price_quote = answer.get("price_timestamp"), answer.get("price_quote")
    cited = _citation(sources, price_time, price_quote)
    try:
        price = coerce_price_value(answer.get("unit_price")) if cited else None
    except ValueError:
        price = None
    if price and all(medium == "TELA" for medium, _ in cited):
        amounts = {price_to_cents(token) for token in _CURRENCY_TOKEN.findall(price_quote)}
        if price_to_cents(price) not in amounts:
            price = None

    if price and price > 0:
        update.update(unit_price=price, total_price=None, raw_price=price_quote,
                      price_basis="per_head", price_timestamp=price_time, price_needs_review=False)
    elif (lot.sold is not True or lot.unit_price is None or lot.price_needs_review
          or not bounds[0] <= lot.unit_price <= bounds[1]):
        update.update(unit_price=None, total_price=None, price_needs_review=True)
    return lot.model_copy(update=update)


def _load_review_progress(path: Path, provenance: dict) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict) or data.get("provenance") != provenance:
        return {}
    batches = data.get("batches")
    return batches if isinstance(batches, dict) else {}


def reconcile_settlements(lots: dict[str, Lot], windows: list[Window], client,
                          prompt: str, bounds: tuple[float, float], progress_path: Path,
                          provenance: dict, disputed: set[str]) -> dict[str, Lot]:
    """Review candidates in batches, with adjacent evidence and resumable progress."""
    sources = _source_lines(windows)
    grouped = defaultdict(list)
    for lot in lots.values():
        lo, hi = bounds
        if (lot.lot_number not in disputed and lot.sold is True and lot.unit_price and not lot.price_needs_review
                and lo <= lot.unit_price <= hi and lot.sale_timestamp in sources):
            continue
        anchor = next(
            (t for value in (lot.sale_timestamp, lot.price_timestamp, lot.timestamp_start)
             if (t := _seconds(value)) is not None),
            None,
        )
        if anchor is None:
            continue
        window = next((w for w in reversed(windows) if w.window_start <= anchor <= w.window_end), None)
        if window:
            grouped[window.window_start].append(lot)

    completed = _load_review_progress(progress_path, provenance)
    updates = {}
    for start, targets in sorted(grouped.items()):
        end = next(w.window_end for w in windows if w.window_start == start)
        # ceiling: only the adjacent two minutes are included. Use a searchable evidence index if recaps regularly fall later.
        evidence = {t: lines for t, lines in sources.items() if start - 60 <= _seconds(t) <= end + 120}
        if not evidence:
            continue
        evidence_text = "\n".join(f"[{t}] {kind}: {body}"
                                  for t, lines in evidence.items() for kind, body in lines)
        for offset in range(0, len(targets), _MAX_TARGETS):
            batch = {lot.lot_number: lot for lot in targets[offset:offset + _MAX_TARGETS]}
            key = f"{start}:{offset}"
            if key in completed:
                try:
                    cached = [Lot.model_validate(row) for row in completed[key]]
                    if all(lot.lot_number in batch for lot in cached):
                        updates.update((lot.lot_number, lot) for lot in cached)
                        continue
                except (TypeError, ValueError):
                    pass

            fields = {"lot_number", "timestamp_start", "category", "num_animals", "sold",
                      "sale_timestamp", "unit_price", "price_timestamp"}
            hints = [lot.model_dump(include=fields) for lot in batch.values()]
            print(f"  Reviewing sale evidence for {len(batch)} lot(s) near {start // 60} min.")
            complete = getattr(client, "complete_settlements", client.complete)
            raw = complete(prompt, f"Lotes a revisar:\n{json.dumps(hints, ensure_ascii=False)}"
                                          f"\n\nEvidência:\n{evidence_text}")
            text = raw.strip()
            if text.startswith("```") and text.endswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            answers = json.loads(text)
            if not isinstance(answers, dict) or not isinstance(answers.get("outcomes"), list):
                raise ValueError("Sale review did not return an outcomes array")
            batch_updates = {}
            seen = set()
            for answer in answers["outcomes"]:
                label = answer.get("lot_number") if isinstance(answer, dict) else None
                if not isinstance(label, str) or label not in batch or label in seen:
                    raise ValueError("Sale review returned an unexpected or duplicate lot label")
                seen.add(label)
                if updated := _review_update(batch[label], answer, evidence, bounds):
                    batch_updates[label] = updated
            completed[key] = [lot.model_dump() for lot in batch_updates.values()]
            write_json(progress_path, {"provenance": provenance, "batches": completed})
            updates.update(batch_updates)
    return updates

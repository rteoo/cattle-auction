import json
import os
import re
import statistics
from pathlib import Path

from models.lot import Lot, coerce_price_value
from pipeline.aggregator import Window

# Number of windows from the start to scan for auction metadata
_METADATA_WINDOWS = 3


# Supported models — picked via benchmark (5 videos × 10 models) against a
# human reference. See bench/ for the analysis. Two options:
#
#   "openrouter" (default): Gemini 2.5 Flash-Lite Preview — best speed/cost
#                           (~$0.05/video, 13-24s, 92-100% lot coverage).
#   "openai"      (alt):    GPT-4.1 Mini — best accuracy (100% coverage,
#                           100% categorical match, 0.1% price MAPE, ~$0.13/video).
_DEFAULT_MODELS = {
    "openrouter": "google/gemini-2.5-flash-lite-preview-09-2025",
    "openai": "gpt-4.1-mini",
}


class LLMClient:
    def __init__(self, provider: str, model: str | None = None):
        self.provider = provider
        self.model = model or _DEFAULT_MODELS[provider]
        # Running token-usage counters. Callers can read these after
        # processing (e.g. to estimate $ cost from published per-token prices).
        self.input_tokens = 0
        self.output_tokens = 0
        self.n_calls = 0

        if provider == "openai":
            import openai
            self._client = openai.OpenAI()

        elif provider == "openrouter":
            import openai
            api_key = os.environ.get("OPENROUTER_API_KEY")
            if not api_key:
                raise EnvironmentError("OPENROUTER_API_KEY environment variable is not set.")
            self._client = openai.OpenAI(
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1",
            )

        else:
            raise ValueError(
                f"Unknown provider: {provider!r}. Use 'openrouter' (default, Gemini 2.5 "
                f"Flash-Lite) or 'openai' (alt, GPT-4.1 Mini)."
            )

    def complete(self, system: str, user: str, max_retries: int = 3) -> str:
        import time as _time

        # openai and openrouter share the same SDK interface.
        for attempt in range(max_retries + 1):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    max_tokens=4096,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                )
                # Track usage — both providers return `usage` in the response.
                # Missing usage (rare) → count the call but no token delta.
                self.n_calls += 1
                if resp.usage is not None:
                    self.input_tokens += resp.usage.prompt_tokens or 0
                    self.output_tokens += resp.usage.completion_tokens or 0
                return resp.choices[0].message.content
            except Exception as e:
                if "429" in str(e) and attempt < max_retries:
                    wait = 2 ** attempt * 5  # 5s, 10s, 20s
                    print(f"    Rate limited, retrying in {wait}s...")
                    _time.sleep(wait)
                    continue
                raise


# Heuristic for catching runaway hallucinations in a single window.
# Real 10-minute auction segments rarely produce more than ~10 new lots.
# When the LLM returns more, it's almost always inventing sequential entries.
_MAX_NEW_LOTS_PER_WINDOW = 12


def extract_lots(
    windows: list[Window],
    client: LLMClient,
    prompt_path: Path,
    output_path: Path,
    verify_prompt_path: Path | None = None,
) -> list[Lot]:
    """Process each window with the LLM, deduplicate, and save lots.json.

    If `verify_prompt_path` is provided, lots whose unit_price falls outside
    the statistical bounds (Tukey fence) get a focused second-pass LLM call
    against their source evidence — either confirming, correcting, or
    discarding the price.
    """
    if output_path.exists():
        print(f"  Lots already extracted, loading from cache.")
        return _load(output_path)

    system_prompt = prompt_path.read_text(encoding="utf-8")
    lots_by_number: dict[int, Lot] = {}
    total = len(windows)
    window_failed = False

    for i, window in enumerate(windows, 1):
        already_found = sorted(lots_by_number.keys())
        already_str = str(already_found) if already_found else "nenhum ainda"

        user_content = (
            f"Lotes já encontrados (não repetir): {already_str}\n\n"
            f"Segmento [{window.label}]:\n\n"
            f"{window.combined_text}"
        )

        print(f"  Extracting window {i}/{total}: {window.label}")

        try:
            response = client.complete(system_prompt, user_content)
            new_lots = _parse_response(response)
        except Exception as e:
            print(f"  WARNING: LLM call failed for window {i}: {e}")
            window_failed = True
            continue

        # Guard against hallucination bursts (LLM inventing sequential lot numbers).
        # Keep entries that either reference an existing lot number or have direct
        # support in this window's transcript/OCR evidence.
        if len(new_lots) > _MAX_NEW_LOTS_PER_WINDOW:
            print(
                f"  WARNING: window {i} returned {len(new_lots)} lots "
                f"(>{_MAX_NEW_LOTS_PER_WINDOW}); likely hallucination burst. "
                f"Keeping only lots with existing or direct window evidence."
            )
            existing = set(lots_by_number.keys())
            filtered = [
                l for l in new_lots
                if l.lot_number in existing or _lot_has_window_support(l.lot_number, window)
            ]
            if not filtered:
                # Nothing salvageable — drop the whole window's output.
                print(f"    -> 0 lot(s) kept (all were unsupported new numbers).")
                continue
            new_lots = filtered

        for lot in new_lots:
            _merge(lots_by_number, lot)

        print(f"    -> {len(new_lots)} lot(s) found, {len(lots_by_number)} total so far.")

    if window_failed:
        raise RuntimeError(
            "Lot extraction incomplete: at least one window failed; "
            "no final checkpoint was written."
        )

    # Finalize: apply statistical outlier filter + re-check invariants
    # + fill any missing total deterministically.
    #
    # Per-lot sanity in `_validate_lots` already ran the shape invariants on
    # each incoming lot, but two things can still produce bad values here:
    #   1. Cross-window stitching: merge takes num_animals from window A
    #      and total_price from window B, so the merged lot becomes internally
    #      inconsistent even though each observation was consistent.
    #   2. Self-consistent hallucinations: the LLM emits unit=30000 *and*
    #      total=900000 for a bezerro; shape invariants pass because
    #      30000 × 30 = 900000, but the magnitude is absurd.
    #
    # The statistical bounds catch (2). The re-run of invariants catches (1).
    # Bounds are computed from all unit_prices surviving the merge, so they
    # represent the auction's own distribution — no hard-coded reference.
    observed_prices = [
        lot.unit_price for lot in lots_by_number.values() if lot.unit_price
    ]
    bounds = _compute_price_bounds(observed_prices)
    lo, hi = bounds
    import math
    if math.isfinite(hi) and len(observed_prices) >= 4:
        q1, median, q3 = statistics.quantiles(observed_prices, n=4)
        iqr = q3 - q1
        print(
            f"  Statistical price bounds (Tukey {_IQR_MULTIPLIER}·IQR): "
            f"R$ {lo:,.0f} – R$ {hi:,.0f}  "
            f"[n={len(observed_prices)}, median=R${median:,.0f}, IQR=R${iqr:,.0f}]"
        )

    # Focused re-verification of outliers against source evidence.
    # Instead of silently nulling every price that falls outside the Tukey
    # fence, we give the LLM a second chance to either confirm or correct
    # the outlier by re-examining the transcript+OCR around that lot's
    # timestamp. This recovers legitimate high-priced touros that the
    # statistical filter would otherwise discard.
    if verify_prompt_path and math.isfinite(hi):
        verify_prompt = verify_prompt_path.read_text(encoding="utf-8")
        flagged = [
            (num, lot) for num, lot in lots_by_number.items()
            if lot.unit_price is not None and (lot.unit_price < lo or lot.unit_price > hi)
        ]
        if flagged:
            print(
                f"  Verifying {len(flagged)} outlier(s) against source evidence..."
            )
            for lot_number, lot in flagged:
                verdict = _verify_lot(lot, windows, client, verify_prompt)
                if verdict == "confirm":
                    # Keep existing unit_price. The post-merge sanity pass
                    # skips the bounds check for confirmed lots so it isn't
                    # re-flagged and nulled.
                    print(f"    Lot {lot_number}: confirmed at R$ {lot.unit_price:,.0f}")
                elif isinstance(verdict, (int, float)):
                    # Price corrected — but re-check the correction against the
                    # Tukey bounds. If the LLM's "fix" is still an outlier, it's
                    # not a trustworthy correction (observed in practice: lots
                    # corrected to R$ 42 or R$ 14,000 with no basis in evidence).
                    # Discard in that case rather than accept a bad correction.
                    if lo <= verdict <= hi:
                        data = lot.model_dump()
                        data["unit_price"] = float(verdict)
                        data["total_price"] = None  # recomputed below
                        lots_by_number[lot_number] = Lot(**data)
                        print(f"    Lot {lot_number}: corrected R$ {lot.unit_price:,.0f} → R$ {verdict:,.0f}")
                    else:
                        data = lot.model_dump()
                        data["unit_price"] = None
                        data["total_price"] = None
                        lots_by_number[lot_number] = Lot(**data)
                        print(
                            f"    Lot {lot_number}: correction rejected "
                            f"(R$ {lot.unit_price:,.0f} → R$ {verdict:,.0f} still outside "
                            f"R$ {lo:,.0f}–R$ {hi:,.0f}); discarded"
                        )
                else:  # "discard" or None on failure
                    data = lot.model_dump()
                    data["unit_price"] = None
                    data["total_price"] = None
                    lots_by_number[lot_number] = Lot(**data)
                    print(f"    Lot {lot_number}: discarded (was R$ {lot.unit_price:,.0f})")

    # Post-merge sanity re-check + deterministic total from unit × num_animals.
    # Confirmed outliers that exceed the Tukey bounds would be nulled here by
    # `_sanity_check(lot, bounds=bounds)`, undoing the verification. So for
    # lots that survived verification, skip the bounds check — they're trusted.
    confirmed_lot_numbers = set()
    if verify_prompt_path:
        confirmed_lot_numbers = {
            num for num, lot in lots_by_number.items()
            if lot.unit_price is not None and (lot.unit_price < lo or lot.unit_price > hi)
        }

    for lot_number, lot in lots_by_number.items():
        applied_bounds = None if lot_number in confirmed_lot_numbers else bounds
        checked = _sanity_check(lot, bounds=applied_bounds)
        if checked.total_price is None and checked.unit_price is not None and checked.num_animals:
            data = checked.model_dump()
            data["total_price"] = round(checked.unit_price * checked.num_animals, 2)
            checked = Lot(**data)
        lots_by_number[lot_number] = checked

    lots = sorted(lots_by_number.values(), key=lambda l: l.lot_number)
    _save(lots, output_path)
    return lots


def _verify_lot(
    lot: Lot,
    windows: list[Window],
    client: LLMClient,
    verify_prompt: str,
) -> str | float | None:
    """
    Re-examine evidence for a flagged outlier lot.

    Returns:
      - "confirm"  → evidence supports the current unit_price; keep it
      - float      → evidence supports a different price; use this instead
      - "discard"  → evidence doesn't support any specific price; null it
      - None       → verification failed (LLM error / parse error); null it
    """
    if not lot.timestamp_start:
        return None

    # Find the window whose interval contains this lot's timestamp.
    try:
        ts_sec = _parse_hhmmss(lot.timestamp_start)
    except (TypeError, ValueError):
        return None
    window = next(
        (w for w in windows if w.window_start <= ts_sec <= w.window_end),
        None,
    )
    if window is None:
        return None

    lines = [
        "Lote a verificar:",
        f"- Número: {lot.lot_number}",
        f"- Categoria: {lot.category}",
        f"- Sexo: {lot.sex}",
        f"- Quantidade: {lot.num_animals}",
    ]
    if lot.age_months is not None:
        lines.append(f"- Idade: {lot.age_months} meses")
    lines += [
        f"- Raça: {lot.breed}",
        f"- Preço por cabeça extraído: R$ {lot.unit_price:,.2f}",
        f"- Timestamp: {lot.timestamp_start}",
        "",
        f"Evidências da janela [{window.label}]:",
        "",
        window.combined_text,
    ]
    user_content = "\n".join(lines)

    try:
        response = client.complete(verify_prompt, user_content)
    except Exception as e:
        print(f"    verify lot {lot.lot_number}: LLM error ({e})")
        return None

    # Tolerate extra text around the JSON
    match = re.search(r"\{.*\}", response, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group())
    except json.JSONDecodeError:
        return None

    confirmed = data.get("confirmed") is True
    correct_price = data.get("correct_unit_price")

    if confirmed:
        return "confirm"
    if correct_price is None:
        return "discard"
    try:
        price = coerce_price_value(correct_price)
    except (TypeError, ValueError):
        return "discard"
    if price is None:
        return "discard"
    # If the LLM says confirmed=false but returns the same price, it's effectively
    # a confirm — collapse to the cleaner verdict so the log reads correctly.
    if lot.unit_price is not None and abs(price - lot.unit_price) < 0.01:
        return "confirm"
    return price


def _lot_has_window_support(lot_number: int, window: Window) -> bool:
    """Return whether the window text directly mentions this lot number."""
    text = window.combined_text
    num = str(lot_number)
    padded = num.zfill(2)
    patterns = [
        rf"\bLOTE\s*[:#-]?\s*0*{re.escape(num)}\b",
        rf"\bLOTE\s*\|\s*[^|\n]*\|\s*VALORPORANIMAL\s*\|\s*0*{re.escape(num)}\b",
        rf"\|\s*0*{re.escape(num)}\s*\|",
    ]
    if padded != num:
        patterns.append(rf"\bLOTE\s*[:#-]?\s*{re.escape(padded)}\b")
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)


def _parse_hhmmss(ts: str) -> int:
    """Parse 'HH:MM:SS' to integer seconds. Tolerates minor variation."""
    parts = ts.split(":")
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + int(s)
    raise ValueError(f"Invalid timestamp: {ts!r}")


def _parse_response(response: str) -> list[Lot]:
    """Parse JSON array from LLM response, tolerating extra text."""
    text = response.strip()

    try:
        data = json.loads(text)
        if isinstance(data, list):
            return _validate_lots(data)
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    # Try every array start so bracketed prose such as "Summary [done]" does
    # not hide a valid payload later in the response. raw_decode also handles
    # brackets inside JSON strings correctly.
    for match in re.finditer(r"\[", text):
        try:
            data, _ = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(data, list):
            return _validate_lots(data)

    raise ValueError("LLM response did not contain a valid JSON array")


# How far `unit_price * num_animals` can drift from `total_price` before we
# distrust total_price. 20% tolerates LLM rounding / display formatting drift
# but reliably catches cross-window stitching where num_animals came from one
# window and total_price from another (e.g. merged num=10 but total implies
# num=20 — rel_err = 0.5 exactly, caught by any tolerance < 0.5).
_PRICE_PRODUCT_TOLERANCE = 0.2

# Statistical outlier parameters — Tukey's 1.5·IQR fence.
# Bounds are `[Q1 − 1.5·IQR, Q3 + 1.5·IQR]` computed on the auction's own
# price distribution. Tukey fences are the classical boxplot outlier rule:
# quartiles are resistant to outliers (they don't move when extreme values
# are added), so the bounds remain meaningful even when hallucinations
# contaminate the input — up to the 25% breakdown point.
#
# Compared empirically on real data against mean ± kσ at 90/95/99.7% CI:
# mean/stdev get pulled up by the very hallucinations we're filtering,
# producing useless bounds (R$ 26k–R$ 42k upper). Tukey IQR gave a clean
# R$ 5,425 upper on the same 61-price sample — because the three R$ 55k
# outliers couldn't move Q3 beyond R$ 3,790.
_IQR_MULTIPLIER = 1.5      # classical Tukey fence
_MIN_SAMPLE_FOR_STATS = 5  # below this, bounds are disabled


def _compute_price_bounds(prices: list[float]) -> tuple[float, float]:
    """
    Per-head price bounds via Tukey's 1.5·IQR fence on observed prices.

    Returns (lower, upper). Below `_MIN_SAMPLE_FOR_STATS` observations,
    returns (0, +inf) — quartile-based bounds need enough data to be meaningful.
    """
    valid = [p for p in prices if p is not None and p > 0]
    if len(valid) < _MIN_SAMPLE_FOR_STATS:
        return (0.0, float("inf"))

    q1, _, q3 = statistics.quantiles(valid, n=4)
    iqr = q3 - q1
    lower = max(0.0, q1 - _IQR_MULTIPLIER * iqr)
    upper = q3 + _IQR_MULTIPLIER * iqr
    return (lower, upper)


def _sanity_check(lot: Lot, bounds: tuple[float, float] | None = None) -> Lot:
    """
    Enforce invariants on a lot's fields. Bad values are nulled out rather
    than propagated — null is recoverable from later windows, but a corrupt
    number poisons the downstream merge permanently.

    Two kinds of checks:
      - **Shape invariants** (always applied): `unit ≤ total`, `unit × n ≈ total`.
        These are physical relationships that must hold regardless of scale.
      - **Statistical outlier check** (only when `bounds` supplied): reject
        unit_price values outside the auction's own distribution. Per-window
        calls don't pass bounds because the distribution isn't known yet;
        bounds are applied post-merge once all prices have been observed.

    Checks are ordered so that the least-destructive action wins. When the
    `unit × n ≈ total` relationship is violated, we trust unit_price (the
    hammer price is the primary auction datum) and clear total_price only.
    """
    data = lot.model_dump()
    u = data.get("unit_price")
    t = data.get("total_price")
    n = data.get("num_animals")

    # 1. Statistical outlier bounds on unit_price — only when bounds provided
    #    (post-merge). Unreasonable values vs the auction's own distribution
    #    are cleared. No category-specific logic needed: a touro at the top
    #    of the auction's distribution naturally gets more slack than a
    #    bezerro because the distribution includes both.
    if bounds is not None and u is not None:
        lo, hi = bounds
        if u < lo or u > hi:
            data["unit_price"] = None
            u = None

    # 2. Product-consistency: trust unit_price, clear mismatched total.
    #    Example (lot 10 failure): unit=55000, total=2920, n=50 →
    #    expected=2.75M, rel_err ≈ 1.0 → clear total.
    if u is not None and t is not None and n and n > 0:
        expected = u * n
        denom = max(expected, t)
        if denom > 0:
            rel_err = abs(expected - t) / denom
            if rel_err > _PRICE_PRODUCT_TOLERANCE:
                data["total_price"] = None
                t = None

    # 3. Fallback inversion: unit > total is impossible. Only fires when
    #    num_animals is missing so the product check couldn't run.
    if u is not None and t is not None and u > t:
        data["unit_price"] = None
        data["total_price"] = None
        u = t = None

    # 4. If only total_price survived, cross-check its implied per-head
    #    against the statistical bounds (if available). Catches the case
    #    where unit was nulled in step 1 but total carries the same garbage
    #    scaled by num_animals.
    if bounds is not None and u is None and t is not None and n and n > 0:
        lo, hi = bounds
        implied = t / n
        if implied < lo or implied > hi:
            data["total_price"] = None

    return Lot(**data)


def _validate_lots(data: list[dict]) -> list[Lot]:
    """
    Parse and shape-check lots from a single LLM response. Statistical bounds
    are NOT applied here — we need all windows' prices to compute them.
    """
    lots = []
    for item in data:
        try:
            lot = Lot(**item)
        except Exception:
            continue
        lots.append(_sanity_check(lot))  # no bounds → shape invariants only
    return lots


def _merge(lots_by_number: dict[int, Lot], new_lot: Lot) -> None:
    if new_lot.lot_number not in lots_by_number:
        lots_by_number[new_lot.lot_number] = new_lot
        return

    existing = lots_by_number[new_lot.lot_number]
    existing_data = existing.model_dump()
    new_data = new_lot.model_dump()

    # Prices use last-non-null: later windows carry the final hammer price
    price_fields = {"unit_price", "total_price"}
    # sold=True is a final determination and always wins
    sold_existing = existing_data["sold"]
    sold_new = new_data["sold"]
    if sold_existing is True or sold_new is True:
        merged_sold = True
    elif sold_existing is False and sold_new is None:
        merged_sold = False  # preserve explicit not-sold over unknown
    else:
        merged_sold = sold_new if sold_new is not None else sold_existing

    merged = {}
    for k in existing_data:
        if k == "sold":
            merged[k] = merged_sold
        elif k in price_fields:
            # last-non-null wins for prices (final hammer beats opening ask)
            merged[k] = new_data[k] if new_data[k] is not None else existing_data[k]
        else:
            merged[k] = existing_data[k] if existing_data[k] is not None else new_data[k]

    lots_by_number[new_lot.lot_number] = Lot(**merged)


def _save(lots: list[Lot], path: Path) -> None:
    data = [lot.model_dump() for lot in lots]
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _load(path: Path) -> list[Lot]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [Lot(**d) for d in data]


def extract_metadata(
    windows: list[Window],
    client: LLMClient,
    prompt_path: Path,
    output_path: Path,
    video_info: dict | None = None,
) -> dict:
    """Extract auction-level metadata (date, city, auctioneer, etc.) from the first windows."""
    if output_path.exists():
        try:
            cached = json.loads(output_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            cached = None
        if isinstance(cached, dict):
            print(f"  Auction metadata already extracted, loading from cache.")
            return cached

    system_prompt = prompt_path.read_text(encoding="utf-8")

    # Prepend video title/description as the most reliable source for city and event name
    header = ""
    if video_info:
        parts = []
        if video_info.get("title"):
            parts.append(f"Título do vídeo: {video_info['title']}")
        if video_info.get("description"):
            parts.append(f"Descrição do vídeo:\n{video_info['description']}")
        if parts:
            header = "\n".join(parts) + "\n\n---\n\n"

    # Combine first N windows for metadata context
    combined = header + "\n\n".join(
        f"[{w.label}]\n{w.combined_text}" for w in windows[:_METADATA_WINDOWS]
    )

    checkpointable = False
    try:
        response = client.complete(system_prompt, combined)
        text = response.strip()
        decoder = json.JSONDecoder()
        metadata = None
        # Try every object start so unrelated braces in a preamble do not make
        # an otherwise valid metadata object unparsable.
        for match in re.finditer(r"\{", text):
            try:
                candidate, _ = decoder.raw_decode(text[match.start():])
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict):
                metadata = candidate
                checkpointable = True
                break
        if metadata is None:
            metadata = {}
    except Exception as e:
        print(f"  WARNING: Metadata extraction failed: {e}")
        metadata = {}

    if checkpointable:
        output_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def default_model(provider: str) -> str:
    if provider not in _DEFAULT_MODELS:
        raise ValueError(
            f"Unknown provider: {provider!r}. Use 'openrouter' or 'openai'."
        )
    return _DEFAULT_MODELS[provider]

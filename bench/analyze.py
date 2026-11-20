"""Benchmark analysis for current and historical models vs the Claude reference.

Produces three sections:
  1. Reference-based accuracy — lot coverage, category/sex/qty match,
     and price MAPE (against the manually-produced Claude references).
  2. Raw extraction stats — lot counts and wall-clock time per video.
  3. Token usage + cost — applies published $/M-token prices.
"""
import argparse
import json
import math
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "bench_results"

VIDEOS = ["TTNfx03Ubr8", "VDzRLEMUagA", "ZCwxnhUZjQM", "sKmUHExf464", "vWs74eYdyXo"]

MODELS = [
    # (display_name, directory_name, provider, input_price/M, output_price/M)
    ("GPT-6 Luna (low)", "gpt-6-luna__low", "openai", 0.10, 0.50),
    ("GPT-6 Luna (medium)", "gpt-6-luna__medium", "openai", 0.10, 0.50),
    ("gemini-2.5-flash-lite", "google_gemini-2.5-flash-lite", "openrouter", 0.10, 0.40),
    ("gemini-2.5-flash-lite-preview",
     "google_gemini-2.5-flash-lite-preview-09-2025",
     "openrouter",
     0.10, 0.40),
    ("gpt-4.1-mini",
     "gpt-4.1-mini",
     "openai",
     0.40, 1.60),
]

# Videos that have a manually-produced Claude reference.
REFERENCE_VIDEOS = {"sKmUHExf464", "VDzRLEMUagA"}


def _is_valid_price(value: object) -> bool:
    """Return whether a benchmark price is safe to include in statistics."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        numeric = float(value)
    except OverflowError:
        return False
    return math.isfinite(numeric) and numeric > 0


def _valid_prices(lots: list[dict]) -> list[float]:
    return [
        float(price)
        for lot in lots
        if _is_valid_price(price := lot.get("unit_price"))
    ]


def _median_price(prices: list[float]) -> float:
    lower = statistics.median_low(prices)
    upper = statistics.median_high(prices)
    # Prices are positive: this midpoint cannot overflow for finite inputs.
    return lower + (upper - lower) / 2


def load_json(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _load_run(directory: Path) -> tuple[dict | None, list | None]:
    latest = load_json(directory / "latest.json")
    run_id = latest.get("run_id") if isinstance(latest, dict) else None
    if isinstance(run_id, str) and re.fullmatch(r"[a-f0-9]{64}", run_id):
        directory /= run_id
    elif (directory / "latest.json").exists():
        return None, None
    summary = load_json(directory / "summary.json")
    lots = load_json(directory / "lots.json")
    if run_id and (not isinstance(summary, dict) or summary.get("run_id") != run_id):
        return None, None
    return summary if isinstance(summary, dict) else None, lots if isinstance(lots, list) else None


def _model_by_label(ref_by_num: dict, model_lots: list[dict]) -> dict:
    predictions = {}
    used = set()
    for lot in model_lots:
        value = lot["lot_number"]
        if value in ref_by_num:
            predictions[value] = lot
            used.add(id(lot))

    remaining = set(ref_by_num) - predictions.keys()
    for lot in model_lots:
        if id(lot) in used:
            continue
        value = lot["lot_number"]
        digits = str(value)
        candidates = []
        if digits.isascii() and digits.isdecimal():
            candidates = [label for label in remaining if (type(label) is int or type(value) is int)
                          and str(label).isascii() and str(label).isdecimal()
                          and int(label) == int(digits)]
            if len(candidates) > 1:
                age = lot.get("age_months")
                candidates = [label for label in candidates
                              if age is not None and ref_by_num[label].get("age_months") == age]
        if len(candidates) == 1 and sum(
            str(row["lot_number"]).isascii() and str(row["lot_number"]).isdecimal()
            and int(row["lot_number"]) == int(digits) and id(row) not in used for row in model_lots
        ) == 1:
            label = candidates[0]
            remaining.remove(label)
        else:
            label = value if value not in ref_by_num else ("unmatched", len(predictions))
        predictions[label] = lot
    return predictions


def compare(ref: list[dict], model_lots: list[dict]) -> dict:
    ref_by_num = {lot["lot_number"]: lot for lot in ref}
    m_by_num = _model_by_label(ref_by_num, model_lots)

    matched = set(ref_by_num) & set(m_by_num)
    missing = set(ref_by_num) - set(m_by_num)
    extra = set(m_by_num) - set(ref_by_num)

    known_status = {n for n, lot in ref_by_num.items() if isinstance(lot.get("sold"), bool)}
    status_matches = sum(
        m_by_num[n].get("sold") is ref_by_num[n]["sold"] for n in matched & known_status
    )
    expected_prices = {
        n for n, lot in ref_by_num.items()
        if lot.get("sold") is not False and _is_valid_price(lot.get("unit_price"))
    }
    hammer_prices = {n for n in expected_prices if ref_by_num[n].get("sold") is True}
    comparable = set()
    errs = []
    exact_prices = 0
    for n in matched & expected_prices:
        candidate = m_by_num[n]
        rp, mp = ref_by_num[n]["unit_price"], candidate.get("unit_price")
        if (not _is_valid_price(mp) or candidate.get("price_needs_review")
                or candidate.get("sold") is False
                or (n in hammer_prices and candidate.get("sold") is not True)):
            continue
        comparable.add(n)
        e = abs(float(mp) - float(rp)) / float(rp)
        if math.isfinite(e):
            errs.append(e)
            exact_prices += rp == mp

    cat_ok = sum(1 for n in matched
                 if (m_by_num[n].get("category") or "").lower() ==
                    (ref_by_num[n].get("category") or "").lower())
    sex_ok = sum(1 for n in matched
                 if (m_by_num[n].get("sex") or "").lower() ==
                    (ref_by_num[n].get("sex") or "").lower())
    qty_ok = sum(1 for n in matched
                 if m_by_num[n].get("num_animals") == ref_by_num[n].get("num_animals"))

    return {
        "ref_n": len(ref_by_num),
        "model_n": len(model_lots),
        "matched": len(matched),
        "missing": len(missing),
        "extra": len(extra) + len(model_lots) - len(m_by_num),
        "precision": len(matched) / len(model_lots) if model_lots else 0,
        "coverage": len(matched) / len(ref_by_num) if ref_by_num else 0,
        "sold_status_accuracy": status_matches / len(known_status) if known_status else None,
        "sold_status_matches": status_matches,
        "sold_status_matched_known": len(matched & known_status),
        "sold_status_known": len(known_status),
        "missing_lot_numbers": sorted(str(n) for n in missing),
        "unmatched_model_lot_numbers": sorted(str(m_by_num[n]["lot_number"]) for n in extra),
        "price_count": len(comparable),
        "price_expected": len(expected_prices),
        "hammer_price_count": len(comparable & hammer_prices),
        "hammer_price_expected": len(hammer_prices),
        "mape_n": len(errs),
        "exact_prices": exact_prices,
        "category_match": cat_ok / len(matched) if matched else 0,
        "sex_match": sex_ok / len(matched) if matched else 0,
        "qty_match": qty_ok / len(matched) if matched else 0,
        "mape": statistics.mean(errs) if errs else None,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, help="Reference lots array or video reference object")
    parser.add_argument("--result", type=Path, help="Extracted lots array or pipeline result object")
    args = parser.parse_args(argv)
    if args.reference or args.result:
        if not args.reference or not args.result:
            parser.error("--reference and --result must be supplied together")
        reference, result = load_json(args.reference), load_json(args.result)
        if isinstance(reference, dict) and isinstance(result, dict):
            if reference.get("video_id") != result.get("video_id"):
                parser.error("reference and result video identifiers differ")
        ref_lots = reference.get("lots") if isinstance(reference, dict) else reference
        model_lots = result.get("lots") if isinstance(result, dict) else result
        if not isinstance(ref_lots, list) or not isinstance(model_lots, list):
            parser.error("both files must contain a readable lots array")
        print(json.dumps(compare(ref_lots, model_lots), ensure_ascii=False, indent=2))
        return

    # Load all summaries + lots
    runs = {}
    for v in VIDEOS:
        for (disp, dirname, provider, in_price, out_price) in MODELS:
            s, l = _load_run(RESULTS / v / dirname)
            runs[(v, dirname)] = (s, l, in_price, out_price)

    # ── Section 1: Reference-based accuracy ───────────────────────────
    print("=" * 120)
    print("SECTION 1 — Reference-based accuracy (vs manual Claude reference)")
    print("=" * 120)
    for v in sorted(REFERENCE_VIDEOS):
        ref = load_json(RESULTS / v / "REFERENCE_claude" / "lots.json")
        if not ref:
            continue
        print(f"\n  Video: {v}   (reference: {len(ref)} lots)")
        manifests = {
            json.dumps(summary["inputs"], sort_keys=True)
            for _, dirname, *_ in MODELS
            if (summary := runs[(v, dirname)][0])
            and summary.get("status") == "ok" and isinstance(summary.get("inputs"), dict)
        }
        if len(manifests) > 1:
            print("  Runs used different inputs or parser versions. Replay before comparing settings.")
        print(f"  {'Model':<44} {'N':>3} {'Miss':>4} {'Extra':>5} {'Prec':>5} {'Recall':>6} {'Status':>6} "
              f"{'Price n/N':>9} {'Hammer n/N':>10} {'Cat':>5} {'Sex':>5} {'Qty':>5} {'MAPE (n)':>13}")
        for (disp, dirname, *_rest) in MODELS:
            (s, lots, *_) = runs[(v, dirname)]
            if lots is None or not s or s.get("status") != "ok":
                print(f"  {disp[:44]:<44} {'—'}")
                continue
            c = compare(ref, lots)
            mape = f"{c['mape']:.1%} ({c['mape_n']})" if c['mape'] is not None else "—"
            status = f"{c['sold_status_accuracy']:.0%}" if c["sold_status_accuracy"] is not None else "—"
            price_coverage = f"{c['price_count']}/{c['price_expected']}"
            hammer_coverage = f"{c['hammer_price_count']}/{c['hammer_price_expected']}"
            print(f"  {disp[:44]:<44} "
                  f"{c['model_n']:>3} {c['missing']:>4} {c['extra']:>5} "
                  f"{c['precision']:>5.0%} {c['coverage']:>6.0%} {status:>6} "
                  f"{price_coverage:>9} {hammer_coverage:>10} {c['category_match']:>5.0%} "
                  f"{c['sex_match']:>5.0%} {c['qty_match']:>5.0%} {mape:>13}")

    # ── Section 2: Raw extraction stats ───────────────────────────────
    print("\n\n" + "=" * 120)
    print("SECTION 2 — Raw extraction stats (all 5 videos)")
    print("=" * 120)
    print(f"\n  {'Model':<44} " + " ".join(f"{v[:12]:>13}" for v in VIDEOS) + f"{'Total':>13}")

    # Row 0 — Claude reference (manual extraction by me, in-context)
    ref_row = [f"  {'Claude Opus (reference, manual)':<44}"]
    ref_total_lots = 0
    for v in VIDEOS:
        ref = load_json(RESULTS / v / "REFERENCE_claude" / "lots.json")
        if ref is None:
            ref_row.append(f"{'not-produced':>13}")
        else:
            ref_total_lots += len(ref)
            ref_row.append(f"{str(len(ref)) + 'L':>13}")
    ref_row.append(f"{str(ref_total_lots) + 'L':>13}")
    print(" ".join(ref_row))

    for (disp, dirname, *_rest) in MODELS:
        row = [f"  {disp[:44]:<44}"]
        total_lots, total_time = 0, 0.0
        for v in VIDEOS:
            (s, lots, *_) = runs[(v, dirname)]
            if not s or s.get("status") != "ok":
                row.append(f"{'—':>13}")
                continue
            total_lots += s["lot_count"]
            total_time += s["elapsed_seconds"]
            row.append(f"{str(s['lot_count']) + 'L/' + str(int(s['elapsed_seconds'])) + 's':>13}")
        row.append(f"{str(total_lots) + 'L/' + str(int(total_time)) + 's':>13}")
        print(" ".join(row))

    # ── Section 2b: Price distribution comparison (reference videos) ──
    print("\n  Price distribution for videos with Claude reference:")
    print(f"  {'Source':<44} {'Video':<14} {'n':>3} {'Sold':>4} {'Mean':>9} {'Median':>9} {'Min':>9} {'Max':>9}")
    for v in sorted(REFERENCE_VIDEOS):
        ref = load_json(RESULTS / v / "REFERENCE_claude" / "lots.json")
        if ref:
            prices = _valid_prices(ref)
            sold = sum(1 for l in ref if l.get("sold") is True)
            if prices:
                print(f"  {'Claude Opus (reference)':<44} {v:<14} {len(ref):>3} {sold:>4} "
                      f"R${statistics.mean(prices):>7,.0f} R${_median_price(prices):>7,.0f} "
                      f"R${min(prices):>7,.0f} R${max(prices):>7,.0f}")
        for (disp, dirname, *_rest) in MODELS:
            (s, lots, *_) = runs[(v, dirname)]
            if not lots:
                continue
            prices = _valid_prices(lots)
            sold = sum(1 for l in lots if l.get("sold") is True)
            if prices:
                print(f"  {disp[:44]:<44} {v:<14} {len(lots):>3} {sold:>4} "
                      f"R${statistics.mean(prices):>7,.0f} R${_median_price(prices):>7,.0f} "
                      f"R${min(prices):>7,.0f} R${max(prices):>7,.0f}")

    # ── Section 3: Token usage and cost ───────────────────────────────
    print("\n\n" + "=" * 120)
    print("SECTION 3 — Token usage + estimated cost")
    print("=" * 120)
    for (disp, dirname, provider, in_price, out_price) in MODELS:
        print(f"\n  {disp}  [{provider}, ${in_price}/M in · ${out_price}/M out]")
        print(f"    {'Video':<14} {'Calls':>6} {'Input tok':>11} {'Output tok':>11} {'Cost':>9}")
        total_in, total_out, total_cost, completed = 0, 0, 0.0, 0
        for v in VIDEOS:
            (s, lots, *_) = runs[(v, dirname)]
            if not s or s.get("status") != "ok":
                print(f"    {v[:14]:<14}   —")
                continue
            in_t = s.get("input_tokens", 0)
            out_t = s.get("output_tokens", 0)
            cost = (in_t / 1_000_000) * in_price + (out_t / 1_000_000) * out_price
            total_in += in_t
            total_out += out_t
            total_cost += cost
            completed += 1
            print(f"    {v[:14]:<14} {s.get('llm_calls', '—'):>6} {in_t:>11,} {out_t:>11,} ${cost:>7.4f}")
        print(f"    {'─'*60}")
        print(f"    {'TOTAL':<14} {'':>6} {total_in:>11,} {total_out:>11,} ${total_cost:>7.4f}")
        if completed:
            print(f"    {'AVG/video':<14} {'':>6} {total_in//completed:>11,} "
                  f"{total_out//completed:>11,} ${total_cost/completed:>7.4f}")


if __name__ == "__main__":
    main()

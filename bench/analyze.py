"""Benchmark analysis for the two shipping models vs the Claude reference.

Produces three sections:
  1. Reference-based accuracy — lot coverage, category/sex/qty match,
     and price MAPE (against the manually-produced Claude references).
  2. Raw extraction stats — lot counts and wall-clock time per video.
  3. Token usage + cost — applies published $/M-token prices.
"""
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "bench_results"

VIDEOS = ["TTNfx03Ubr8", "VDzRLEMUagA", "ZCwxnhUZjQM", "sKmUHExf464", "vWs74eYdyXo"]

MODELS = [
    # (display_name, directory_name, provider, input_price/M, output_price/M)
    ("gemini-2.5-flash-lite-preview (default)",
     "google_gemini-2.5-flash-lite-preview-09-2025",
     "openrouter",
     0.10, 0.40),
    ("gpt-4.1-mini (alternative)",
     "gpt-4.1-mini",
     "openai",
     0.40, 1.60),
]

# Videos that have a manually-produced Claude reference.
REFERENCE_VIDEOS = {"sKmUHExf464", "VDzRLEMUagA"}


def load_json(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def compare(ref: list[dict], model_lots: list[dict]) -> dict:
    ref_by_num = {l["lot_number"]: l for l in ref}
    m_by_num = {l["lot_number"]: l for l in model_lots}

    matched = set(ref_by_num) & set(m_by_num)
    missing = set(ref_by_num) - set(m_by_num)
    extra = set(m_by_num) - set(ref_by_num)

    # MAPE on matched lots with both prices set
    errs, price_within_5pct = [], 0
    for n in matched:
        rp, mp = ref_by_num[n].get("unit_price"), m_by_num[n].get("unit_price")
        if rp and mp:
            e = abs(mp - rp) / rp
            errs.append(e)
            if e < 0.05:
                price_within_5pct += 1

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
        "model_n": len(m_by_num),
        "matched": len(matched),
        "missing": len(missing),
        "extra": len(extra),
        "coverage": len(matched) / len(ref_by_num) if ref_by_num else 0,
        "category_match": cat_ok / len(matched) if matched else 0,
        "sex_match": sex_ok / len(matched) if matched else 0,
        "qty_match": qty_ok / len(matched) if matched else 0,
        "mape": statistics.mean(errs) if errs else None,
    }


def main():
    # Load all summaries + lots
    runs = {}
    for v in VIDEOS:
        for (disp, dirname, provider, in_price, out_price) in MODELS:
            s = load_json(RESULTS / v / dirname / "summary.json")
            l = load_json(RESULTS / v / dirname / "lots.json")
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
        print(f"  {'Model':<44} {'N':>3} {'Match':>5} {'Miss':>4} {'Extra':>5} {'Cov':>5} {'Cat':>5} {'Sex':>5} {'Qty':>5} {'MAPE':>7}")
        print(f"  {'-'*44} {'-'*3} {'-'*5} {'-'*4} {'-'*5} {'-'*5} {'-'*5} {'-'*5} {'-'*5} {'-'*7}")
        for (disp, dirname, *_rest) in MODELS:
            (s, lots, *_) = runs[(v, dirname)]
            if not lots or s.get("status") != "ok":
                print(f"  {disp[:44]:<44} {'—'}")
                continue
            c = compare(ref, lots)
            mape = f"{c['mape']*100:.1f}%" if c['mape'] is not None else "   —"
            print(f"  {disp[:44]:<44} "
                  f"{c['model_n']:>3} {c['matched']:>5} {c['missing']:>4} {c['extra']:>5} "
                  f"{c['coverage']:>4.0%} {c['category_match']:>4.0%} "
                  f"{c['sex_match']:>4.0%} {c['qty_match']:>4.0%} {mape:>7}")

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
            prices = [l["unit_price"] for l in ref if l.get("unit_price")]
            sold = sum(1 for l in ref if l.get("sold") is True)
            print(f"  {'Claude Opus (reference)':<44} {v:<14} {len(ref):>3} {sold:>4} "
                  f"R${statistics.mean(prices):>7,.0f} R${statistics.median(prices):>7,.0f} "
                  f"R${min(prices):>7,.0f} R${max(prices):>7,.0f}")
        for (disp, dirname, *_rest) in MODELS:
            (s, lots, *_) = runs[(v, dirname)]
            if not lots:
                continue
            prices = [l.get("unit_price") for l in lots if l.get("unit_price")]
            sold = sum(1 for l in lots if l.get("sold") is True)
            if prices:
                print(f"  {disp[:44]:<44} {v:<14} {len(lots):>3} {sold:>4} "
                      f"R${statistics.mean(prices):>7,.0f} R${statistics.median(prices):>7,.0f} "
                      f"R${min(prices):>7,.0f} R${max(prices):>7,.0f}")

    # ── Section 3: Token usage and cost ───────────────────────────────
    print("\n\n" + "=" * 120)
    print("SECTION 3 — Token usage + estimated cost")
    print("=" * 120)
    for (disp, dirname, provider, in_price, out_price) in MODELS:
        print(f"\n  {disp}  [{provider}, ${in_price}/M in · ${out_price}/M out]")
        print(f"    {'Video':<14} {'Calls':>6} {'Input tok':>11} {'Output tok':>11} {'Cost':>9}")
        total_in, total_out, total_cost = 0, 0, 0.0
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
            print(f"    {v[:14]:<14} {s.get('llm_calls', '—'):>6} {in_t:>11,} {out_t:>11,} ${cost:>7.4f}")
        avg = total_cost / len(VIDEOS)
        print(f"    {'─'*60}")
        print(f"    {'TOTAL':<14} {'':>6} {total_in:>11,} {total_out:>11,} ${total_cost:>7.4f}")
        print(f"    {'AVG/video':<14} {'':>6} {total_in//len(VIDEOS):>11,} {total_out//len(VIDEOS):>11,} ${avg:>7.4f}")


if __name__ == "__main__":
    main()

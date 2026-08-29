import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import click
from dotenv import load_dotenv

load_dotenv()

from rich.console import Console
from rich.table import Table

from models.lot import AuctionResult
from pipeline import aggregator, costs, downloader, extractor, ocr, screenshotter
from pipeline import transcriber as transcriber_mod
from pipeline.transcript_quality import check_transcript

console = Console()

PROMPTS_DIR = Path(__file__).parent / "prompts"


@click.command()
@click.argument("urls", nargs=-1)
@click.option(
    "--batch-file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Text file with one YouTube URL per line. Blank lines and lines starting with # are ignored.",
)
@click.option(
    "--batch-name",
    default=None,
    help="Name for the saved batch report folder. Defaults to a timestamp.",
)
@click.option(
    "--stop-on-error",
    is_flag=True,
    default=False,
    help="In batch mode, stop after the first failed URL instead of continuing.",
)
@click.option(
    "--provider",
    type=click.Choice(["openrouter", "openai"]),
    default="openrouter",
    show_default=True,
    help=(
        "LLM provider for lot extraction. "
        "'openrouter' → Gemini 2.5 Flash-Lite Preview (default, cheapest/fastest). "
        "'openai' → GPT-4.1 Mini (alternative, highest accuracy)."
    ),
)
@click.option(
    "--output-dir",
    default="output",
    show_default=True,
    help="Base directory for all outputs.",
)
@click.option(
    "--transcriber",
    type=click.Choice(["mlx", "cpp", "groq"]),
    default="groq",
    show_default=True,
    help=(
        "Transcription backend. "
        "mlx: Apple Silicon via Metal (requires --extra local). "
        "cpp: whisper.cpp binary (requires brew install whisper-cpp). "
        "groq: Groq cloud API, Whisper Large v3 Turbo (requires GROQ_API_KEY)."
    ),
)
@click.option(
    "--whisper-model",
    default="medium",
    show_default=True,
    help="Whisper model size for mlx/cpp backends (tiny/base/small/medium/large-v3).",
)
@click.option(
    "--cpp-model",
    default=None,
    type=click.Path(exists=True, path_type=Path),
    help="Path to ggml model file for whisper.cpp (auto-detected if omitted).",
)
@click.option(
    "--screenshot-interval",
    type=click.IntRange(min=1),
    default=30,
    show_default=True,
    help="Seconds between screenshots.",
)
@click.option(
    "--frame-sampling",
    type=click.Choice(["interval", "scene"]),
    default="interval",
    show_default=True,
    help=(
        "How to pick screenshots. 'interval' samples every --screenshot-interval "
        "seconds. 'scene' samples where the lot board actually changes, unioned "
        "with a coarse safety grid — usually fewer frames and closer to lot "
        "boundaries, but not yet calibrated on real auction footage."
    ),
)
@click.option(
    "--safety-interval",
    type=click.IntRange(min=1),
    default=60,
    show_default=True,
    help="Seconds between safety-grid frames added on top of detected scene changes.",
)
@click.option(
    "--ocr-video-height",
    type=click.Choice(["480", "720"]),
    default="480",
    show_default=True,
    help="Maximum video height to download for OCR screenshots.",
)
@click.option(
    "--no-resume",
    is_flag=True,
    default=False,
    help="Ignore cached stage outputs and rerun everything.",
)
@click.option(
    "--metadata/--no-metadata",
    "show_metadata",
    default=True,
    help="Display auction metadata (date, city, auctioneer, farm, type).",
)
@click.option(
    "--summary/--no-summary",
    "show_summary",
    default=True,
    help="Display summary statistics (totals, averages, counts by category).",
)
@click.option(
    "--table/--no-table",
    "show_table",
    default=True,
    help="Display full table of all lots with detailed information.",
)
def cli(
    urls,
    batch_file,
    batch_name,
    stop_on_error,
    provider,
    output_dir,
    transcriber,
    whisper_model,
    cpp_model,
    screenshot_interval,
    frame_sampling,
    safety_interval,
    ocr_video_height,
    no_resume,
    show_metadata,
    show_summary,
    show_table,
):
    """Extract lot data from one or more Brazilian cattle auction YouTube videos."""
    batch_urls = _load_batch_urls(urls, batch_file)
    if not batch_urls:
        raise click.UsageError("Provide at least one YouTube URL or use --batch-file.")

    is_batch = batch_file is not None or len(batch_urls) > 1
    if is_batch:
        report = _run_batch(
            batch_urls,
            provider=provider,
            output_dir=output_dir,
            transcriber=transcriber,
            whisper_model=whisper_model,
            cpp_model=cpp_model,
            screenshot_interval=screenshot_interval,
            frame_sampling=frame_sampling,
            safety_interval=safety_interval,
            ocr_video_height=int(ocr_video_height),
            no_resume=no_resume,
            show_metadata=show_metadata,
            show_summary=show_summary,
            show_table=show_table,
            batch_name=batch_name,
            stop_on_error=stop_on_error,
        )
        if report["totals"]["failed"]:
            raise click.ClickException(
                f"{report['totals']['failed']} batch item(s) failed. See batch report for details."
            )
        return

    _run_single_url(
        batch_urls[0],
        provider=provider,
        output_dir=output_dir,
        transcriber=transcriber,
        whisper_model=whisper_model,
        cpp_model=cpp_model,
        screenshot_interval=screenshot_interval,
        frame_sampling=frame_sampling,
        safety_interval=safety_interval,
        ocr_video_height=int(ocr_video_height),
        no_resume=no_resume,
        show_metadata=show_metadata,
        show_summary=show_summary,
        show_table=show_table,
    )


def _run_single_url(
    url,
    *,
    provider,
    output_dir,
    transcriber,
    whisper_model,
    cpp_model,
    screenshot_interval,
    ocr_video_height,
    no_resume,
    show_metadata,
    show_summary,
    show_table,
    frame_sampling="interval",
    safety_interval=60,
):
    model = extractor.default_model(provider)

    console.rule("[bold]Cattle Auction Extractor")
    console.print(f"  URL:         {url}")
    console.print(f"  Transcriber: {transcriber}" + (f" / {whisper_model}" if transcriber != "groq" else " (Whisper Large v3 Turbo)"))
    console.print(f"  Extractor:   {provider} / {model}")
    console.print(f"  OCR video:   {ocr_video_height}p")
    console.print()

    video_id = downloader.get_video_id(url)
    run_dir = Path(output_dir) / video_id
    run_dir.mkdir(parents=True, exist_ok=True)

    if no_resume:
        _clear_cache(run_dir, video_id)

    video_info = downloader.get_video_info(url, run_dir, video_id)

    # ── Stage 1: Download audio ──────────────────────────────────────────
    _stage("1/7", "Download audio")
    t0 = time.time()
    audio_path = downloader.download_audio(url, run_dir, video_id)
    _done(t0)

    # ── Stage 2: Transcribe ──────────────────────────────────────────────
    _stage("2/7", f"Transcribe audio ({transcriber})")
    t0 = time.time()
    # Checked before the call: a transcript served from checkpoint costs
    # nothing, so billing it again would overstate a resumed run's spend.
    transcript_path = run_dir / f"transcript_{video_id}.json"
    transcript_was_cached = transcript_path.exists()
    segments = transcriber_mod.transcribe(
        audio_path,
        output_path=transcript_path,
        backend=transcriber,
        whisper_model=whisper_model,
        cpp_model_path=cpp_model,
    )
    # Gate the transcript on the way out rather than on the way in: the
    # checkpoint keeps Whisper's raw output, so tightening these heuristics
    # later does not require re-transcribing anything.
    audio_seconds = video_info.get("duration")
    quality = check_transcript(segments, audio_duration=audio_seconds)
    segments = quality.segments
    if quality.status != "ok":
        console.print(f"  [yellow]Transcript {quality.status}:[/yellow] {quality.warning}")
    _done(t0)

    # ── Stage 3: Download OCR video ──────────────────────────────────────
    _stage("3/7", f"Download OCR video ({ocr_video_height}p)")
    t0 = time.time()
    video_path = downloader.download_ocr_video(
        url,
        run_dir,
        video_id,
        height=ocr_video_height,
    )
    _done(t0)

    # ── Stage 4: Screenshots ─────────────────────────────────────────────
    if frame_sampling == "scene":
        _stage("4/7", f"Extract screenshots (scene changes + {safety_interval}s grid)")
    else:
        _stage("4/7", f"Extract screenshots (every {screenshot_interval}s)")
    t0 = time.time()
    shots = screenshotter.extract_screenshots(
        video_path,
        output_dir=run_dir,
        video_id=video_id,
        interval=screenshot_interval,
        sampling=frame_sampling,
        safety_interval=safety_interval,
    )
    _done(t0)

    # ── Stage 5: OCR ─────────────────────────────────────────────────────
    _stage("5/7", "Run OCR on screenshots")
    t0 = time.time()
    ocr_results = ocr.run_ocr(shots, output_path=run_dir / f"ocr_results_{video_id}.json")
    _done(t0)

    # ── Stage 6: Extract lots ─────────────────────────────────────────────
    _stage("6/7", f"Extract lots with {provider}/{model}")
    t0 = time.time()
    windows = aggregator.aggregate(segments, ocr_results)
    console.print(f"  Aggregated into {len(windows)} windows.")

    client = extractor.LLMClient(provider=provider, model=model)
    lots = extractor.extract_lots(
        windows,
        client=client,
        prompt_path=PROMPTS_DIR / "extraction.txt",
        output_path=run_dir / f"lots_{video_id}.json",
        verify_prompt_path=PROMPTS_DIR / "verify.txt",
    )
    _done(t0)

    # ── Stage 7: Extract auction metadata ────────────────────────────────
    _stage("7/7", f"Extract auction metadata with {provider}/{model}")
    t0 = time.time()
    metadata = extractor.extract_metadata(
        windows,
        client=client,
        prompt_path=PROMPTS_DIR / "metadata.txt",
        output_path=run_dir / f"metadata_{video_id}.json",
        video_info=video_info,
    )
    _done(t0)

    # ── Summary ───────────────────────────────────────────────────────────
    run_cost = costs.estimate_cost(
        model,
        client.input_tokens,
        client.output_tokens,
        transcriber=transcriber,
        audio_seconds=0.0 if transcript_was_cached else (audio_seconds or 0.0),
    )

    result = AuctionResult(
        video_url=url,
        video_id=video_id,
        date=metadata.get("date"),
        city=metadata.get("city"),
        auctioneer=metadata.get("auctioneer"),
        farm=metadata.get("farm"),
        auction_type=metadata.get("auction_type"),
        notes=metadata.get("notes"),
        total_lots=len(lots),
        lots=lots,
        cost_usd=run_cost["total_usd"],
    )
    summary_path = run_dir / f"result_{video_id}.json"
    summary_path.write_text(
        json.dumps(result.model_dump(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    console.rule("[bold green]Done")
    if show_metadata and (metadata.get("date") or metadata.get("city") or metadata.get("auctioneer")):
        console.print(f"  Data:       {metadata.get('date') or '-'}")
        console.print(f"  Cidade:     {metadata.get('city') or '-'}")
        console.print(f"  Leiloeiro:  {metadata.get('auctioneer') or '-'}")
        if metadata.get("farm"):
            console.print(f"  Fazenda:    {metadata.get('farm')}")
        if metadata.get("auction_type"):
            console.print(f"  Tipo:       {metadata.get('auction_type')}")
        if metadata.get("notes"):
            console.print(f"  Obs:        {metadata.get('notes')}")
        console.print()
    console.print(f"  Found [bold]{len(lots)}[/bold] lots.")
    console.print(
        f"  Cost:   {costs.format_cost(run_cost['total_usd'])}"
        f"  (LLM {costs.format_cost(run_cost['llm_usd'])}"
        f"  |  transcription {costs.format_cost(run_cost['transcription_usd'])})"
    )
    console.print(f"  Output: [cyan]{run_dir}/[/cyan]")
    console.print()

    if show_summary:
        summary_stats = _calculate_summary(lots)
        _print_summary(summary_stats)

    if show_table:
        _print_table(lots)

    return result


def _load_batch_urls(urls, batch_file: Path | None) -> list[str]:
    """Load positional URLs plus an optional text file of URLs."""
    loaded = [url.strip() for url in urls if url.strip()]
    if batch_file is None:
        return loaded

    for line in batch_file.read_text(encoding="utf-8").splitlines():
        url = line.strip()
        if not url or url.startswith("#"):
            continue
        loaded.append(url)

    return loaded


def _run_batch(
    urls,
    *,
    provider,
    output_dir,
    transcriber,
    whisper_model,
    cpp_model,
    screenshot_interval,
    ocr_video_height,
    no_resume,
    show_metadata,
    show_summary,
    show_table,
    batch_name,
    stop_on_error,
    frame_sampling="interval",
    safety_interval=60,
):
    output_root = Path(output_dir)
    report_dir = _batch_report_dir(output_root, batch_name)
    report_name = report_dir.name

    console.rule("[bold]Cattle Auction Batch")
    console.print(f"  Videos:      {len(urls)}")
    console.print(f"  Batch:       {report_name}")
    console.print(f"  Output:      [cyan]{output_root}/[/cyan]")
    console.print()

    items = []
    for index, url in enumerate(urls, start=1):
        console.rule(f"[bold cyan]Batch {index}/{len(urls)}")
        console.print(f"  URL: {url}")
        try:
            result = _run_single_url(
                url,
                provider=provider,
                output_dir=output_dir,
                transcriber=transcriber,
                whisper_model=whisper_model,
                cpp_model=cpp_model,
                screenshot_interval=screenshot_interval,
                frame_sampling=frame_sampling,
                safety_interval=safety_interval,
                ocr_video_height=ocr_video_height,
                no_resume=no_resume,
                show_metadata=show_metadata,
                show_summary=show_summary,
                show_table=show_table,
            )
        except Exception as exc:
            console.print(f"  [red]Failed:[/red] {exc}\n")
            items.append(_batch_failure_item(index, url, exc))
            if stop_on_error:
                break
        else:
            items.append(_batch_success_item(index, result, output_root))

    report = _build_batch_report(report_name, items)
    paths = _write_batch_report(report, report_dir)
    _print_batch_summary(report, paths["json"])
    return report


def _batch_report_dir(output_root: Path, batch_name: str | None) -> Path:
    name = batch_name or datetime.now().strftime("batch_%Y%m%d_%H%M%S")
    safe_name = "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in name).strip(".-")
    return output_root / "batches" / (safe_name or "batch")


def _batch_success_item(index: int, result: AuctionResult, output_root: Path) -> dict:
    summary = _calculate_summary(result.lots)
    category_animals = summary.get("category_animals", {})
    top_category = next(iter(category_animals), None)
    return {
        "index": index,
        "status": "success",
        "url": result.video_url,
        "video_id": result.video_id,
        "output_dir": str(output_root / result.video_id),
        "date": result.date,
        "city": result.city,
        "auctioneer": result.auctioneer,
        "farm": result.farm,
        "auction_type": result.auction_type,
        "notes": result.notes,
        "auction_name": _infer_auction_name(result.auctioneer, result.farm, result.notes),
        "auctioneer_display": _infer_auctioneer(result.auctioneer, result.notes),
        "total_lots": summary.get("total_lots", 0),
        "total_animals": summary.get("total_animals", 0),
        "sold": summary.get("sold", 0),
        "not_sold": summary.get("not_sold", 0),
        "avg_price": summary.get("avg_price", 0),
        "cost_usd": result.cost_usd,
        "top_category": top_category,
        "category_animals": category_animals,
        "category_prices": summary.get("category_prices", {}),
    }


def _batch_failure_item(index: int, url: str, exc: Exception) -> dict:
    return {
        "index": index,
        "status": "failed",
        "url": url,
        "error": str(exc),
    }


def _infer_auction_name(auctioneer: str | None, farm: str | None, notes: str | None) -> str | None:
    if farm:
        return farm
    if auctioneer and "leil" in auctioneer.lower():
        return auctioneer
    if notes:
        quoted = re.search(r"(?:canal do YouTube|anunciado como)\s+'([^']+)'", notes, flags=re.IGNORECASE)
        if quoted:
            return quoted.group(1)
        leilao = re.search(r"\b(Leilão\s+\d+º)", notes, flags=re.IGNORECASE)
        if leilao:
            return leilao.group(1)
        cal = re.search(r"\bCAL\s+LEIL[ÕO]ES\b", notes, flags=re.IGNORECASE)
        if cal:
            return "CAL LEILÕES"
    return auctioneer


def _infer_auctioneer(auctioneer: str | None, notes: str | None) -> str | None:
    if notes:
        mentioned = re.search(
            r"[Oo]?\s*[Ll]eiloeiro(?:\s+mencionado(?:\s+no\s+áudio)?)?\s*(?:é|:)\s*([^\.]+)",
            notes,
        )
        if mentioned:
            return mentioned.group(1).strip()
    return auctioneer


def _build_batch_report(batch_name: str, items: list[dict]) -> dict:
    successes = [item for item in items if item["status"] == "success"]
    failures = [item for item in items if item["status"] == "failed"]

    category_totals = {}
    for item in successes:
        for category, count in item.get("category_animals", {}).items():
            category_totals[category] = category_totals.get(category, 0) + count
    top_price_categories = [
        category
        for category, _ in sorted(
            category_totals.items(),
            key=lambda row: row[1],
            reverse=True,
        )[:3]
    ]

    comparison = {
        "most_lots": _comparison_item(successes, "total_lots"),
        "most_animals": _comparison_item(successes, "total_animals"),
        "highest_avg_price": _comparison_item(
            [item for item in successes if item.get("avg_price")],
            "avg_price",
        ),
    }

    return {
        "batch": batch_name,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "totals": {
            "videos": len(items),
            "successful": len(successes),
            "failed": len(failures),
            "lots": sum(item.get("total_lots", 0) for item in successes),
            "animals": sum(item.get("total_animals", 0) for item in successes),
            "sold": sum(item.get("sold", 0) for item in successes),
            "not_sold": sum(item.get("not_sold", 0) for item in successes),
            "cost_usd": round(sum(item.get("cost_usd") or 0.0 for item in successes), 6),
        },
        "comparison": comparison,
        "category_totals": category_totals,
        "top_price_categories": top_price_categories,
        "items": items,
    }


def _comparison_item(items: list[dict], field: str) -> dict | None:
    if not items:
        return None
    item = max(items, key=lambda row: row.get(field, 0))
    return {
        "video_id": item.get("video_id"),
        "url": item.get("url"),
        "value": item.get(field, 0),
    }


def _write_batch_report(report: dict, report_dir: Path) -> dict[str, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / "batch_summary.json"
    markdown_path = report_dir / "comparison.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(_format_batch_markdown(report), encoding="utf-8")
    return {"json": json_path, "markdown": markdown_path}


def _format_batch_markdown(report: dict) -> str:
    totals = report["totals"]
    price_categories = report.get("top_price_categories", [])
    lines = [
        f"# Batch {report['batch']}",
        "",
        "## Summary",
        "",
        f"- Videos processed: {totals['videos']}",
        f"- Successful: {totals['successful']}",
        f"- Failed: {totals['failed']}",
        f"- Lots: {totals['lots']}",
        f"- Animals: {totals['animals']}",
        f"- Sold: {totals['sold']}",
        f"- Not sold: {totals['not_sold']}",
        "",
        "## Comparison",
        "",
    ]

    labels = {
        "most_lots": "Most lots",
        "most_animals": "Most animals",
        "highest_avg_price": "Highest average price",
    }
    for key, label in labels.items():
        item = report["comparison"].get(key)
        if item:
            lines.append(f"- {label}: {item['video_id']} ({item['value']})")

    lines.extend(
        [
            "",
            "## Videos",
            "",
            "| Status | Video | Data | Cidade | Leilão | Leiloeiro | Lots | Animals | Avg price |",
            "|---|---|---|---|---|---|---:|---:|---:|",
        ]
    )
    for item in report["items"]:
        if item["status"] == "success":
            lines.append(
                "| success "
                f"| {item['video_id']} "
                f"| {_batch_text(item.get('date'))} "
                f"| {_batch_text(item.get('city'))} "
                f"| {_batch_text(_item_auction_name(item))} "
                f"| {_batch_text(_item_auctioneer(item))} "
                f"| {item['total_lots']} "
                f"| {item['total_animals']} "
                f"| {item['avg_price']:.2f} |"
            )
        else:
            lines.append(f"| failed | {item['url']} | - | - | - | - | 0 | 0 | 0.00 |")

    if price_categories:
        lines.extend(
            [
                "",
                "## Preço por Categoria",
                "",
                "| Video | " + " | ".join(price_categories) + " |",
                "|---|" + "|".join("---:" for _ in price_categories) + "|",
            ]
        )
        for item in report["items"]:
            if item["status"] != "success":
                continue
            prices = _format_batch_category_price_cells(
                item.get("category_prices", {}),
                price_categories,
            )
            lines.append(f"| {item['video_id']} | " + " | ".join(prices) + " |")

    lines.append("")
    return "\n".join(lines)


def _batch_text(value) -> str:
    return str(value) if value else "-"


def _item_auction_name(item: dict) -> str | None:
    return item.get("auction_name") or item.get("farm") or item.get("auctioneer")


def _item_auctioneer(item: dict) -> str | None:
    return item.get("auctioneer_display") or item.get("auctioneer")


def _format_batch_category_prices(category_prices: dict, categories: list[str]) -> str:
    if not categories:
        return "-"
    return "  |  ".join(
        f"{category}: {price}"
        for category, price in zip(
            categories,
            _format_batch_category_price_cells(category_prices, categories),
        )
    )


def _format_batch_category_price_cells(category_prices: dict, categories: list[str]) -> list[str]:
    parts = []
    for category in categories:
        price = category_prices.get(category)
        value = f"R$ {price:,.2f}" if price else "-"
        parts.append(value)
    return parts


def _print_batch_summary(report: dict, report_path: Path) -> None:
    totals = report["totals"]
    console.rule("[bold green]Batch Summary")
    console.print(
        f"  Videos: {totals['videos']}"
        f"  |  OK: {totals['successful']}"
        f"  |  Failed: {totals['failed']}"
    )
    console.print(
        f"  Lotes: {totals['lots']}"
        f"  |  Animais: {totals['animals']}"
        f"  |  Vendidos: {totals['sold']}"
        f"  |  Não vendidos: {totals['not_sold']}"
    )
    console.print(f"  Report: [cyan]{report_path}[/cyan]\n")

    table = Table(title="Batch Comparison", show_lines=True)
    table.add_column("Status", overflow="fold")
    table.add_column("Video", overflow="fold")
    table.add_column("Data", overflow="fold")
    table.add_column("Cidade", overflow="fold")
    table.add_column("Leilão", overflow="fold")
    table.add_column("Leiloeiro", overflow="fold")
    table.add_column("Lotes", justify="right", overflow="fold")
    table.add_column("Animais", justify="right", overflow="fold")
    table.add_column("Preço Médio", justify="right", overflow="fold")

    price_categories = report.get("top_price_categories", [])

    for item in report["items"]:
        if item["status"] == "success":
            avg_price = f"R$ {item['avg_price']:,.2f}" if item.get("avg_price") else "-"
            table.add_row(
                "[green]OK[/green]",
                item["video_id"],
                _batch_text(item.get("date")),
                _batch_text(item.get("city")),
                _batch_text(_item_auction_name(item)),
                _batch_text(_item_auctioneer(item)),
                str(item["total_lots"]),
                str(item["total_animals"]),
                avg_price,
            )
        else:
            table.add_row("[red]FAIL[/red]", item["url"], "-", "-", "-", "-", "-", "-", "-")

    console.print(table)

    if not price_categories:
        return

    price_table = Table(title="Preço por Categoria (top 3 por animais)", show_lines=True)
    price_table.add_column("Video", no_wrap=True)
    for category in price_categories:
        price_table.add_column(category.title(), justify="right", no_wrap=True)

    for item in report["items"]:
        if item["status"] != "success":
            continue
        price_table.add_row(
            item["video_id"],
            *_format_batch_category_price_cells(
                item.get("category_prices", {}),
                price_categories,
            ),
        )

    console.print(price_table)


_MALE_CATEGORIES = ("bezerro", "garrote", "novilho", "boi", "touro")
_FEMALE_CATEGORIES = ("bezerra", "novilha", "vaca", "bezerra desmamada", "vaca parida", "vaca prenha", "vaca com bezerro")


def _calculate_summary(lots: list) -> dict:
    """Calculate summary statistics from lots."""
    if not lots:
        return {}

    from collections import Counter, defaultdict

    total_lots = len(lots)
    total_animals = sum(lot.num_animals for lot in lots)

    # Count by sex
    sex_animals = {}
    for sex in ["macho", "fêmea", "misto"]:
        animals = sum(lot.num_animals for lot in lots if lot.sex == sex)
        if animals > 0:
            sex_animals[sex] = animals

    # Animal count by category (all categories, sorted by count desc)
    cat_counter = Counter()
    for lot in lots:
        cat_counter[lot.category] += lot.num_animals
    category_animals = dict(cat_counter.most_common())

    # Price statistics
    priced_lots = [
        lot for lot in lots
        if lot.unit_price and lot.unit_price > 0 and lot.num_animals > 0
    ]
    priced_animals = sum(lot.num_animals for lot in priced_lots)
    avg_price = (
        sum(lot.unit_price * lot.num_animals for lot in priced_lots) / priced_animals
        if priced_animals else 0
    )

    # Avg price by category
    cat_price_totals: dict = defaultdict(float)
    cat_priced_animals: dict = defaultdict(int)
    for lot in priced_lots:
        cat_price_totals[lot.category] += lot.unit_price * lot.num_animals
        cat_priced_animals[lot.category] += lot.num_animals
    category_prices = {
        cat: cat_price_totals[cat] / cat_priced_animals[cat]
        for cat in cat_price_totals
    }

    # Sold status
    sold = sum(1 for lot in lots if lot.sold is True)
    not_sold = sum(1 for lot in lots if lot.sold is False)

    return {
        "total_lots": total_lots,
        "total_animals": total_animals,
        "sex_animals": sex_animals,
        "category_animals": category_animals,
        "avg_price": avg_price,
        "category_prices": category_prices,
        "sold": sold,
        "not_sold": not_sold,
    }


def _print_summary(summary: dict) -> None:
    """Print summary statistics."""
    if not summary:
        return

    cat_animals = summary.get("category_animals", {})
    cat_prices = summary.get("category_prices", {})

    def _cat_animal_str(cats):
        parts = [f"{c}: {cat_animals[c]}" for c in cats if c in cat_animals]
        return "  |  ".join(parts)

    def _cat_price_str(cats):
        parts = [f"{c}: R$ {cat_prices[c]:,.2f}" for c in cats if c in cat_prices]
        return "  |  ".join(parts)

    console.print("[bold cyan]Resumo:[/bold cyan]")

    # Lotes line — same 2-space indent as all other lines
    console.print(
        f"  Lotes: {summary['total_lots']}"
        f"  |  Vendidos: {summary.get('sold', 0)}"
        f"  |  Não vendidos: {summary.get('not_sold', 0)}"
    )

    # Animais line
    sex = summary.get("sex_animals", {})
    sex_str = "  |  ".join(f"{k.title()}: {v}" for k, v in sex.items())
    console.print(f"  Animais: {summary['total_animals']}" + (f"  |  {sex_str}" if sex_str else ""))

    # Category animal-count lines (male / female / other)
    male_str = _cat_animal_str(_MALE_CATEGORIES)
    female_str = _cat_animal_str(_FEMALE_CATEGORIES)
    other_cats = [c for c in cat_animals if c not in _MALE_CATEGORIES and c not in _FEMALE_CATEGORIES]
    other_str = _cat_animal_str(other_cats)

    if male_str:
        console.print(f"  Categorias M:  {male_str}")
    if female_str:
        console.print(f"  Categorias F:  {female_str}")
    if other_str:
        console.print(f"  Categorias +:  {other_str}")

    # Overall avg price
    if summary.get("avg_price"):
        console.print(f"  Preço Médio: R$ {summary['avg_price']:,.2f} / cabeça")

    # Avg price by category — same M/F split to keep lines short and aligned
    male_price_str = _cat_price_str(_MALE_CATEGORIES)
    female_price_str = _cat_price_str(_FEMALE_CATEGORIES)
    other_price_cats = [c for c in cat_prices if c not in _MALE_CATEGORIES and c not in _FEMALE_CATEGORIES]
    other_price_str = _cat_price_str(other_price_cats)

    if male_price_str or female_price_str or other_price_str:
        console.print("  Preço por Categoria:")
        if male_price_str:
            console.print(f"    M:  {male_price_str}")
        if female_price_str:
            console.print(f"    F:  {female_price_str}")
        if other_price_str:
            console.print(f"    +:  {other_price_str}")

    console.print()


def _stage(num: str, name: str) -> None:
    console.rule(f"[bold cyan][{num}] {name}")


def _done(t0: float) -> None:
    elapsed = time.time() - t0
    mins, secs = divmod(int(elapsed), 60)
    console.print(f"  [green]✓[/green] Done in {mins}m {secs:02d}s\n")


def _clear_cache(run_dir: Path, video_id: str) -> None:
    import shutil
    for name in [
        f"video_info_{video_id}.json",
        f"transcript_{video_id}.json",
        f"screenshots_{video_id}.json",
        f"ocr_results_{video_id}.json",
        f"lots_{video_id}.json",
        f"metadata_{video_id}.json",
        f"result_{video_id}.json",
    ]:
        p = run_dir / name
        if p.exists():
            p.unlink()
    shots_dir = run_dir / f"screenshots_{video_id}"
    if shots_dir.exists():
        shutil.rmtree(shots_dir)
    console.print("  [yellow]Cache cleared.[/yellow]\n")


def _print_table(lots) -> None:
    if not lots:
        console.print("[yellow]No lots found.[/yellow]")
        return

    table = Table(title=f"{len(lots)} Lots Found", show_lines=True)
    table.add_column("Lote", style="bold", justify="right")
    table.add_column("Sexo")
    table.add_column("Categoria")
    table.add_column("Qtd", justify="right")
    table.add_column("Idade")
    table.add_column("Raça")
    table.add_column("Preço/cab (R$)", justify="right")
    table.add_column("Vendido")
    table.add_column("Timestamp")

    sorted_lots = sorted(lots, key=lambda l: l.timestamp_start or "99:99:99")
    for lot in sorted_lots:
        age = f"{lot.age_months}m" if lot.age_months else "-"
        price = f"{lot.unit_price:,.2f}" if lot.unit_price else "-"
        if lot.sold is True:
            sold = "[green]✓[/green]"
        elif lot.sold is False:
            sold = "[red]✗[/red]"
        else:
            sold = "-"
        table.add_row(
            str(lot.lot_number),
            lot.sex,
            lot.category,
            str(lot.num_animals),
            age,
            lot.breed,
            price,
            sold,
            lot.timestamp_start or "-",
        )

    console.print(table)


if __name__ == "__main__":
    cli()

import json
import sys
import time
from pathlib import Path

import click
from dotenv import load_dotenv

load_dotenv()

from rich.console import Console
from rich.table import Table

from models.lot import AuctionResult
from pipeline import aggregator, downloader, extractor, ocr, screenshotter
from pipeline import transcriber as transcriber_mod

console = Console()

PROMPTS_DIR = Path(__file__).parent / "prompts"


@click.command()
@click.argument("url")
@click.option(
    "--provider",
    type=click.Choice(["openai", "openrouter", "ollama"]),
    default="openai",
    show_default=True,
    help="LLM provider for lot extraction.",
)
@click.option(
    "--model",
    default=None,
    help=(
        "Model name or alias. Defaults: gpt-4.1-nano (openai), "
        "google/gemma-4-31b-it:free (openrouter), qwen3.5:397b-cloud (ollama). "
        "OpenAI models: gpt-4.1-nano, gpt-4o-mini, gpt-5-nano. "
        "Alias: gemini-2.5-flash-lite."
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
    default=30,
    show_default=True,
    help="Seconds between screenshots.",
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
def cli(url, provider, model, output_dir, transcriber, whisper_model, cpp_model, screenshot_interval, no_resume, show_metadata, show_summary, show_table):
    """Extract lot data from a Brazilian cattle auction YouTube video."""
    model = model or extractor.default_model(provider)

    console.rule("[bold]Cattle Auction Extractor")
    console.print(f"  URL:         {url}")
    console.print(f"  Transcriber: {transcriber}" + (f" / {whisper_model}" if transcriber != "groq" else " (Whisper Large v3 Turbo)"))
    console.print(f"  Extractor:   {provider} / {model}")
    console.print()

    video_id = downloader.get_video_id(url)
    run_dir = Path(output_dir) / video_id
    run_dir.mkdir(parents=True, exist_ok=True)

    if no_resume:
        _clear_cache(run_dir, video_id)

    video_info = downloader.get_video_info(url, run_dir, video_id)

    # ── Stage 1: Download ────────────────────────────────────────────────
    _stage("1/6", "Download")
    t0 = time.time()
    video_path, audio_path = downloader.download(url, run_dir, video_id)
    _done(t0)

    # ── Stage 2: Transcribe ──────────────────────────────────────────────
    _stage("2/6", f"Transcribe audio ({transcriber})")
    t0 = time.time()
    segments = transcriber_mod.transcribe(
        audio_path,
        output_path=run_dir / f"transcript_{video_id}.json",
        backend=transcriber,
        whisper_model=whisper_model,
        cpp_model_path=cpp_model,
    )
    _done(t0)

    # ── Stage 3: Screenshots ─────────────────────────────────────────────
    _stage("3/6", f"Extract screenshots (every {screenshot_interval}s)")
    t0 = time.time()
    shots = screenshotter.extract_screenshots(
        video_path,
        output_dir=run_dir,
        video_id=video_id,
        interval=screenshot_interval,
    )
    _done(t0)

    # ── Stage 4: OCR ─────────────────────────────────────────────────────
    _stage("4/6", "Run OCR on screenshots")
    t0 = time.time()
    ocr_results = ocr.run_ocr(shots, output_path=run_dir / f"ocr_results_{video_id}.json")
    _done(t0)

    # ── Stage 5: Extract lots ─────────────────────────────────────────────
    _stage("5/6", f"Extract lots with {provider}/{model}")
    t0 = time.time()
    windows = aggregator.aggregate(segments, ocr_results)
    console.print(f"  Aggregated into {len(windows)} windows.")

    client = extractor.LLMClient(provider=provider, model=model)
    lots = extractor.extract_lots(
        windows,
        client=client,
        prompt_path=PROMPTS_DIR / "extraction.txt",
        output_path=run_dir / f"lots_{video_id}.json",
    )
    _done(t0)

    # ── Stage 6: Extract auction metadata ────────────────────────────────
    _stage("6/6", f"Extract auction metadata with {provider}/{model}")
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
    console.print(f"  Output: [cyan]{run_dir}/[/cyan]")
    console.print()

    if show_summary:
        summary_stats = _calculate_summary(lots)
        _print_summary(summary_stats)

    if show_table:
        _print_table(lots)


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
    prices = [lot.unit_price for lot in lots if lot.unit_price and lot.unit_price > 0]
    avg_price = sum(prices) / len(prices) if prices else 0

    # Avg price by category
    cat_price_lists: dict = defaultdict(list)
    for lot in lots:
        if lot.unit_price and lot.unit_price > 0:
            cat_price_lists[lot.category].append(lot.unit_price)
    category_prices = {cat: sum(pl) / len(pl) for cat, pl in cat_price_lists.items()}

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

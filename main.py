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
    type=click.Choice(["claude", "openai", "openrouter"]),
    default="openai",
    show_default=True,
    help="LLM provider for lot extraction.",
)
@click.option(
    "--model",
    default=None,
    help=(
        "Model name or alias. Defaults: gpt-4o-mini (openai), claude-sonnet-4-6 (claude), "
        "google/gemini-2.5-flash-lite-preview-09-2025 (openrouter). "
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
    default="mlx",
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
def cli(url, provider, model, output_dir, transcriber, whisper_model, cpp_model, screenshot_interval, no_resume):
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

    # ── Stage 1: Download ────────────────────────────────────────────────
    _stage("1/5", "Download")
    t0 = time.time()
    video_path, audio_path = downloader.download(url, run_dir, video_id)
    _done(t0)

    # ── Stage 2: Transcribe ──────────────────────────────────────────────
    _stage("2/5", f"Transcribe audio ({transcriber})")
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
    _stage("3/5", f"Extract screenshots (every {screenshot_interval}s)")
    t0 = time.time()
    shots = screenshotter.extract_screenshots(
        video_path,
        output_dir=run_dir,
        video_id=video_id,
        interval=screenshot_interval,
    )
    _done(t0)

    # ── Stage 4: OCR ─────────────────────────────────────────────────────
    _stage("4/5", "Run OCR on screenshots")
    t0 = time.time()
    ocr_results = ocr.run_ocr(shots, output_path=run_dir / f"ocr_results_{video_id}.json")
    _done(t0)

    # ── Stage 5: Extract lots ─────────────────────────────────────────────
    _stage("5/5", f"Extract lots with {provider}/{model}")
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

    # ── Summary ───────────────────────────────────────────────────────────
    result = AuctionResult(
        video_url=url,
        video_id=video_id,
        total_lots=len(lots),
        lots=lots,
    )
    summary_path = run_dir / f"result_{video_id}.json"
    summary_path.write_text(
        json.dumps(result.model_dump(), ensure_ascii=False, indent=2)
    )

    console.rule("[bold green]Done")
    console.print(f"  Found [bold]{len(lots)}[/bold] lots.")
    console.print(f"  Output: [cyan]{run_dir}/[/cyan]")
    console.print()

    _print_table(lots)


def _stage(num: str, name: str) -> None:
    console.rule(f"[bold cyan][{num}] {name}")


def _done(t0: float) -> None:
    elapsed = time.time() - t0
    mins, secs = divmod(int(elapsed), 60)
    console.print(f"  [green]✓[/green] Done in {mins}m {secs:02d}s\n")


def _clear_cache(run_dir: Path, video_id: str) -> None:
    import shutil
    for name in [
        f"transcript_{video_id}.json",
        f"screenshots_{video_id}.json",
        f"ocr_results_{video_id}.json",
        f"lots_{video_id}.json",
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
    table.add_column("Timestamp")

    for lot in lots:
        age = f"{lot.age_months}m" if lot.age_months else "-"
        price = f"{lot.unit_price:,.2f}" if lot.unit_price else "-"
        table.add_row(
            str(lot.lot_number),
            lot.sex,
            lot.category,
            str(lot.num_animals),
            age,
            lot.breed,
            price,
            lot.timestamp_start or "-",
        )

    console.print(table)


if __name__ == "__main__":
    cli()

"""Tests for batch CLI/report behavior."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from click.testing import CliRunner

from main import _batch_success_item, _build_batch_report, _format_batch_markdown, _load_batch_urls, cli
from models.lot import AuctionResult, Lot


def _result(video_url, video_id, lots, *, date=None, city=None, auctioneer=None,
            farm=None, notes=None, cost_usd=None):
    return AuctionResult(
        video_url=video_url,
        video_id=video_id,
        date=date,
        city=city,
        auctioneer=auctioneer,
        farm=farm,
        notes=notes,
        total_lots=len(lots),
        lots=lots,
        cost_usd=cost_usd,
    )


def _lot(lot_number, category, num_animals, unit_price=None, sold=True):
    return Lot(
        lot_number=lot_number,
        sex="macho",
        category=category,
        num_animals=num_animals,
        breed="Nelore",
        unit_price=unit_price,
        sold=sold,
    )


def test_load_batch_urls_reads_text_file_ignoring_blank_and_comment_lines(tmp_path):
    batch_file = tmp_path / "links.txt"
    batch_file.write_text(
        "\n"
        "# leilões de maio\n"
        "https://www.youtube.com/watch?v=aaa\n"
        "   \n"
        "https://youtu.be/bbb\n",
        encoding="utf-8",
    )

    assert _load_batch_urls((), batch_file) == [
        "https://www.youtube.com/watch?v=aaa",
        "https://youtu.be/bbb",
    ]


def test_batch_file_runs_urls_in_sequence_and_saves_report(monkeypatch, tmp_path):
    batch_file = tmp_path / "links.txt"
    batch_file.write_text(
        "https://www.youtube.com/watch?v=aaa\n"
        "https://www.youtube.com/watch?v=bbb\n",
        encoding="utf-8",
    )
    calls = []

    def fake_run_single_url(url, **kwargs):
        calls.append((url, kwargs["output_dir"]))
        if url.endswith("aaa"):
            return _result(
                url,
                "aaa",
                [_lot(1, "bezerro", 10, 3000.0, sold=True)],
                date="01/05/2026",
                city="Quirinópolis",
                auctioneer="Leilões A",
            )
        return _result(
            url,
            "bbb",
            [
                _lot(1, "vaca", 3, 5000.0, sold=False),
                _lot(2, "vaca", 2, 5200.0, sold=True),
            ],
            date="02/05/2026",
            city="Rio Verde",
            auctioneer="Leilões B",
        )

    monkeypatch.setattr("main._run_single_url", fake_run_single_url)

    result = CliRunner().invoke(
        cli,
        [
            "--batch-file",
            str(batch_file),
            "--batch-name",
            "maio",
            "--output-dir",
            str(tmp_path / "output"),
            "--no-table",
        ],
    )

    assert result.exit_code == 0, result.output
    assert [call[0] for call in calls] == [
        "https://www.youtube.com/watch?v=aaa",
        "https://www.youtube.com/watch?v=bbb",
    ]

    report_path = tmp_path / "output" / "batches" / "maio" / "batch_summary.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["totals"]["videos"] == 2
    assert report["totals"]["successful"] == 2
    assert report["totals"]["failed"] == 0
    assert report["totals"]["lots"] == 3
    assert report["totals"]["animals"] == 15
    assert report["comparison"]["most_animals"]["video_id"] == "aaa"
    assert report["comparison"]["highest_avg_price"]["video_id"] == "bbb"
    assert report["category_totals"] == {"bezerro": 10, "vaca": 5}
    assert report["top_price_categories"] == ["bezerro", "vaca"]
    assert report["items"][0]["date"] == "01/05/2026"
    assert report["items"][0]["city"] == "Quirinópolis"
    assert report["items"][0]["auctioneer"] == "Leilões A"
    assert report["items"][0]["auction_name"] == "Leilões A"
    assert report["items"][0]["auctioneer_display"] == "Leilões A"
    assert report["items"][0]["category_prices"] == {"bezerro": 3000.0}
    assert "Batch Summary" in result.output
    assert "maio/batch_summary.json" in result.output

    markdown = (tmp_path / "output" / "batches" / "maio" / "comparison.md").read_text(encoding="utf-8")
    assert "| Status | Video | Data | Cidade | Leilão | Leiloeiro | Lots | Animals | Avg price |" in markdown
    assert "## Preço por Categoria" in markdown
    assert "| Video | bezerro | vaca |" in markdown
    assert "Top Categoria" not in markdown
    assert "Output / Error" not in markdown
    assert "| aaa | R$ 3,000.00 | - |" in markdown


def test_batch_continues_after_failed_url_and_records_error(monkeypatch, tmp_path):
    batch_file = tmp_path / "links.txt"
    batch_file.write_text(
        "https://www.youtube.com/watch?v=bad\n"
        "https://www.youtube.com/watch?v=ok\n",
        encoding="utf-8",
    )

    def fake_run_single_url(url, **kwargs):
        if url.endswith("bad"):
            raise RuntimeError("download failed")
        return _result(url, "ok", [_lot(1, "garrote", 7, 4000.0)])

    monkeypatch.setattr("main._run_single_url", fake_run_single_url)

    result = CliRunner().invoke(
        cli,
        [
            "--batch-file",
            str(batch_file),
            "--batch-name",
            "mixed",
            "--output-dir",
            str(tmp_path / "output"),
            "--no-table",
        ],
    )

    assert result.exit_code == 1, result.output
    report_path = tmp_path / "output" / "batches" / "mixed" / "batch_summary.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["totals"]["videos"] == 2
    assert report["totals"]["successful"] == 1
    assert report["totals"]["failed"] == 1
    assert report["items"][0]["status"] == "failed"
    assert report["items"][0]["error"] == "download failed"
    assert report["items"][1]["status"] == "success"


def test_batch_report_uses_top_three_categories_for_price_comparison():
    items = [
        {
            "index": 1,
            "status": "success",
            "url": "https://example.com/a",
            "video_id": "aaa",
            "output_dir": "output/aaa",
            "date": "01/05/2026",
            "city": "Quirinópolis",
            "auctioneer": "Leilões A",
            "total_lots": 4,
            "total_animals": 27,
            "sold": 4,
            "not_sold": 0,
            "avg_price": 3000.0,
            "category_animals": {
                "bezerro": 12,
                "garrote": 8,
                "novilha": 4,
                "vaca": 3,
            },
            "category_prices": {
                "bezerro": 2800.0,
                "garrote": 3600.0,
                "novilha": 2900.0,
                "vaca": 4100.0,
            },
        },
        {
            "index": 2,
            "status": "success",
            "url": "https://example.com/b",
            "video_id": "bbb",
            "output_dir": "output/bbb",
            "date": "02/05/2026",
            "city": "Rio Verde",
            "auctioneer": "Leilões B",
            "total_lots": 3,
            "total_animals": 20,
            "sold": 2,
            "not_sold": 1,
            "avg_price": 2500.0,
            "category_animals": {
                "bezerra": 10,
                "bezerro": 5,
                "garrote": 5,
            },
            "category_prices": {
                "bezerra": 2300.0,
                "bezerro": 2600.0,
                "garrote": 3400.0,
            },
        },
    ]

    report = _build_batch_report("maio", items)

    assert report["top_price_categories"] == ["bezerro", "garrote", "bezerra"]

    markdown = _format_batch_markdown(report)
    assert "| Status | Video | Data | Cidade | Leilão | Leiloeiro | Lots | Animals | Avg price |" in markdown
    assert "## Preço por Categoria" in markdown
    assert "| Video | bezerro | garrote | bezerra |" in markdown
    assert "Top Categoria" not in markdown
    assert "Output/Error" not in markdown
    assert "| aaa | R$ 2,800.00 | R$ 3,600.00 | - |" in markdown
    assert "| bbb | R$ 2,600.00 | R$ 3,400.00 | R$ 2,300.00 |" in markdown


def test_batch_success_item_infers_missing_auction_metadata_from_farm_and_notes(tmp_path):
    result = _result(
        "https://www.youtube.com/watch?v=abc",
        "abc",
        [_lot(1, "bezerro", 10, 3000.0)],
        date="30/04/2026",
        city="Quirinópolis",
        farm="Clube dos Amigos Leilões",
        notes=(
            "Leilão anunciado como 'LEILÃO DO SINDICATO RURAL DE QUIRINÓPOLIS-GO'. "
            "Leiloeiro mencionado no áudio é Eliseu Vieira."
        ),
    )

    report = _build_batch_report("metadata", [_batch_success_item(1, result, tmp_path)])

    item = report["items"][0]
    assert item["auction_name"] == "Clube dos Amigos Leilões"
    assert item["auctioneer_display"] == "Eliseu Vieira"

    markdown = _format_batch_markdown(report)
    assert "Clube dos Amigos Leilões" in markdown
    assert "Eliseu Vieira" in markdown


def test_batch_report_sums_estimated_cost_across_videos(tmp_path):
    items = [
        _batch_success_item(1, _result("u1", "aaa", [_lot(1, "boi", 5, 4000.0)], cost_usd=0.0512), tmp_path),
        _batch_success_item(2, _result("u2", "bbb", [_lot(2, "vaca", 3, 3000.0)], cost_usd=0.0130), tmp_path),
    ]

    assert [item["cost_usd"] for item in items] == [0.0512, 0.0130]

    report = _build_batch_report("maio", items)
    assert report["totals"]["cost_usd"] == 0.0642


def test_batch_cost_total_tolerates_runs_that_reported_no_cost(tmp_path):
    """A fully resumed run spends nothing and reports cost_usd=None."""
    items = [
        _batch_success_item(1, _result("u1", "aaa", [_lot(1, "boi", 5, 4000.0)], cost_usd=None), tmp_path),
        _batch_success_item(2, _result("u2", "bbb", [_lot(2, "vaca", 3, 3000.0)], cost_usd=0.02), tmp_path),
    ]

    report = _build_batch_report("maio", items)
    assert report["totals"]["cost_usd"] == 0.02


def test_frame_sampling_flag_reaches_the_pipeline(monkeypatch, tmp_path):
    captured = {}

    def fake_run_single_url(url, **kwargs):
        captured.update(kwargs)
        return _result(url, "aaa", [_lot(1, "boi", 5, 4000.0)])

    monkeypatch.setattr("main._run_single_url", fake_run_single_url)

    result = CliRunner().invoke(
        cli,
        [
            "https://www.youtube.com/watch?v=aaa",
            "--frame-sampling", "scene",
            "--safety-interval", "45",
            "--output-dir", str(tmp_path),
            "--no-summary", "--no-table", "--no-metadata",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["frame_sampling"] == "scene"
    assert captured["safety_interval"] == 45


def test_frame_sampling_defaults_to_interval(monkeypatch, tmp_path):
    captured = {}

    def fake_run_single_url(url, **kwargs):
        captured.update(kwargs)
        return _result(url, "aaa", [_lot(1, "boi", 5, 4000.0)])

    monkeypatch.setattr("main._run_single_url", fake_run_single_url)

    result = CliRunner().invoke(
        cli,
        [
            "https://www.youtube.com/watch?v=aaa",
            "--output-dir", str(tmp_path),
            "--no-summary", "--no-table", "--no-metadata",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["frame_sampling"] == "interval"
    assert captured["safety_interval"] == 60

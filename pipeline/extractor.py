import json
import os
import re
from pathlib import Path

from models.lot import Lot
from pipeline.aggregator import Window

# Number of windows from the start to scan for auction metadata
_METADATA_WINDOWS = 3


_DEFAULT_MODELS = {
    "openai": "gpt-4.1-nano",
    "openrouter": "google/gemma-4-31b-it:free",
    "ollama": "qwen3.5:397b-cloud",
}

# Short aliases accepted by --model
_MODEL_ALIASES = {
    "gemini-2.5-flash-lite": "google/gemini-2.5-flash-lite-preview-09-2025",
}


class LLMClient:
    def __init__(self, provider: str, model: str):
        self.provider = provider
        self.model = _MODEL_ALIASES.get(model, model)  # resolve alias

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

        elif provider == "ollama":
            import openai
            self._client = openai.OpenAI(
                api_key="ollama",
                base_url="http://127.0.0.1:11434/v1",
            )

        else:
            raise ValueError(f"Unknown provider: {provider!r}. Use 'openai', 'openrouter', or 'ollama'.")

    def complete(self, system: str, user: str, max_retries: int = 3) -> str:
        import time as _time

        # openai, openrouter, and ollama all share the same SDK interface
        # Newer OpenAI models (gpt-5-*) require max_completion_tokens instead of max_tokens
        token_param = "max_completion_tokens" if self.model.startswith("gpt-5") else "max_tokens"

        for attempt in range(max_retries + 1):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    **{token_param: 4096},
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                )
                return resp.choices[0].message.content
            except Exception as e:
                if "429" in str(e) and attempt < max_retries:
                    wait = 2 ** attempt * 5  # 5s, 10s, 20s
                    print(f"    Rate limited, retrying in {wait}s...")
                    _time.sleep(wait)
                    continue
                raise


def extract_lots(
    windows: list[Window],
    client: LLMClient,
    prompt_path: Path,
    output_path: Path,
) -> list[Lot]:
    """Process each window with the LLM, deduplicate, and save lots.json."""
    if output_path.exists():
        print(f"  Lots already extracted, loading from cache.")
        return _load(output_path)

    system_prompt = prompt_path.read_text(encoding="utf-8")
    lots_by_number: dict[int, Lot] = {}
    total = len(windows)

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
            continue

        for lot in new_lots:
            _merge(lots_by_number, lot)

        print(f"    -> {len(new_lots)} lot(s) found, {len(lots_by_number)} total so far.")

    lots = sorted(lots_by_number.values(), key=lambda l: l.lot_number)
    _save(lots, output_path)
    return lots


def _parse_response(response: str) -> list[Lot]:
    """Parse JSON array from LLM response, tolerating extra text."""
    text = response.strip()

    try:
        data = json.loads(text)
        if isinstance(data, list):
            return _validate_lots(data)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\[.*?\]", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            if isinstance(data, list):
                return _validate_lots(data)
        except json.JSONDecodeError:
            pass

    return []


def _validate_lots(data: list[dict]) -> list[Lot]:
    lots = []
    for item in data:
        try:
            lots.append(Lot(**item))
        except Exception:
            pass
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
    if sold_new is True:
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
        print(f"  Auction metadata already extracted, loading from cache.")
        return json.loads(output_path.read_text(encoding="utf-8"))

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

    try:
        response = client.complete(system_prompt, combined)
        text = response.strip()
        # Tolerate extra text around the JSON object
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            metadata = json.loads(match.group())
        else:
            metadata = json.loads(text)
    except Exception as e:
        print(f"  WARNING: Metadata extraction failed: {e}")
        metadata = {}

    output_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def default_model(provider: str) -> str:
    return _DEFAULT_MODELS.get(provider, "gpt-4o-mini")

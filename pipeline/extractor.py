import json
import os
import re
from pathlib import Path

from models.lot import Lot
from pipeline.aggregator import Window


_DEFAULT_MODELS = {
    "claude": "claude-sonnet-4-6",
    "openai": "gpt-4o-mini",
    "openrouter": "google/gemini-2.5-flash-lite-preview-09-2025",
}

# Short aliases accepted by --model
_MODEL_ALIASES = {
    "gemini-2.5-flash-lite": "google/gemini-2.5-flash-lite-preview-09-2025",
}


class LLMClient:
    def __init__(self, provider: str, model: str):
        self.provider = provider
        self.model = _MODEL_ALIASES.get(model, model)  # resolve alias

        if provider == "claude":
            import anthropic
            self._client = anthropic.Anthropic()

        elif provider == "openai":
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
            raise ValueError(f"Unknown provider: {provider!r}. Use 'claude', 'openai', or 'openrouter'.")

    def complete(self, system: str, user: str) -> str:
        if self.provider == "claude":
            msg = self._client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return msg.content[0].text

        else:  # openai + openrouter share the same SDK interface
            resp = self._client.chat.completions.create(
                model=self.model,
                max_tokens=4096,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            return resp.choices[0].message.content


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

        print(f"    → {len(new_lots)} lot(s) found, {len(lots_by_number)} total so far.")

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

    merged = {
        k: existing_data[k] if existing_data[k] is not None else new_data[k]
        for k in existing_data
    }
    lots_by_number[new_lot.lot_number] = Lot(**merged)


def _save(lots: list[Lot], path: Path) -> None:
    data = [lot.model_dump() for lot in lots]
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))


def _load(path: Path) -> list[Lot]:
    data = json.loads(path.read_text())
    return [Lot(**d) for d in data]


def default_model(provider: str) -> str:
    return _DEFAULT_MODELS.get(provider, "gpt-4o-mini")

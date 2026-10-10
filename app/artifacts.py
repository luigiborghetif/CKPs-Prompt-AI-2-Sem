"""Resultados identificados por execução, sem sobrescrever evidências antigas."""

import csv
import json
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

from app.config import ROOT, Settings
from app.tokens import ENCODING


def run_directory(name: str, destination: Path | None = None) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = (destination or ROOT / "evidence" / "ckp01") / f"{name}_{stamp}_{uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_json(path: Path, data) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metadata(settings: Settings) -> dict:
    # Lista positiva de campos: nunca serializar Settings com a credencial.
    return {
        "utc": datetime.now(timezone.utc).isoformat(),
        "model": settings.model,
        "api_model": settings.effective_model,
        "base_url": settings.base_url,
        "temperature": settings.temperature,
        "max_output_tokens": settings.max_output_tokens,
        "memory_tokens": settings.memory_tokens,
        "tokenizer": ENCODING,
        "tokens_are_estimates": True,
        "versions": {
            name: version(name)
            for name in [
                "langchain-classic",
                "langchain-core",
                "langchain-ollama",
                "pydantic",
                "tiktoken",
            ]
        },
    }

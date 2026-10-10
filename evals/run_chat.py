"""Conjunto de consultas reais e verificações observáveis, sem juiz oculto."""

import argparse
import json
import unicodedata
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from app.artifacts import metadata, run_directory, write_csv, write_json
from app.config import ROOT, Settings, load_settings
from app.prompts import SYSTEM_V1
from app.service import Assistant, public_error
from app.tokens import count_text

CASES = ROOT / "evals" / "chat_cases.json"


def normalize(value: str) -> str:
    return "".join(
        char
        for char in unicodedata.normalize("NFKD", value.casefold())
        if not unicodedata.combining(char)
    )


def checks_for(answer: str, analysis: dict, expected: dict) -> dict[str, bool]:
    checks = {}
    text = normalize(answer)
    for key, value in expected.items():
        if key == "answer_contains_all":
            checks[key] = all(normalize(item) in text for item in value)
        elif key == "answer_contains_any":
            checks[key] = any(normalize(item) in text for item in value)
        elif key == "answer_not_contains":
            checks[key] = all(normalize(item) not in text for item in value)
        elif key == "tipo_residuo_contains":
            checks[key] = normalize(value) in normalize(analysis.get("tipo_residuo") or "")
        else:
            checks[key] = analysis.get(key) == value
    return checks


def evaluate(settings: Settings, *, system: str = SYSTEM_V1, assistant=None) -> list[dict]:
    assistant = assistant if assistant is not None else Assistant(settings, system=system)
    cases = json.loads(CASES.read_text(encoding="utf-8"))
    records = []
    for case in cases:
        session_id = uuid4().hex
        start = perf_counter()
        record = {"id": case["id"], "turns": [], "status": "error", "checks": {}, "passed": False}
        try:
            for query in case["turns"]:
                result = assistant.consult(query, session_id)
                record["turns"].append(
                    {
                        "query": query,
                        "answer": result.answer,
                        "analysis": result.analysis.model_dump(mode="json"),
                        "history_tokens": result.history_tokens,
                        "answer_tokens": count_text(result.answer),
                    }
                )
            last = record["turns"][-1]
            record["checks"] = checks_for(last["answer"], last["analysis"], case["expected"])
            record["status"] = "ok"
            record["passed"] = all(record["checks"].values())
        except Exception as exc:
            record["error"] = public_error(exc)
        finally:
            assistant.clear(session_id)
        record["latency_seconds"] = round(perf_counter() - start, 4)
        records.append(record)
    return records


def summary(records: list[dict]) -> dict:
    return {
        "cases": len(records),
        "passed": sum(item["passed"] for item in records),
        "errors": sum(item["status"] == "error" for item in records),
        "method": "Verificações de schema, campos e expressões; requer revisão humana da fundamentação.",
    }


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", type=Path)
    cli.add_argument("--system-prompt", type=Path)
    args = cli.parse_args()
    settings = load_settings()
    if not settings.configured:
        raise SystemExit("Preencha OLLAMA_API_KEY no .env antes de executar a avaliação.")
    system = args.system_prompt.read_text(encoding="utf-8") if args.system_prompt else SYSTEM_V1
    path = run_directory("chat", args.output)
    write_json(path / "metadata.json", metadata(settings))
    (path / "system_prompt.md").write_text(system, encoding="utf-8")
    records = evaluate(settings, system=system)
    write_json(path / "results.json", records)
    write_json(path / "summary.json", summary(records))
    write_csv(
        path / "results.csv",
        [
            {
                "id": item["id"],
                "status": item["status"],
                "passed": item["passed"],
                "latency_seconds": item["latency_seconds"],
            }
            for item in records
        ],
    )
    print(f"Evidências: {path}")
    if not all(item["passed"] for item in records):
        raise SystemExit(
            "Há casos com falha. Revise results.json; resultados não foram descartados."
        )


if __name__ == "__main__":
    main()

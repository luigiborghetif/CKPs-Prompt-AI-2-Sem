"""Comparação controlada e auditável, sem trocar métricas RAGAS por heurísticas."""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from time import perf_counter

from app.artifacts import metadata, run_directory, write_csv, write_json
from app.config import ROOT, Settings, load_settings
from app.rag.config import RagSettings, load_rag_settings
from app.rag.loaders import sha256
from app.rag.pipeline import RagPipeline
from app.rag.retrieval import SearchFilters
from app.service import public_error


@dataclass(frozen=True)
class Experiment:
    name: str
    chunk_size: int
    overlap: int
    rerank: bool = False


DEFAULT_EXPERIMENTS = (
    Experiment("256_32_baseline", 256, 32),
    Experiment("512_64_baseline", 512, 64),
    Experiment("512_64_rerank", 512, 64, True),
)


def summarize(rows: list[dict]) -> tuple[list[dict], dict]:
    summaries = []
    for name in dict.fromkeys(item["configuration"] for item in rows):
        group = [row for row in rows if row["configuration"] == name]
        valid = [row for row in group if row["status"] == "ok"]
        complete = len(valid) == len(group) and len(group) >= 5
        summaries.append(
            {
                "configuration": name,
                "questions": len(group),
                "valid": len(valid),
                "errors": len(group) - len(valid),
                "complete": complete,
                "faithfulness_mean": mean(row["faithfulness"] for row in valid) if valid else None,
                "answer_relevancy_mean": mean(row["answer_relevancy"] for row in valid)
                if valid
                else None,
                "latency_mean_seconds": mean(row["latency_seconds"] for row in group),
            }
        )
    # Configuração com falhas não pode ganhar por ter avaliado só as perguntas fáceis.
    eligible = [item for item in summaries if item["complete"] and item["faithfulness_mean"] >= 0.7]
    winner = (
        max(eligible, key=lambda item: (item["faithfulness_mean"], item["answer_relevancy_mean"]))
        if eligible
        else None
    )
    return summaries, {
        "selected": winner["configuration"] if winner else None,
        "reason": "Maior faithfulness; answer_relevancy como desempate; todas as perguntas válidas."
        if winner
        else "Nenhuma configuração completa atingiu faithfulness médio ≥ 0,7.",
        "target_ideal": 0.9,
    }


def run_comparison(
    settings: Settings,
    config: RagSettings,
    *,
    destination: Path | None = None,
    experiments=DEFAULT_EXPERIMENTS,
    pipeline_factory=None,
    evaluator=None,
) -> Path:
    if len({item.chunk_size for item in experiments}) < 2:
        raise ValueError("Compare ao menos dois tamanhos de chunk.")
    questions_path = ROOT / "evals/rag_questions.json"
    questions = json.loads(questions_path.read_text(encoding="utf-8"))
    if len(questions) < 5:
        raise ValueError("A comparação exige pelo menos cinco perguntas.")
    owns_evaluator = evaluator is None
    if evaluator is None:
        from app.rag.evaluation import RagasEvaluator

        evaluator = RagasEvaluator(settings, config)
    factory = pipeline_factory or (lambda run_config: RagPipeline(settings, run_config))
    path = run_directory("rag_comparison", destination or ROOT / "evidence/ckp02")
    info = metadata(settings.model_copy(update={"temperature": 0}))
    info.update(
        {
            "ragas_version": __import__("importlib.metadata", fromlist=["version"]).version(
                "ragas"
            ),
            "manifest_sha256": sha256(config.manifest),
            "questions_sha256": sha256(questions_path),
            "chunk_units": "characters",
            "candidates": config.candidates,
            "top_k": config.top_k,
            "embedding_model": config.embedding_model,
            "embedding_endpoint": config.embedding_base_url,
            "reranker_model": config.reranker_model,
            "reranker_revision": config.reranker_revision,
            "experiments": [vars(experiment) for experiment in experiments],
        }
    )
    write_json(path / "metadata.json", info)
    rows, details = [], []
    for experiment in experiments:
        run_config = config.model_copy(
            update={
                "chunk_size": experiment.chunk_size,
                "chunk_overlap": experiment.overlap,
                "rerank": experiment.rerank,
            }
        )
        run_config = RagSettings.model_validate(run_config.model_dump())
        pipeline = factory(run_config)
        for question in questions:
            start = perf_counter()
            row = {
                "configuration": experiment.name,
                "question_id": question["id"],
                "question": question["question"],
                "status": "error",
                "faithfulness": None,
                "answer_relevancy": None,
                "expected_source_found": False,
                "error": None,
            }
            detail = {"configuration": experiment.name, "question": question}
            try:
                result = pipeline.buscar(question["question"])
                detail["result"] = result.model_dump(mode="json")
                scores = evaluator.score(
                    question["question"],
                    result.evaluation_answer,
                    result.evaluation_contexts,
                    question["reference"],
                )
                row.update(scores)
                row["expected_source_found"] = bool(
                    set(question["source_ids"])
                    & {item.metadata["source_id"] for item in result.chunks}
                )
                row["status"] = "ok"
            except Exception as exc:
                row["error"] = public_error(exc)
                row["error_type"] = type(exc).__name__
            row["latency_seconds"] = round(perf_counter() - start, 4)
            detail["evaluation"] = row.copy()
            rows.append(row)
            details.append(detail)
            write_json(path / "results.json", details)
    # Campo constante evita diferenças de cabeçalho entre linhas com/sem erro.
    if owns_evaluator:
        evaluator.close()
    for row in rows:
        row.setdefault("error_type", None)
    write_csv(path / "metrics.csv", rows)
    summaries, selection = summarize(rows)
    write_json(path / "summary.json", summaries)
    write_csv(path / "summary.csv", summaries)
    write_json(path / "selection.json", selection)
    return path


def run_boundary_cases(pipeline: RagPipeline, path: Path) -> list[dict]:
    cases = json.loads((ROOT / "evals/rag_boundary_cases.json").read_text(encoding="utf-8"))
    rows = []
    for case in cases:
        row = {"id": case["id"], "expected": case["expected_status"], "passed": False}
        try:
            result = pipeline.buscar(case["question"])
            row.update(
                {
                    "actual": result.structured.status.value,
                    "answer": result.answer,
                    "passed": result.structured.status.value == case["expected_status"],
                }
            )
        except Exception as exc:
            row["error"] = public_error(exc)
        rows.append(row)
    write_json(path / "boundary_cases.json", rows)
    return rows


def demonstrate_filters(pipeline: RagPipeline, path: Path) -> dict:
    filters = SearchFilters(tipo="norma", categoria="ecopontos", published_from_year=2020)
    result = pipeline.buscar(
        "Quais condições a norma estabelece para logística reversa em Ecopontos?", filters
    )
    if not result.chunks or any(
        item.metadata["tipo"] != "norma"
        or item.metadata["categoria"] != "ecopontos"
        or item.metadata["ano"] < 2020
        for item in result.chunks
    ):
        raise ValueError("Filtro de metadados não retornou os documentos esperados.")
    data = {
        "filters": filters.model_dump(),
        "where": filters.where(),
        "result": result.model_dump(mode="json"),
    }
    write_json(path / "metadata_filter.json", data)
    return data


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", type=Path)
    cli.add_argument(
        "--without-reranking",
        action="store_true",
        help="Executa apenas as duas configurações obrigatórias.",
    )
    args = cli.parse_args()
    settings = load_settings()
    if not settings.configured:
        raise SystemExit("Preencha OLLAMA_API_KEY antes de avaliar o RAG.")
    config = load_rag_settings(settings)
    experiments = DEFAULT_EXPERIMENTS[:2] if args.without_reranking else DEFAULT_EXPERIMENTS
    try:
        path = run_comparison(settings, config, destination=args.output, experiments=experiments)
        selection = json.loads((path / "selection.json").read_text())
        print(f"Evidências: {path}")
        print(json.dumps(selection, ensure_ascii=False, indent=2))
        if selection["selected"] is None:
            raise SystemExit(
                "Revise resultados e configuração; nenhuma comparação completa atingiu a meta mínima."
            )
    except Exception as exc:
        raise SystemExit(public_error(exc)) from None


if __name__ == "__main__":
    main()

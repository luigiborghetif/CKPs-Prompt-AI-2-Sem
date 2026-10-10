import asyncio
import json
from types import SimpleNamespace

import pytest
from conftest import OfflineChatModel
from test_rag import OfflineEmbeddings

from app.config import Settings
from app.rag.config import RagSettings
from app.rag.evaluation import RagasEvaluator
from evals.run_rag import Experiment, run_comparison, summarize


def test_actual_ragas_metrics_with_deterministic_test_models():
    # JSONs de teste conhecidos; não correspondem a execução/evidência do Ollama.
    responses = [
        '{"statements":["O limite é 1 m³ por descarga."]}',
        '{"statements":[{"statement":"O limite é 1 m³ por descarga.","reason":"O contexto informa o limite.","verdict":1}]}',
        '{"question":"Qual o limite por descarga?","noncommittal":0}',
        '{"question":"Qual o limite por descarga?","noncommittal":0}',
        '{"question":"Qual o limite por descarga?","noncommittal":0}',
    ]
    evaluator = RagasEvaluator(
        Settings(timeout=3),
        RagSettings(),
        llm=OfflineChatModel(responses=responses),
        embeddings=OfflineEmbeddings(),
    )
    scores = evaluator.score(
        "Qual o limite por descarga?",
        "O limite é 1 m³ por descarga.",
        ["Art. 3: limite de 1 m³ por descarga."],
        "1 m³.",
    )
    assert scores["faithfulness"] == 1
    assert scores["answer_relevancy"] == pytest.approx(1)


def test_nan_is_failure_not_a_successful_zero():
    class Metric:
        async def single_turn_ascore(self, *args, **kwargs):
            return float("nan")

    evaluator = object.__new__(RagasEvaluator)
    evaluator.faithfulness = Metric()
    evaluator.answer_relevancy = Metric()
    evaluator.timeout = 1
    with pytest.raises(ValueError, match="métrica ausente"):
        evaluator.score("Q", "A", ["C"], "R")


def test_notebook_event_loop_can_run_metric_adapter():
    class Metric:
        async def single_turn_ascore(self, *args, **kwargs):
            return 0.9

    evaluator = object.__new__(RagasEvaluator)
    evaluator.faithfulness = Metric()
    evaluator.answer_relevancy = Metric()
    evaluator.timeout = 1

    async def notebook_cell():
        return evaluator.score("Q", "A", ["C"], "R")

    assert asyncio.run(notebook_cell())["faithfulness"] == 0.9


def test_failed_configuration_cannot_win_from_partial_average():
    rows = [
        {
            "configuration": "complete",
            "status": "ok",
            "faithfulness": 0.8,
            "answer_relevancy": 0.9,
            "latency_seconds": 1,
        }
        for _ in range(5)
    ]
    rows += [
        {
            "configuration": "partial",
            "status": "ok" if index < 4 else "error",
            "faithfulness": 1 if index < 4 else None,
            "answer_relevancy": 1 if index < 4 else None,
            "latency_seconds": 1,
        }
        for index in range(5)
    ]
    summaries, selection = summarize(rows)
    assert selection["selected"] == "complete"
    assert summaries[1]["errors"] == 1 and not summaries[1]["complete"]


def test_repeated_scores_keep_client_event_loop_including_notebook():
    class LoopBoundMetric:
        loop = None

        async def single_turn_ascore(self, *args, **kwargs):
            loop = asyncio.get_running_loop()
            if self.loop is None:
                self.loop = loop
            if self.loop is not loop:
                raise RuntimeError("Cliente reutilizado em outro loop")
            return 0.9

    evaluator = object.__new__(RagasEvaluator)
    evaluator.faithfulness = LoopBoundMetric()
    evaluator.answer_relevancy = LoopBoundMetric()
    evaluator.timeout = 1
    try:
        assert evaluator.score("Q", "A", ["C"], "R")["faithfulness"] == 0.9
        assert evaluator.score("Q2", "A", ["C"], "R")["faithfulness"] == 0.9

        async def cell():
            return evaluator.score("Q3", "A", ["C"], "R")

        assert asyncio.run(cell())["faithfulness"] == 0.9
    finally:
        evaluator.close()


def test_comparison_preserves_results_for_every_question(tmp_path):
    class Pipeline:
        def buscar(self, query):
            return SimpleNamespace(
                evaluation_answer="Resposta de teste",
                evaluation_contexts=["Contexto de teste"],
                chunks=[
                    SimpleNamespace(text="Contexto de teste", metadata={"source_id": "SP-002"})
                ],
                model_dump=lambda **kwargs: {"test_only": True},
            )

    class Evaluator:
        def score(self, *args):
            return {"faithfulness": 0.8, "answer_relevancy": 0.9}

    path = run_comparison(
        Settings(),
        RagSettings(),
        destination=tmp_path,
        experiments=[Experiment("a", 256, 32), Experiment("b", 512, 64)],
        pipeline_factory=lambda config: Pipeline(),
        evaluator=Evaluator(),
    )
    rows = json.loads((path / "results.json").read_text())
    assert len(rows) == 20
    assert all(row["evaluation"]["status"] == "ok" for row in rows)
    assert (path / "metrics.csv").is_file()

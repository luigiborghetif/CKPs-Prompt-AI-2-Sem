"""RAGAS com adaptadores explícitos para Ollama, sem chave OpenAI."""

import asyncio
import math
import os
from concurrent.futures import ThreadPoolExecutor
from functools import cached_property

os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")

from langchain_core.outputs import ChatGeneration, LLMResult
from ragas import SingleTurnSample
from ragas.embeddings import BaseRagasEmbeddings
from ragas.llms import BaseRagasLLM
from ragas.metrics import Faithfulness, ResponseRelevancy
from ragas.run_config import RunConfig

from app.chain import create_llm
from app.config import Settings
from app.rag.config import RagSettings
from app.rag.embeddings import create_embeddings


class OllamaJudge(BaseRagasLLM):
    """Contrato RAGAS adaptado ao ChatOllama, incluindo múltiplas completions."""

    def __init__(self, llm, timeout: float):
        super().__init__(run_config=RunConfig(timeout=timeout, max_retries=1, max_workers=1))
        self.llm = llm

    def is_finished(self, response: LLMResult) -> bool:
        return all(
            generation.message.response_metadata.get("done_reason") != "length"
            for generations in response.generations
            for generation in generations
        )

    def generate_text(self, prompt, n=1, temperature=None, stop=None, callbacks=None):
        generations = [
            ChatGeneration(
                message=self.llm.invoke(
                    prompt, stop=stop, config={"callbacks": callbacks} if callbacks else {}
                )
            )
            for _ in range(n)
        ]
        return LLMResult(generations=[generations])

    async def agenerate_text(self, prompt, n=1, temperature=None, stop=None, callbacks=None):
        generations = []
        for _ in range(n):
            message = await self.llm.ainvoke(
                prompt, stop=stop, config={"callbacks": callbacks} if callbacks else {}
            )
            generations.append(ChatGeneration(message=message))
        return LLMResult(generations=[generations])


class OllamaMetricEmbeddings(BaseRagasEmbeddings):
    def __init__(self, embeddings):
        super().__init__()
        self.embeddings = embeddings

    def embed_query(self, text):
        return self.embeddings.embed_query(text)

    def embed_documents(self, texts):
        return self.embeddings.embed_documents(texts)

    async def aembed_query(self, text):
        return await self.embeddings.aembed_query(text)

    async def aembed_documents(self, texts):
        return await self.embeddings.aembed_documents(texts)


class RagasEvaluator:
    def __init__(self, settings: Settings, config: RagSettings, *, llm=None, embeddings=None):
        judge_settings = settings.model_copy(update={"temperature": 0, "max_output_tokens": 4096})
        llm = llm if llm is not None else create_llm(judge_settings, structured=True)
        embeddings = embeddings if embeddings is not None else create_embeddings(settings, config)
        judge = OllamaJudge(llm, settings.timeout)
        self.faithfulness = Faithfulness(llm=judge)
        self.answer_relevancy = ResponseRelevancy(
            llm=judge, embeddings=OllamaMetricEmbeddings(embeddings)
        )
        self.timeout = settings.timeout

    async def score_async(
        self, question: str, answer: str, contexts: list[str], reference: str
    ) -> dict:
        sample = SingleTurnSample(
            user_input=question, response=answer, retrieved_contexts=contexts, reference=reference
        )
        faithfulness = float(
            await self.faithfulness.single_turn_ascore(sample, timeout=self.timeout)
        )
        relevancy = float(
            await self.answer_relevancy.single_turn_ascore(sample, timeout=self.timeout)
        )
        if not all(math.isfinite(value) and 0 <= value <= 1 for value in (faithfulness, relevancy)):
            raise ValueError("RAGAS retornou uma métrica ausente/não finita ou fora da escala.")
        return {"faithfulness": faithfulness, "answer_relevancy": relevancy}

    @cached_property
    def _sync_worker(self):
        return ThreadPoolExecutor(max_workers=1, thread_name_prefix="ragas-score")

    @cached_property
    def _sync_runner(self):
        return asyncio.Runner()

    def close(self):
        worker = self.__dict__.get("_sync_worker")
        if worker is not None:
            runner = self.__dict__.get("_sync_runner")
            if runner is not None:
                worker.submit(runner.close).result()
            worker.shutdown()
            self.__dict__.pop("_sync_worker", None)
            self.__dict__.pop("_sync_runner", None)

    def score(self, question: str, answer: str, contexts: list[str], reference: str) -> dict:
        async def evaluate():
            return await self.score_async(question, answer, contexts, reference)

        # Cliente HTTP assíncrono mantém o mesmo loop entre perguntas, também no Jupyter.
        return self._sync_worker.submit(lambda: self._sync_runner.run(evaluate())).result()

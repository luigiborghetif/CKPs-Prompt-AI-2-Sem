"""Cross-encoder inicializado somente ao solicitar reordenação."""

from threading import RLock

from app.rag.config import RagSettings
from app.rag.schemas import RetrievedChunk


class CrossEncoderReranker:
    def __init__(self, config: RagSettings, model=None):
        self.config = config
        self._model = model
        self._lock = RLock()

    def rank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        if not chunks:
            return []
        with self._lock:
            if self._model is None:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(
                    self.config.reranker_model,
                    cache_folder=str(self.config.model_cache_path),
                    device="cpu",
                    revision=self.config.reranker_revision,
                    local_files_only=self.config.reranker_local_only,
                )
            scores = self._model.predict([(query, item.text) for item in chunks])
        if len(scores) != len(chunks):
            raise ValueError("Reranker não produziu uma pontuação por candidato.")
        scored = [
            item.model_copy(update={"rerank_score": float(score)})
            for item, score in zip(chunks, scores)
        ]
        # Revalidar evita deixar NaN passar pelo model_copy sem validação.
        scored = [RetrievedChunk.model_validate(item.model_dump()) for item in scored]
        return sorted(scored, key=lambda item: item.rerank_score, reverse=True)

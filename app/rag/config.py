"""Parâmetros de recuperação; não cria clientes nem baixa modelos."""

import os
from pathlib import Path

from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.config import ROOT, Settings


class RagSettings(BaseModel):
    model_config = ConfigDict(frozen=True)
    manifest: Path = ROOT / "data" / "sources.json"
    store_path: Path = ROOT / ".chroma"
    embedding_base_url: str = "https://ollama.com"
    embedding_model: str = "nomic-embed-text"
    chunk_size: int = Field(default=512, ge=128, le=4096)
    chunk_overlap: int = Field(default=64, ge=0)
    candidates: int = Field(default=8, ge=1, le=50)
    top_k: int = Field(default=4, ge=1, le=20)
    rerank: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_revision: str | None = None
    reranker_local_only: bool = False
    model_cache_path: Path = ROOT / ".model-cache"
    embedding_batch_size: int = Field(default=32, ge=1, le=256)

    @model_validator(mode="after")
    def validate_settings(self):
        if self.embedding_model != "nomic-embed-text":
            raise ValueError("A CKP02 exige exclusivamente nomic-embed-text.")
        Settings(base_url=self.embedding_base_url)
        if not 0.10 <= self.chunk_overlap / self.chunk_size <= 0.15:
            raise ValueError("chunk_overlap deve ser 10–15% de chunk_size.")
        if self.top_k > self.candidates:
            raise ValueError("top_k não pode exceder o número de candidatos.")
        return self


def load_rag_settings(settings: Settings, env_file: Path | None = None) -> RagSettings:
    values = {**dotenv_values(env_file or ROOT / ".env"), **os.environ}
    fields = {
        "RAG_CHUNK_SIZE": "chunk_size",
        "RAG_CHUNK_OVERLAP": "chunk_overlap",
        "RAG_CANDIDATES": "candidates",
        "RAG_TOP_K": "top_k",
        "RAG_RERANK": "rerank",
        "RAG_STORE_PATH": "store_path",
        "OLLAMA_EMBED_BASE_URL": "embedding_base_url",
        "RAG_RERANKER_MODEL": "reranker_model",
        "RAG_RERANKER_REVISION": "reranker_revision",
        "RAG_RERANKER_LOCAL_ONLY": "reranker_local_only",
        "RAG_MODEL_CACHE_PATH": "model_cache_path",
    }
    args = {field: values[key] for key, field in fields.items() if values.get(key)}
    args.setdefault("embedding_base_url", settings.base_url)
    for field in ("store_path", "model_cache_path"):
        if field in args:
            path = Path(args[field])
            args[field] = path if path.is_absolute() else ROOT / path
    return RagSettings(**args)

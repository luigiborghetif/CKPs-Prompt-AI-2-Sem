"""Embeddings exclusivamente nomic-embed-text; sem fallback automático."""

from langchain_ollama import OllamaEmbeddings

from app.chain import MissingConfiguration
from app.config import Settings
from app.rag.config import RagSettings


def create_embeddings(settings: Settings, config: RagSettings) -> OllamaEmbeddings:
    if not settings.configured:
        raise MissingConfiguration("Preencha OLLAMA_API_KEY para usar os embeddings do Ollama.")
    return OllamaEmbeddings(
        model="nomic-embed-text",
        base_url=config.embedding_base_url,
        client_kwargs={
            "headers": {"Authorization": f"Bearer {settings.api_key.get_secret_value()}"},
            "timeout": settings.timeout,
        },
    )

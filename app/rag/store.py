"""Chroma local sem embedding padrão implícito; índices por conteúdo/configuração."""

import hashlib
import json
import math

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_core.documents import Document

from app.rag.config import RagSettings
from app.rag.schemas import RetrievedChunk


def fingerprint(chunks: list[Document], config: RagSettings) -> str:
    identity = {
        "chunk_size": config.chunk_size,
        "overlap": config.chunk_overlap,
        "embedding_model": config.embedding_model,
        "endpoint": config.embedding_base_url,
        "chunks": [
            (item.metadata["chunk_id"], item.page_content, item.metadata) for item in chunks
        ],
    }
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def validate_vectors(vectors, expected: int):
    if len(vectors) != expected or not vectors:
        raise ValueError("Embeddings não correspondem ao lote solicitado.")
    dimensions = len(vectors[0])
    if dimensions == 0 or any(len(vector) != dimensions for vector in vectors):
        raise ValueError("Dimensões de embedding incompatíveis.")
    if any(not math.isfinite(value) for vector in vectors for value in vector):
        raise ValueError("Embedding contém valor não finito.")


class VectorStore:
    def __init__(self, config: RagSettings, *, client=None):
        self.config = config
        self.client = (
            client
            if client is not None
            else chromadb.PersistentClient(
                path=str(config.store_path), settings=ChromaSettings(anonymized_telemetry=False)
            )
        )
        self.collection = None

    def build(self, chunks: list[Document], embeddings):
        if not chunks:
            raise ValueError("Não indexar uma base vazia.")
        signature = fingerprint(chunks, self.config)
        name = f"residuos_sp_{self.config.chunk_size}_{signature[:16]}"
        collection = self.client.get_or_create_collection(
            name=name,
            embedding_function=None,
            configuration={"hnsw": {"space": "cosine"}},
            metadata={"fingerprint": signature, "complete": False},
        )
        if collection.count() == len(chunks) and collection.metadata.get("complete"):
            self.collection = collection
            return collection
        for start in range(0, len(chunks), self.config.embedding_batch_size):
            batch = chunks[start : start + self.config.embedding_batch_size]
            vectors = embeddings.embed_documents([item.page_content for item in batch])
            validate_vectors(vectors, len(batch))
            collection.upsert(
                ids=[item.metadata["chunk_id"] for item in batch],
                embeddings=vectors,
                documents=[item.page_content for item in batch],
                metadatas=[item.metadata for item in batch],
            )
        if collection.count() != len(chunks):
            raise ValueError("O índice não contém todos os chunks esperados.")
        collection.modify(metadata={**collection.metadata, "complete": True})
        self.collection = collection
        return collection

    def query(self, vector: list[float], *, where=None) -> list[RetrievedChunk]:
        if self.collection is None:
            raise ValueError("Índice ainda não foi construído.")
        validate_vectors([vector], 1)
        if self.collection.count() == 0:
            return []
        result = self.collection.query(
            query_embeddings=[vector],
            n_results=min(self.config.candidates, self.collection.count()),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        return [
            RetrievedChunk(id=chunk_id, text=text, metadata=meta, distance=distance)
            for chunk_id, text, meta, distance in zip(
                result["ids"][0],
                result["documents"][0],
                result["metadatas"][0],
                result["distances"][0],
            )
        ]

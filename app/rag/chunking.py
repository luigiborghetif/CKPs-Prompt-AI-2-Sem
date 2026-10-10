"""Divisão em caracteres, conforme os separadores exigidos no PDF."""

import hashlib

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.rag.config import RagSettings

SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def split_documents(documents: list[Document], config: RagSettings) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        separators=SEPARATORS,
        length_function=len,
        add_start_index=True,
    )
    chunks = splitter.split_documents(documents)
    for index, chunk in enumerate(chunks):
        digest = hashlib.sha256(
            (
                chunk.metadata["source_id"]
                + chunk.metadata["section"]
                + str(chunk.metadata["start_index"])
                + chunk.page_content
            ).encode()
        ).hexdigest()[:16]
        chunk.metadata["chunk_id"] = f"{chunk.metadata['source_id']}-{config.chunk_size}-{digest}"
        chunk.metadata["chunk_index"] = index
    return chunks

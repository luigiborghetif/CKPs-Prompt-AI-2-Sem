"""Carrega fontes entregues e verifica integridade antes de indexar."""

import hashlib
import json
import re
from pathlib import Path

from langchain_core.documents import Document
from pypdf import PdfReader

from app.config import ROOT
from app.rag.schemas import Source


def safe_path(relative: str, root: Path = ROOT) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or Path(relative).is_absolute():
        raise ValueError("Caminho da fonte deve permanecer dentro do projeto.")
    return path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_sources(manifest: Path, root: Path = ROOT) -> list[Source]:
    sources = [
        Source.model_validate(item) for item in json.loads(manifest.read_text(encoding="utf-8"))
    ]
    if len(sources) < 5 or len({source.id for source in sources}) != len(sources):
        raise ValueError("A base exige ≥ 5 fontes com IDs distintos.")
    if len({source.url for source in sources}) != len(sources):
        raise ValueError("Não contar cópias do mesmo documento como fontes diferentes.")
    for source in sources:
        for relative, expected in [
            (source.raw_file, source.raw_sha256),
            (source.processed_file, source.processed_sha256),
        ]:
            if sha256(safe_path(relative, root)) != expected:
                raise ValueError(
                    f"Fonte {source.id} alterada. Revise e atualize o manifesto antes de indexar."
                )
    return sources


def load_documents(sources: list[Source], root: Path = ROOT) -> list[Document]:
    documents = []
    for source in sources:
        path = safe_path(source.processed_file, root)
        base = {
            "source_id": source.id,
            "title": source.title,
            "url": source.url,
            "tipo": source.type,
            "categoria": source.category,
            "jurisdicao": source.jurisdiction,
            "data_publicacao": str(source.publication_date)
            if source.publication_date
            else "desconhecida",
            "ano": source.publication_date.year if source.publication_date else 0,
            "data_coleta": str(source.collected_at),
            "sha256": source.processed_sha256,
        }
        if path.suffix.lower() == ".pdf":
            for index, page in enumerate(PdfReader(path).pages, 1):
                text = (page.extract_text() or "").strip()
                if text:
                    documents.append(
                        Document(page_content=text, metadata={**base, "section": f"Página {index}"})
                    )
        elif path.suffix.lower() in {".md", ".txt"}:
            text = path.read_text(encoding="utf-8").strip()
            parts = re.split(r"(?m)^## ", text)
            for part in parts:
                heading, _, body = part.partition("\n")
                if body.strip():
                    documents.append(
                        Document(
                            page_content=part.strip(),
                            metadata={**base, "section": heading.strip("# ")},
                        )
                    )
        else:
            raise ValueError("Documentos processados devem ser PDF, TXT ou Markdown.")
    if {document.metadata["source_id"] for document in documents} != {
        source.id for source in sources
    }:
        raise ValueError("Uma das fontes não produziu texto utilizável.")
    return documents

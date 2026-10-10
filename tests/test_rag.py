"""Integração offline usa Chroma real e modelos determinísticos de teste."""

import hashlib
import json
import re

import numpy as np
import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import ValidationError

from app.config import ROOT, Settings
from app.rag.chunking import split_documents
from app.rag.config import RagSettings
from app.rag.loaders import load_documents, load_sources, safe_path
from app.rag.pipeline import RagPipeline, validate_citations
from app.rag.reranking import CrossEncoderReranker
from app.rag.retrieval import SearchFilters, infer_source_filter
from app.rag.schemas import GeneratedAnswer, RetrievedChunk
from app.rag.store import VectorStore, fingerprint, validate_vectors
from scripts.prepare_sources import extract_articles


class OfflineEmbeddings:
    calls: int = 0

    def embed_query(self, text):
        vector = [0.0] * 64
        for token in re.findall(r"\w+", text.lower()):
            vector[int(hashlib.sha256(token.encode()).hexdigest()[:4], 16) % 64] += 1
        return vector

    def embed_documents(self, texts):
        self.calls += 1
        return [self.embed_query(text) for text in texts]

    async def aembed_query(self, text):
        return self.embed_query(text)

    async def aembed_documents(self, texts):
        return self.embed_documents(texts)


class CitingTestModel(BaseChatModel):
    wrong_id: bool = False

    @property
    def _llm_type(self):
        return "offline-test-only"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        ids = re.findall(r"<trecho id='([^']+)'", str(messages[0].content))
        data = {
            "status": "respondida",
            "claims": [
                {
                    "text": "Texto determinístico de teste.",
                    "chunk_ids": ["inexistente" if self.wrong_id else ids[0]],
                }
            ],
            "message": "",
            "limitations": [],
        }
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=json.dumps(data)))])


@pytest.fixture
def corpus():
    config = RagSettings(rerank=False)
    sources = load_sources(config.manifest)
    return sources, load_documents(sources)


@pytest.fixture
def pipeline(tmp_path):
    config = RagSettings(rerank=False, store_path=tmp_path / "chroma")
    return RagPipeline(Settings(), config, embeddings=OfflineEmbeddings(), llm=CitingTestModel())


def test_five_real_sources_and_relevant_sections(corpus):
    sources, docs = corpus
    assert len(sources) == 5
    assert len({source.url for source in sources}) == 5
    assert any("150m³" in doc.page_content for doc in docs)
    assert any("1m³" in doc.page_content and doc.metadata["section"] == "Art. 3" for doc in docs)
    assert any("18.299/2025" in doc.page_content for doc in docs)


def test_explicit_norm_filter_preserves_ambiguity_and_user_filters(corpus):
    sources, _ = corpus
    filters = infer_source_filter("Segundo o Decreto 55.113/2014?", sources, SearchFilters())
    assert filters.source_id == "SP-002"
    assert (
        infer_source_filter("Compare 55.113/2014 e 14.803/2008", sources, SearchFilters()).source_id
        is None
    )
    explicit = SearchFilters(source_id="SP-005")
    assert infer_source_filter("55.113/2014", sources, explicit) == explicit
    assert infer_source_filter("O que é um Ecoponto?", sources, SearchFilters()).source_id is None
    article = infer_source_filter(
        "Portaria SMSUB 8/2024, artigo 3, parágrafo 3", sources, SearchFilters()
    )
    assert article.source_id == "SP-005" and article.section == "Art. 3"
    assert infer_source_filter("O que diz o artigo 3?", sources, SearchFilters()).section is None


def test_schema_retry_keeps_citations_and_context(pipeline):
    class InitiallyInvalidModel(CitingTestModel):
        calls: int = 0

        def _generate(self, messages, **kwargs):
            self.calls += 1
            result = super()._generate(messages, **kwargs)
            if self.calls == 1:
                data = json.loads(result.generations[0].message.content)
                data["message"] = "Resumo duplicado não permitido"
                result.generations[0].message.content = json.dumps(data)
            return result

    model = InitiallyInvalidModel()
    pipeline._llm = model
    result = pipeline.buscar("Qual o limite do Ecoponto?")
    assert model.calls == 2
    assert result.structured.message == ""
    assert result.sources[0]["chunk_id"] in {chunk.id for chunk in result.chunks}
    assert all(
        chunk.metadata["title"] in context
        for chunk, context in zip(result.chunks, result.evaluation_contexts)
    )


def test_revoked_text_not_turned_into_active_rule():
    raw = b'<div class="bx-retornaLei"><div class="customStyle"><p>Art. 1 Regra atual <s>regra revogada</s></p></div></div>'
    text, removed = extract_articles(raw)
    assert "Regra atual" in text and "revogada" not in text
    assert removed == 1


def test_loader_rejects_tampered_files(tmp_path):
    import shutil

    shutil.copytree(ROOT / "data", tmp_path / "data")
    (tmp_path / "data/processed/SP-002.md").write_text("alterado")
    with pytest.raises(ValueError, match="alterada"):
        load_sources(tmp_path / "data/sources.json", tmp_path)


def test_paths_cannot_escape_project():
    with pytest.raises(ValueError):
        safe_path("../segredo")


@pytest.mark.parametrize("size,overlap", [(256, 32), (512, 64), (1024, 128)])
def test_chunk_size_identifiers_and_overlap(corpus, size, overlap):
    config = RagSettings(chunk_size=size, chunk_overlap=overlap)
    chunks = split_documents(corpus[1], config)
    assert all(len(chunk.page_content) <= size for chunk in chunks)
    assert len({chunk.metadata["chunk_id"] for chunk in chunks}) == len(chunks)
    assert 0.10 <= overlap / size <= 0.15


def test_bad_overlap_rejected():
    with pytest.raises(ValidationError):
        RagSettings(chunk_overlap=100)


def test_chroma_build_reuse_and_config_isolation(corpus, tmp_path):
    config = RagSettings(store_path=tmp_path)
    chunks = split_documents(corpus[1], config)
    embeddings = OfflineEmbeddings()
    store = VectorStore(config)
    first = store.build(chunks, embeddings)
    calls = embeddings.calls
    again = store.build(chunks, embeddings)
    assert first.name == again.name and embeddings.calls == calls
    other = config.model_copy(update={"chunk_size": 256, "chunk_overlap": 32})
    assert fingerprint(split_documents(corpus[1], other), other) != fingerprint(chunks, config)


def test_pipeline_returns_only_verified_citations_and_real_metadata(pipeline):
    result = pipeline.buscar("Ecoponto limite descarga")
    assert result.structured.status == "respondida"
    assert result.sources
    assert set(source["chunk_id"] for source in result.sources).issubset(
        {chunk.id for chunk in result.chunks}
    )
    assert all(
        source["url"].startswith("https://legislacao.prefeitura.sp.gov.br")
        for source in result.sources
    )
    assert "https://legislacao.prefeitura.sp.gov.br" in result.answer
    assert result.input_tokens > 0


def test_filters_and_empty_result_do_not_generate(pipeline):
    filters = SearchFilters(tipo="norma", categoria="ecopontos", published_from_year=2020)
    result = pipeline.buscar("logística reversa", filters)
    assert all(chunk.metadata["source_id"] == "SP-005" for chunk in result.chunks)
    empty = pipeline.buscar("consulta", SearchFilters(categoria="categoria_inexistente"))
    assert empty.structured.status == "sem_base"
    assert empty.sources == [] and empty.input_tokens == 0


def test_unknown_citation_is_rejected(pipeline):
    pipeline._llm = CitingTestModel(wrong_id=True)
    with pytest.raises(ValueError, match="ausente"):
        pipeline.buscar("Ecopontos")


def test_uncited_answer_cannot_pass_schema():
    with pytest.raises(ValidationError):
        GeneratedAnswer(status="respondida", message="Regra inventada.")


def test_urls_are_not_taken_from_generated_text():
    result = GeneratedAnswer(
        status="respondida", claims=[{"text": "https://fake.example", "chunk_ids": ["id"]}]
    )
    with pytest.raises(ValueError, match="URLs"):
        validate_citations(result, [RetrievedChunk(id="id", text="fonte", metadata={})])


def test_crossencoder_reorders_before_selection():
    class Model:
        def predict(self, pairs):
            assert len(pairs) == 3
            return np.array([0.1, 0.9, 0.4])

    chunks = [RetrievedChunk(id=str(index), text="teste", metadata={}) for index in range(3)]
    result = CrossEncoderReranker(RagSettings(), model=Model()).rank("consulta", chunks)
    assert [item.id for item in result] == ["1", "2", "0"]
    assert [item.rerank_score for item in chunks] == [None, None, None]


@pytest.mark.parametrize("vectors,expected", [([], 1), ([[1], [1, 2]], 2), ([[float("nan")]], 1)])
def test_malformed_embeddings_rejected(vectors, expected):
    with pytest.raises(ValueError):
        validate_vectors(vectors, expected)


def test_gradio_rag_builds_without_clients():
    from app.rag.main import build_app

    config = RagSettings()
    pipeline = RagPipeline(Settings(), config)
    assert build_app(pipeline).blocks
    assert pipeline._embeddings is None and pipeline._store is None and pipeline._llm is None

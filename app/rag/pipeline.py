"""Load → split → embed → store → retrieve → rerank → generate."""

from functools import lru_cache
from threading import RLock

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.chain import create_llm
from app.config import ROOT, Settings, load_settings
from app.rag.chunking import split_documents
from app.rag.config import RagSettings, load_rag_settings
from app.rag.embeddings import create_embeddings
from app.rag.loaders import load_documents, load_sources
from app.rag.prompts import HUMAN, SYSTEM
from app.rag.reranking import CrossEncoderReranker
from app.rag.retrieval import SearchFilters, infer_source_filter
from app.rag.schemas import AnswerStatus, GeneratedAnswer, RagResult, RetrievedChunk
from app.rag.store import VectorStore
from app.tokens import count_messages


def context_for(chunks: list[RetrievedChunk]) -> str:
    return "\n\n".join(
        f"<trecho id='{item.id}'>\n{item.evaluation_context}\n</trecho>" for item in chunks
    )


def validate_citations(answer: GeneratedAnswer, chunks: list[RetrievedChunk]) -> None:
    available = {item.id for item in chunks}
    for claim in answer.claims:
        if len(set(claim.chunk_ids)) != len(claim.chunk_ids):
            raise ValueError("Uma afirmação repetiu a mesma citação.")
        if not set(claim.chunk_ids).issubset(available):
            raise ValueError("O modelo citou um chunk ausente do contexto recuperado.")
        if "http://" in claim.text or "https://" in claim.text:
            raise ValueError("URLs devem ser construídas pela aplicação, não pelo modelo.")


def render_answer(answer: GeneratedAnswer, chunks: list[RetrievedChunk]) -> tuple[str, list[dict]]:
    lookup = {item.id: item for item in chunks}
    sources, texts = [], []
    if answer.status != AnswerStatus.RESPONDIDA:
        texts.append(answer.message)
    for claim in answer.claims:
        links = []
        for chunk_id in claim.chunk_ids:
            item = lookup[chunk_id]
            meta = item.metadata
            links.append(f"[{meta['source_id']} · {meta['section']} · {chunk_id}]({meta['url']})")
            if not any(source["chunk_id"] == chunk_id for source in sources):
                sources.append(
                    {
                        "chunk_id": chunk_id,
                        "document_id": meta["source_id"],
                        "title": meta["title"],
                        "section": meta["section"],
                        "url": meta["url"],
                        "collected_at": meta["data_coleta"],
                    }
                )
        texts.append(claim.text + "\n\nFonte: " + "; ".join(links))
    if answer.limitations:
        texts.append("Limitações: " + " ".join(answer.limitations))
    return "\n\n".join(texts), sources


class RagPipeline:
    def __init__(
        self,
        settings: Settings,
        config: RagSettings,
        *,
        root=ROOT,
        embeddings=None,
        llm=None,
        store=None,
        reranker=None,
    ):
        self.settings = settings.model_copy(update={"temperature": 0})
        self.config = config
        self.root = root
        self._embeddings = embeddings
        self._llm = llm
        self._store = store
        self._reranker = reranker
        self._ready = False
        self._lock = RLock()
        self.sources = []
        self.chunks = []

    def index(self) -> dict:
        with self._lock:
            if not self._ready:
                sources = load_sources(self.config.manifest, self.root)
                documents = load_documents(sources, self.root)
                chunks = split_documents(documents, self.config)
                if self._embeddings is None:
                    self._embeddings = create_embeddings(self.settings, self.config)
                if self._store is None:
                    self._store = VectorStore(self.config)
                collection = self._store.build(chunks, self._embeddings)
                self.sources, self.chunks = sources, chunks
                self._ready = True
            else:
                collection = self._store.collection
            return {
                "collection": collection.name,
                "documents": len(self.sources),
                "chunks": len(self.chunks),
                "chunk_size": self.config.chunk_size,
                "chunk_overlap": self.config.chunk_overlap,
            }

    def buscar(self, consulta: str, filtros: SearchFilters | None = None) -> RagResult:
        consulta = consulta.strip()
        if not consulta or len(consulta) > self.settings.max_input_chars:
            raise ValueError("Consulta vazia ou acima do limite de caracteres.")
        self.index()
        vector = self._embeddings.embed_query(consulta)
        filters = infer_source_filter(consulta, self.sources, filtros or SearchFilters())
        candidates = self._store.query(vector, where=filters.where())
        if self.config.rerank and candidates:
            with self._lock:
                if self._reranker is None:
                    self._reranker = CrossEncoderReranker(self.config)
            ranked = self._reranker.rank(consulta, candidates)
        else:
            ranked = candidates
        selected = ranked[: self.config.top_k]
        tokens = 0
        if not selected:
            answer = GeneratedAnswer(
                status=AnswerStatus.SEM_BASE,
                message="Não há trechos disponíveis para esta consulta e os filtros escolhidos.",
            )
        else:
            with self._lock:
                if self._llm is None:
                    self._llm = create_llm(self.settings, structured=True)
            parser = PydanticOutputParser(pydantic_object=GeneratedAnswer)
            prompt = ChatPromptTemplate.from_messages(
                [("system", SYSTEM), ("human", HUMAN)]
            ).partial(format_instructions=parser.get_format_instructions())
            args = {"consulta": consulta, "contexto": context_for(selected)}
            tokens = count_messages(prompt.invoke(args).to_messages())
            raw = (prompt | self._llm).invoke(args)
            try:
                answer = parser.invoke(raw)
            except OutputParserException:
                # Uma tentativa de correção de formato, sem relaxar schema/citações.
                messages = prompt.invoke(args).to_messages()
                messages.extend(
                    [
                        raw,
                        HumanMessage(
                            content="A saída não passou no schema. Corrija o JSON: status respondida exige "
                            "message vazio e todo conteúdo factual em claims citados. Outros status exigem "
                            "claims vazio e message com motivo. Não acrescente fatos ou IDs ausentes do contexto."
                        ),
                    ]
                )
                tokens += count_messages(messages)
                answer = parser.invoke(self._llm.invoke(messages))
            validate_citations(answer, selected)
        text, sources = render_answer(answer, selected)
        return RagResult(
            query=consulta,
            answer=text,
            structured=answer,
            chunks=selected,
            sources=sources,
            collection=self._store.collection.name,
            base_date=str(max(source.collected_at for source in self.sources)),
            input_tokens=tokens,
            ranking_before=[item.id for item in candidates],
            ranking_after=[item.id for item in ranked],
        )


@lru_cache(maxsize=1)
def default_pipeline() -> RagPipeline:
    settings = load_settings()
    return RagPipeline(settings, load_rag_settings(settings))


def buscar(consulta: str, filtros: dict | None = None) -> dict:
    """Contrato serializável para reutilizar como ferramenta no próximo CKP."""
    return (
        default_pipeline()
        .buscar(consulta, SearchFilters.model_validate(filtros or {}))
        .model_dump(mode="json")
    )

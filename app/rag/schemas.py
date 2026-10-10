"""Fontes auditáveis e saída estruturada com citações por afirmação."""

from datetime import date
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Source(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(pattern=r"^SP-\d{3}$")
    title: str = Field(min_length=1)
    url: str = Field(pattern=r"^https://")
    type: str = Field(pattern=r"^(norma|orientacao)$")
    category: str = Field(min_length=1)
    publisher: str
    jurisdiction: str = "sao_paulo_capital"
    publication_date: date | None
    collected_at: date
    raw_file: str
    processed_file: str
    raw_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    processed_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    excerpt: str
    excluded_revoked_elements: int = Field(ge=0)


class AnswerStatus(StrEnum):
    RESPONDIDA = "respondida"
    SEM_BASE = "sem_base"
    ESCLARECIMENTO = "esclarecimento"
    FORA_ESCOPO = "fora_escopo"


class Claim(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    text: str = Field(min_length=1, max_length=2000)
    chunk_ids: list[str] = Field(min_length=1, max_length=8)


class GeneratedAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    status: AnswerStatus
    claims: list[Claim] = Field(default_factory=list, max_length=8)
    message: str = Field(default="", max_length=1200)
    limitations: list[str] = Field(default_factory=list, max_length=8)

    @model_validator(mode="after")
    def validate_status(self):
        if self.status == AnswerStatus.RESPONDIDA and not self.claims:
            raise ValueError("Resposta fundamentada exige afirmações com citações.")
        if self.status != AnswerStatus.RESPONDIDA and self.claims:
            raise ValueError("Abstenção/esclarecimento não pode introduzir afirmações normativas.")
        if self.status != AnswerStatus.RESPONDIDA and not self.message:
            raise ValueError("Informe o motivo da abstenção ou o esclarecimento necessário.")
        if self.status == AnswerStatus.RESPONDIDA and self.message:
            raise ValueError("Afirmações da resposta devem estar em claims, com fonte.")
        return self


class RetrievedChunk(BaseModel):
    id: str
    text: str
    metadata: dict
    distance: float | None = Field(default=None, allow_inf_nan=False)
    rerank_score: float | None = Field(default=None, allow_inf_nan=False)

    @property
    def evaluation_context(self) -> str:
        return (
            f"Fonte: {self.metadata['title']}\nLocalização: {self.metadata['section']}\n"
            f"Tipo: {self.metadata['tipo']}\nPublicação: {self.metadata['data_publicacao']}\n"
            f"{self.text}"
        )


class RagResult(BaseModel):
    query: str
    answer: str
    structured: GeneratedAnswer
    chunks: list[RetrievedChunk]
    sources: list[dict]
    collection: str
    base_date: str
    input_tokens: int
    ranking_before: list[str] = Field(default_factory=list)
    ranking_after: list[str] = Field(default_factory=list)

    @property
    def evaluation_answer(self) -> str:
        # Avaliar o texto factual; IDs e URLs de apresentação não são afirmações.
        return "\n".join(claim.text for claim in self.structured.claims) or self.structured.message

    @property
    def evaluation_contexts(self) -> list[str]:
        # Metadados são evidência usada pelo gerador; omiti-los penaliza citações corretas.
        return [chunk.evaluation_context for chunk in self.chunks]

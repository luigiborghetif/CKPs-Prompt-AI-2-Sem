"""Contratos Pydantic v2 da análise de atendimento e dos experimentos."""

from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Tema(StrEnum):
    COLETA_SELETIVA = "coleta_seletiva"
    ECOPONTOS = "ecopontos"
    CATA_BAGULHO = "cata_bagulho"
    DESCARTE = "descarte"
    OUTRO = "outro"


class Jurisdicao(StrEnum):
    SAO_PAULO_CAPITAL = "sao_paulo_capital"
    OUTRA = "outra"
    NAO_INFORMADA = "nao_informada"


class StatusAnalise(StrEnum):
    ORIENTACAO = "orientacao"
    ESCLARECIMENTO = "esclarecimento"
    FORA_ESCOPO = "fora_escopo"
    SEM_BASE = "sem_base"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class AnaliseConsulta(StrictModel):
    """Classifica o atendimento; não transforma a análise em parecer jurídico."""

    tema: Tema
    jurisdicao: Jurisdicao
    resumo_consulta: str = Field(min_length=1, max_length=800)
    tipo_residuo: str | None = Field(default=None, min_length=1, max_length=200)
    volume_m3: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    informacoes_faltantes: list[str] = Field(default_factory=list, max_length=8)
    necessita_esclarecimento: bool = Field(strict=True)
    status: StatusAnalise
    limitacoes: list[str] = Field(default_factory=list, max_length=8)

    @field_validator("informacoes_faltantes", "limitacoes")
    @classmethod
    def validate_lists(cls, values: list[str]) -> list[str]:
        values = [value.strip() for value in values]
        if any(not value for value in values):
            raise ValueError("Listas não podem conter textos vazios.")
        return list(dict.fromkeys(values))

    @model_validator(mode="after")
    def validate_consistency(self) -> Self:
        if self.necessita_esclarecimento != bool(self.informacoes_faltantes):
            raise ValueError("Esclarecimento deve corresponder às informações faltantes.")
        if self.status == StatusAnalise.ESCLARECIMENTO and not self.necessita_esclarecimento:
            raise ValueError("Status esclarecimento exige uma informação faltante.")
        if self.jurisdicao == Jurisdicao.OUTRA and self.status != StatusAnalise.FORA_ESCOPO:
            raise ValueError("Outra jurisdição deve ser classificada como fora do escopo.")
        return self


class RespostaExperimento(StrictModel):
    municipio: str | None
    residuo: str | None
    volume_m3: float | None = Field(ge=0, allow_inf_nan=False)
    codigo: str | None

"""Filtros tipados traduzidos para where= do Chroma."""

import re

from pydantic import BaseModel, ConfigDict, Field


class SearchFilters(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tipo: str | None = Field(default=None, pattern=r"^(norma|orientacao)$")
    categoria: str | None = None
    source_id: str | None = Field(default=None, pattern=r"^SP-\d{3}$")
    section: str | None = Field(default=None, pattern=r"^Art\. \d+$")
    published_from_year: int | None = Field(default=None, ge=1900, le=2100)

    def where(self) -> dict | None:
        conditions = []
        for field in ("tipo", "categoria", "source_id", "section"):
            value = getattr(self, field)
            if value:
                conditions.append({field: {"$eq": value}})
        if self.published_from_year is not None:
            conditions.append({"ano": {"$gte": self.published_from_year}})
        if not conditions:
            return None
        return conditions[0] if len(conditions) == 1 else {"$and": conditions}


def infer_source_filter(query, sources, filters: SearchFilters) -> SearchFilters:
    """Uma norma explicitamente nomeada restringe a busca; não inferir por gabarito."""
    normalized = query.replace(".", "")
    matches = []
    for source in sources:
        reference = re.search(r"(\d[\d.]*)/(\d{4})", source.title)
        if reference:
            number, year = reference.groups()
            if re.search(rf"\b{re.escape(number.replace('.', ''))}\s*/\s*{year}\b", normalized):
                matches.append(source.id)
    if filters.source_id is None and len(matches) == 1:
        filters = filters.model_copy(update={"source_id": matches[0]})
    if filters.source_id and filters.section is None:
        articles = set(re.findall(r"\b(?:artigo|art\.?)[ \t]*(\d+)\b", query.lower()))
        if len(articles) == 1:
            filters = filters.model_copy(update={"section": f"Art. {next(iter(articles))}"})
    return filters

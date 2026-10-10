import pytest
from pydantic import ValidationError

from app.schemas import AnaliseConsulta


def test_valid_and_normalized(analysis_dict):
    analysis_dict["limitacoes"] = [" texto ", "texto"]
    result = AnaliseConsulta.model_validate(analysis_dict)
    assert result.limitacoes == ["texto"]
    assert result.volume_m3 == 0.5


@pytest.mark.parametrize(
    "field,value",
    [
        ("tema", "inexistente"),
        ("volume_m3", -1),
        ("volume_m3", float("nan")),
        ("resumo_consulta", " "),
        ("limitacoes", [" "]),
        ("necessita_esclarecimento", "false"),
        ("artigo_inventado", "1"),
    ],
)
def test_invalid_values_rejected(analysis_dict, field, value):
    analysis_dict[field] = value
    with pytest.raises(ValidationError):
        AnaliseConsulta.model_validate(analysis_dict)


def test_missing_information_requires_clarification(analysis_dict):
    analysis_dict["informacoes_faltantes"] = ["volume"]
    with pytest.raises(ValidationError):
        AnaliseConsulta.model_validate(analysis_dict)
    analysis_dict["necessita_esclarecimento"] = True
    assert AnaliseConsulta.model_validate(analysis_dict).necessita_esclarecimento


def test_other_jurisdiction_cannot_be_orientation(analysis_dict):
    analysis_dict["jurisdicao"] = "outra"
    with pytest.raises(ValidationError):
        AnaliseConsulta.model_validate(analysis_dict)
    analysis_dict["status"] = "fora_escopo"
    assert AnaliseConsulta.model_validate(analysis_dict).status == "fora_escopo"

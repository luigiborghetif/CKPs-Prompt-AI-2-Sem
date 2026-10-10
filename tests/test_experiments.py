import json

import pytest
from conftest import OfflineChatModel

from app.context_rot import TARGET, build_context, run, score_response
from app.meta_prompting import validate_candidate
from app.prompts import SYSTEM_V1
from app.schemas import RespostaExperimento
from evals.run_chat import checks_for


def test_target_retained_and_context_grows():
    from app.tokens import count_text

    contexts = [build_context(turns) for turns in [0, 5, 10, 15, 20]]
    assert all("REGISTRO ALVO:" in context and "SP-ALVO-7319" in context for context in contexts)
    assert all(count_text(a) < count_text(b) for a, b in zip(contexts, contexts[1:]))


def test_quality_uses_independent_ground_truth():
    assert score_response(RespostaExperimento(**TARGET)) == 1
    assert score_response(RespostaExperimento(**{**TARGET, "codigo": "OUTRO"})) == 0.75


def test_experiment_saves_errors_without_inventing_quality(settings, tmp_path):
    llm = OfflineChatModel(responses=[json.dumps(TARGET), "inválido"])
    path = run(settings, windows=[0, 1], repeats=1, destination=tmp_path, llm=llm)
    rows = json.loads((path / "results.json").read_text())
    assert rows[0]["quality"] == 1
    assert rows[1]["quality"] is None
    assert rows[1]["status"] == "error"
    assert (path / "context_rot.png").is_file()
    assert (path / "input_0.json").is_file()


def test_meta_preserves_template_contract():
    assert validate_candidate(SYSTEM_V1) == SYSTEM_V1.strip()
    with pytest.raises(ValueError):
        validate_candidate(SYSTEM_V1.replace("{contexto_oficial}", "{outro}"))
    with pytest.raises(ValueError):
        validate_candidate(SYSTEM_V1.replace("<seguranca>", "<outra>"))


def test_failed_experiment_does_not_claim_stable_quality(settings, tmp_path):
    path = run(
        settings,
        windows=[0],
        repeats=1,
        destination=tmp_path,
        llm=OfflineChatModel(responses=["inválido"]),
    )
    conclusion = (path / "conclusion.md").read_text()
    assert "nenhuma janela teve execução válida" in conclusion
    assert "Não houve degradação" not in conclusion


def test_eval_checks_are_explicit(analysis_dict):
    checks = checks_for(
        "São Paulo, entulho, 0,5 m³.",
        analysis_dict,
        {
            "volume_m3": 0.5,
            "answer_contains_all": ["sao paulo", "entulho"],
            "answer_not_contains": ["multa de R$"],
        },
    )
    assert all(checks.values())

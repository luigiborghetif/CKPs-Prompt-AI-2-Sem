"""Modelo propõe uma revisão e as duas versões recebem as mesmas consultas."""

import argparse
from pathlib import Path
from string import Formatter

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.artifacts import metadata, run_directory, write_json
from app.chain import create_llm
from app.config import load_settings
from app.prompts import META_SYSTEM, SYSTEM_V1
from app.service import public_error
from evals.run_chat import evaluate, summary


def validate_candidate(candidate: str) -> str:
    candidate = candidate.strip()
    fields = {field for _, field, _, _ in Formatter().parse(candidate) if field is not None}
    if fields != {"contexto_oficial"}:
        raise ValueError("A proposta alterou os placeholders obrigatórios.")
    for section in ("identidade", "escopo", "fontes", "conversa", "seguranca", "contexto_oficial"):
        if f"<{section}>" not in candidate or f"</{section}>" not in candidate:
            raise ValueError(f"A proposta removeu a seção {section}.")
    if "```" in candidate or len(candidate) > 16000:
        raise ValueError("Formato da proposta inválido.")
    return candidate


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", type=Path)
    cli.add_argument(
        "--generate-only", action="store_true", help="Salva proposta sem avaliar v1/v2."
    )
    args = cli.parse_args()
    try:
        settings = load_settings()
        llm = create_llm(settings)
        path = run_directory("meta_prompting", args.output)
        write_json(path / "metadata.json", metadata(settings))
        (path / "prompt_v1.md").write_text(SYSTEM_V1, encoding="utf-8")
        prompt = ChatPromptTemplate.from_messages(
            [("system", META_SYSTEM), ("human", "<prompt_original>{original}</prompt_original>")]
        )
        candidate = (prompt | llm | StrOutputParser()).invoke({"original": SYSTEM_V1})
        (path / "proposal_raw.md").write_text(candidate, encoding="utf-8")
        candidate = validate_candidate(candidate)
        (path / "prompt_v2.md").write_text(candidate, encoding="utf-8")
        write_json(path / "request.json", {"system": META_SYSTEM, "original": SYSTEM_V1})
        if not args.generate_only:
            before = evaluate(settings, system=SYSTEM_V1)
            after = evaluate(settings, system=candidate)
            write_json(path / "v1_results.json", before)
            write_json(path / "v2_results.json", after)
            write_json(path / "comparison.json", {"v1": summary(before), "v2": summary(after)})
        (path / "review.md").write_text(
            "# Revisão necessária\n\nA proposta não foi ativada na aplicação. "
            "Compare regras, resultados, latência e regressões antes de adotá-la. "
            "A validação de placeholders/XML não garante preservação semântica das regras.\n",
            encoding="utf-8",
        )
        print(f"Proposta e evidências: {path}")
        if not args.generate_only and any(item["status"] == "error" for item in before + after):
            raise SystemExit(
                "Comparação salva com erros; confira os resultados e repita após corrigir a conexão."
            )
    except Exception as exc:
        raise SystemExit(public_error(exc)) from None


if __name__ == "__main__":
    main()

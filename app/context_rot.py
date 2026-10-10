"""Experimento real: pergunta fixa, fato preservado e contextos crescentes."""

import argparse
import hashlib
from pathlib import Path
from statistics import mean
from time import perf_counter

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.artifacts import metadata, run_directory, write_csv, write_json
from app.chain import create_llm
from app.config import Settings, load_settings
from app.prompts import EXPERIMENT_QUESTION, EXPERIMENT_SYSTEM
from app.schemas import RespostaExperimento
from app.service import public_error
from app.tokens import count_messages, count_text

TARGET = {
    "municipio": "São Paulo capital",
    "residuo": "entulho",
    "volume_m3": 0.5,
    "codigo": "SP-ALVO-7319",
}


def build_context(turns: int) -> str:
    """Fixtures experimentais, sem uso como base jurídica do assistente."""
    if turns < 0:
        raise ValueError("A janela não pode ser negativa.")
    target = "REGISTRO ALVO: município São Paulo capital; resíduo entulho; volume 0,5 m³; código SP-ALVO-7319.\n"
    distraction = []
    for index in range(turns):
        distraction.append(
            f"REGISTRO DE DISTRAÇÃO {index + 1}: município Cidade Exemplo {index + 1}; "
            f"resíduo papel; volume {index + 2} m³; código OUTRO-{index + 1}. "
            "Usuário: quero organizar um atendimento de teste diferente do registro alvo. "
            "Assistente: este registro fictício serve somente para aumentar o contexto experimental. "
            + "Há informações auxiliares de agendamento, conferência e organização de registros. "
            * 12
            + "\n"
        )
    return target + "".join(distraction)


def score_response(response: RespostaExperimento) -> float:
    return sum(getattr(response, key) == value for key, value in TARGET.items()) / len(TARGET)


def run(
    settings: Settings,
    *,
    windows=(0, 5, 10, 15, 20),
    repeats=3,
    destination: Path | None = None,
    llm=None,
) -> Path:
    if repeats < 1 or not windows or any(window < 0 for window in windows):
        raise ValueError("Use janelas não negativas e pelo menos uma repetição.")
    llm = llm if llm is not None else create_llm(settings, structured=True)
    parser = PydanticOutputParser(pydantic_object=RespostaExperimento)
    prompt = ChatPromptTemplate.from_messages(
        [("system", EXPERIMENT_SYSTEM), ("human", "{pergunta}")]
    ).partial(format_instructions=parser.get_format_instructions())
    pipeline = prompt | llm
    path = run_directory("context_rot", destination)
    info = metadata(settings)
    info.update(
        {
            "windows": list(windows),
            "repeats": repeats,
            "target": TARGET,
            "question": EXPERIMENT_QUESTION,
            "system_prompt": EXPERIMENT_SYSTEM,
            "method": "Sem memória/poda; registro alvo presente em todas as janelas.",
        }
    )
    write_json(path / "metadata.json", info)
    records, rows = [], []
    for window in windows:
        context = build_context(window)
        args = {"contexto": context, "pergunta": EXPERIMENT_QUESTION}
        messages = prompt.invoke(args).to_messages()
        write_json(
            path / f"input_{window}.json", [message.model_dump(mode="json") for message in messages]
        )
        for repeat in range(1, repeats + 1):
            start = perf_counter()
            record = {
                "window": window,
                "repeat": repeat,
                "context_tokens": count_text(context),
                "input_tokens": count_messages(messages),
                "output_tokens": 0,
                "quality": None,
                "status": "error",
                "error": None,
                "response": None,
                "input_sha256": hashlib.sha256(context.encode()).hexdigest(),
            }
            try:
                raw = pipeline.invoke(args)
                record["response"] = str(raw.content)
                record["output_tokens"] = count_text(str(raw.content))
                record["quality"] = score_response(parser.invoke(raw))
                record["status"] = "ok"
            except Exception as exc:
                record["error"] = public_error(exc)
            record["latency_seconds"] = round(perf_counter() - start, 4)
            records.append(record)
            rows.append({key: value for key, value in record.items() if key != "response"})
            write_json(path / "results.json", records)
    write_csv(path / "results.csv", rows)
    summaries = []
    for window in windows:
        group = [row for row in rows if row["window"] == window]
        valid = [row["quality"] for row in group if row["status"] == "ok"]
        summaries.append(
            {
                "window": window,
                "input_tokens": group[0]["input_tokens"],
                "quality_valid_mean": mean(valid) if valid else None,
                "successful_runs": len(valid),
                "errors": len(group) - len(valid),
                "latency_mean_seconds": mean(row["latency_seconds"] for row in group),
            }
        )
    write_csv(path / "summary.csv", summaries)
    write_json(path / "summary.json", summaries)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    valid = [item for item in summaries if item["quality_valid_mean"] is not None]
    ax.plot(
        [item["input_tokens"] for item in valid],
        [item["quality_valid_mean"] for item in valid],
        "o-",
    )
    ax.set(
        xlabel="Tokens de entrada estimados (cl100k_base)",
        ylabel="Fração de fatos corretos",
        title="Contexto crescente: qualidade observada",
        ylim=(-0.05, 1.05),
    )
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path / "context_rot.png", dpi=180)
    plt.close(fig)
    note = "Não houve degradação observada nas execuções válidas."
    if not valid:
        note = "Não foi possível medir a qualidade: nenhuma janela teve execução válida."
    elif len(valid) < 2:
        note = "Não há duas janelas válidas para comparar degradação de qualidade."
    elif valid[-1]["quality_valid_mean"] < valid[0]["quality_valid_mean"]:
        note = "Houve queda de qualidade entre a primeira e a última janela válida."
    (path / "conclusion.md").write_text(
        f"# Resultado observado\n\n{note}\n\nErros de execução: {sum(item['errors'] for item in summaries)}. "
        "A média considera somente execuções válidas, com contagem de falhas explícita. "
        "Ausência de queda não comprova ausência de context rot em outros cenários.\n",
        encoding="utf-8",
    )
    return path


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--windows", type=int, nargs="+", default=[0, 5, 10, 15, 20])
    cli.add_argument("--repeats", type=int, default=3)
    cli.add_argument("--output", type=Path)
    args = cli.parse_args()
    try:
        path = run(
            load_settings(), windows=args.windows, repeats=args.repeats, destination=args.output
        )
    except Exception as exc:
        raise SystemExit(public_error(exc)) from None
    print(f"Evidências: {path}")
    import json

    if any(item["status"] != "ok" for item in json.loads((path / "results.json").read_text())):
        raise SystemExit(
            "Experimento salvo com falhas. Confira results.json e repita após corrigir a conexão."
        )


if __name__ == "__main__":
    main()

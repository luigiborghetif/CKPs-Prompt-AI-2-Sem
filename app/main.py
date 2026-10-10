"""Interface local: python -m app.main. Nenhuma chamada no startup."""

import os
from uuid import uuid4

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr

from app.config import Settings, load_settings
from app.service import Assistant, ConsultationError


def handle_query(query, history, session_id, assistant: Assistant):
    history = list(history or [])
    if not query or not query.strip():
        return query, history, None, "Escreva uma pergunta."
    try:
        result = assistant.consult(query, session_id)
    except ConsultationError as exc:
        return query, history, None, str(exc)
    history.extend(
        [
            {"role": "user", "content": query.strip()},
            {"role": "assistant", "content": result.answer},
        ]
    )
    # O histórico visual tem limite próprio; a memória do modelo é TokenBuffer.
    history = history[-100:]
    return (
        "",
        history,
        result.analysis.model_dump(mode="json"),
        f"Histórico do modelo: ~{result.history_tokens} tokens.",
    )


def build_app(settings: Settings | None = None, assistant: Assistant | None = None) -> gr.Blocks:
    settings = settings or load_settings()
    assistant = assistant or Assistant(settings)
    with gr.Blocks(title="Resíduos SP · Assistente") as demo:
        gr.Markdown(
            "# Resíduos SP\nConsulta às normas e orientações de descarte em São Paulo, capital."
        )
        gr.Markdown(
            "Base inicial do CKP01: coleta seletiva, Ecopontos e Cata-Bagulho. "
            "As respostas usam os trechos oficiais fornecidos ao assistente."
        )
        session_id = gr.State(value=lambda: uuid4().hex)
        with gr.Row():
            with gr.Column(scale=3):
                chat = gr.Chatbot(label="Conversa", height=460, sanitize_html=True)
                query = gr.Textbox(
                    label="Sua pergunta",
                    placeholder="Como funciona a entrega em Ecopontos?",
                    lines=2,
                )
                with gr.Row():
                    send = gr.Button("Enviar", variant="primary")
                    reset = gr.Button("Nova conversa")
            with gr.Column(scale=2):
                analysis = gr.JSON(label="Análise estruturada do atendimento")
                status = gr.Markdown(
                    "Pronto para conversar."
                    if settings.configured
                    else "Configure sua chave em `.env` e reinicie para enviar consultas."
                )
                gr.Markdown(
                    "A análise identifica tema, município, resíduo, volume e informações faltantes."
                )
        gr.Examples(
            examples=[
                ["Qual é a diferença entre Cata-Bagulho e Ecoponto?"],
                ["Estou em São Paulo capital e tenho 0,5 m³ de entulho de uma reforma."],
                ["Você tem os horários da coleta na minha rua?"],
            ],
            inputs=query,
        )

        def submit(message, messages, sid):
            return handle_query(message, messages, sid, assistant)

        def clear(sid):
            assistant.clear(sid)
            return "", [], None, "Nova conversa iniciada."

        inputs = [query, chat, session_id]
        outputs = [query, chat, analysis, status]
        # Serializar ações da UI evita respostas/limpezas chegando fora de ordem.
        send.click(submit, inputs, outputs, concurrency_id="chat", concurrency_limit=1)
        query.submit(submit, inputs, outputs, concurrency_id="chat", concurrency_limit=1)
        reset.click(clear, [session_id], outputs, concurrency_id="chat", concurrency_limit=1)
    return demo


def main() -> None:
    import argparse
    from pathlib import Path

    from app.meta_prompting import validate_candidate

    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--system-prompt", type=Path, help="Proposta de meta prompting revisada.")
    args = cli.parse_args()
    try:
        settings = load_settings()
        assistant = (
            Assistant(settings)
            if args.system_prompt is None
            else Assistant(
                settings, system=validate_candidate(args.system_prompt.read_text(encoding="utf-8"))
            )
        )
    except ValueError:
        raise SystemExit(
            "Configuração inválida no .env. Confira os valores e .env.example."
        ) from None
    build_app(settings, assistant).queue().launch(
        server_name=settings.host,
        server_port=settings.port,
        share=False,
        inbrowser=False,
        show_error=False,
        theme=gr.themes.Soft(),
    )


if __name__ == "__main__":
    main()

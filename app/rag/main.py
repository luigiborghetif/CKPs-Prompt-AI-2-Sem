"""Interface Gradio RAG; nenhum download/chamada ao carregar a página."""

import os

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr

from app.config import load_settings
from app.rag.config import load_rag_settings
from app.rag.loaders import load_sources
from app.rag.pipeline import RagPipeline
from app.rag.retrieval import SearchFilters
from app.service import public_error


def build_app(pipeline: RagPipeline | None = None):
    if pipeline is None:
        settings = load_settings()
        pipeline = RagPipeline(settings, load_rag_settings(settings))
    sources = load_sources(pipeline.config.manifest, pipeline.root)
    categories = sorted({source.category for source in sources})
    with gr.Blocks(title="DocMind · Resíduos SP") as demo:
        gr.Markdown(
            "# DocMind · Resíduos SP\nConsulta fundamentada em cinco normas municipais de São Paulo, capital."
        )
        gr.Markdown(
            "Cada afirmação informa a fonte e o trecho recuperado. "
            "A base é um recorte documental e não consulta atualizações em tempo real."
        )
        with gr.Row():
            tipo = gr.Dropdown(
                ["Todos", "norma", "orientacao"], value="Todos", label="Tipo de documento"
            )
            category = gr.Dropdown(["Todas", *categories], value="Todas", label="Tema")
            year = gr.Number(value=None, precision=0, label="Publicado a partir do ano (opcional)")
        query = gr.Textbox(
            label="Pergunta", lines=2, placeholder="Qual é o limite por descarga em um Ecoponto?"
        )
        send = gr.Button("Consultar", variant="primary")
        answer = gr.Markdown()
        with gr.Accordion("Fontes e trechos recuperados", open=False):
            citations = gr.JSON(label="Fontes citadas")
            contexts = gr.JSON(label="Trechos recuperados")
        status = gr.Markdown(
            "Pronto. A primeira consulta constrói ou reutiliza o índice configurado."
        )
        with gr.Accordion("Documentos da base", open=False):
            gr.Markdown(
                "\n\n".join(
                    f"- [{source.title}]({source.url}) — {source.excerpt}" for source in sources
                )
            )

        def submit(question, document_type, topic, published_year):
            try:
                filters = SearchFilters(
                    tipo=None if document_type == "Todos" else document_type,
                    categoria=None if topic == "Todas" else topic,
                    published_from_year=published_year,
                )
                result = pipeline.buscar(question, filters)
                return (
                    result.answer,
                    result.sources,
                    [item.model_dump() for item in result.chunks],
                    f"Base coletada em {result.base_date}. {len(result.chunks)} trechos; ~{result.input_tokens} tokens de entrada.",
                )
            except Exception as exc:
                return "", [], [], public_error(exc)

        inputs = [query, tipo, category, year]
        outputs = [answer, citations, contexts, status]
        send.click(submit, inputs, outputs, concurrency_limit=1, concurrency_id="rag")
        query.submit(submit, inputs, outputs, concurrency_limit=1, concurrency_id="rag")
    return demo


def main():
    settings = load_settings()
    build_app().queue().launch(
        server_name=settings.host,
        server_port=settings.port,
        share=False,
        inbrowser=False,
        show_error=False,
        theme=gr.themes.Soft(),
    )


if __name__ == "__main__":
    main()

"""Checagem opcional de chat, embeddings, geração e RAGAS pelo usuário."""

import json

from app.artifacts import metadata, write_json
from app.config import ROOT, load_settings
from app.rag.config import load_rag_settings
from app.rag.embeddings import create_embeddings
from app.rag.pipeline import RagPipeline
from app.service import public_error


def main():
    try:
        settings = load_settings()
        config = load_rag_settings(settings).model_copy(update={"rerank": False})
        embeddings = create_embeddings(settings, config)
        query_vector = embeddings.embed_query("coleta seletiva")
        document_vectors = embeddings.embed_documents(
            ["Coleta seletiva de resíduos recicláveis secos."]
        )
        if len(query_vector) != len(document_vectors[0]):
            raise ValueError("Dimensões incompatíveis.")
        print(f"Embeddings nomic-embed-text: {len(query_vector)} dimensões.")
        result = RagPipeline(settings, config, embeddings=embeddings).buscar(
            "Segundo o Decreto 55.113/2014, qual o limite por descarga em um Ecoponto?"
        )
        print("Pipeline e validação de citações concluídos.")
        from app.rag.evaluation import RagasEvaluator

        evaluator = RagasEvaluator(settings, config, embeddings=embeddings)
        try:
            scores = evaluator.score(
                result.query,
                result.evaluation_answer,
                result.evaluation_contexts,
                "O artigo 3 estabelece limite de 1 m³ por descarga.",
            )
        finally:
            evaluator.close()
        report = metadata(settings)
        report.update(
            {
                "status": "ok",
                "embedding_model": config.embedding_model,
                "embedding_base_url": config.embedding_base_url,
                "embedding_dimensions": len(query_vector),
                "embed_query": "ok",
                "embed_documents": "ok",
                "pipeline": result.model_dump(mode="json"),
                "ragas": scores,
                "reranking": "not_tested_in_connection_check",
            }
        )
        path = ROOT / "evidence/ckp02/connection_check.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, report)
        print(json.dumps(scores, ensure_ascii=False, indent=2))
    except Exception as exc:
        raise SystemExit(public_error(exc)) from None


if __name__ == "__main__":
    main()

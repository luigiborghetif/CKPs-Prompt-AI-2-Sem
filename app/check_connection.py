"""Verificação opcional de conexão; só executa quando chamada explicitamente."""

from uuid import uuid4

from app.config import load_settings
from app.service import Assistant, ConsultationError, public_error


def main() -> None:
    try:
        settings = load_settings()
        if not settings.configured:
            raise SystemExit("Preencha OLLAMA_API_KEY no .env antes de testar a conexão.")
        assistant = Assistant(settings)
        result = assistant.consult(
            "Estou em São Paulo capital. Explique brevemente o que é um Ecoponto, conforme sua base.",
            uuid4().hex,
        )
        print("Conexão, ConversationChain e análise LCEL/Pydantic executadas com sucesso.")
        print(
            f"Modelo lógico: {settings.model}; identificador enviado: {settings.effective_model}."
        )
        print(f"Histórico estimado: {result.history_tokens} tokens.")
    except ConsultationError as exc:
        raise SystemExit(str(exc)) from None
    except Exception as exc:
        raise SystemExit(public_error(exc)) from None


if __name__ == "__main__":
    main()

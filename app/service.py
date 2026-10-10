"""Orquestra duas chains com atualização transacional da memória."""

from dataclasses import dataclass
from threading import RLock

import httpx
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models.chat_models import BaseChatModel
from ollama import ResponseError
from pydantic import ValidationError

from app.chain import (
    MissingConfiguration,
    create_analysis,
    create_conversation,
    create_llm,
    load_context,
)
from app.config import Settings
from app.memory_manager import MemoryManager
from app.prompts import SYSTEM_V1
from app.schemas import AnaliseConsulta
from app.tokens import count_messages


class ConsultationError(RuntimeError):
    """Erro público sem detalhes internos do serviço ou credenciais."""


def public_error(exc: Exception) -> str:
    if isinstance(exc, MissingConfiguration):
        return str(exc)
    if isinstance(exc, (OutputParserException, ValidationError)):
        return "A análise não passou na validação. O turno não foi salvo; tente novamente."
    if isinstance(exc, ResponseError):
        code = getattr(exc, "status_code", None)
        if code in {401, 403}:
            return "O Ollama recusou a autenticação. Confira sua chave e o acesso ao modelo."
        if code == 404:
            return "Modelo ou endpoint não encontrado. Confira OLLAMA_BASE_URL e OLLAMA_API_MODEL no .env."
        if code == 429:
            return "O Ollama atingiu um limite de uso. Aguarde e tente novamente."
        return "O Ollama não concluiu a consulta. Confira a configuração e tente novamente."
    if isinstance(exc, httpx.TimeoutException):
        return "A consulta excedeu o tempo configurado. Tente novamente."
    if isinstance(exc, (httpx.HTTPError, ConnectionError)):
        return "Não foi possível alcançar o Ollama. Confira endpoint e conexão."
    return "Não foi possível concluir a consulta. Verifique a configuração e tente novamente."


@dataclass(frozen=True)
class Consultation:
    answer: str
    analysis: AnaliseConsulta
    history_tokens: int


class Assistant:
    def __init__(
        self,
        settings: Settings,
        *,
        chat_llm: BaseChatModel | None = None,
        analysis_llm: BaseChatModel | None = None,
        system: str = SYSTEM_V1,
    ):
        self.settings = settings
        self.system = system
        self.context = load_context()
        self._chat_llm = chat_llm
        self._analysis_llm = analysis_llm
        self._memory: MemoryManager | None = None
        self._analysis = None
        self._init_lock = RLock()

    def _initialize(self) -> None:
        # Carregar a UI não cria clientes nem exige conexão/chave.
        with self._init_lock:
            if self._memory is not None:
                return
            chat = self._chat_llm if self._chat_llm is not None else create_llm(self.settings)
            analysis = (
                self._analysis_llm
                if self._analysis_llm is not None
                else create_llm(self.settings, structured=True)
            )
            self._analysis = create_analysis(analysis)
            self._chat_llm = chat
            self._memory = MemoryManager(chat, self.settings)

    def consult(self, query: str, session_id: str) -> Consultation:
        query = query.strip()
        if not query:
            raise ConsultationError("Escreva uma pergunta antes de enviar.")
        if len(query) > self.settings.max_input_chars:
            raise ConsultationError(
                f"Use até {self.settings.max_input_chars} caracteres por pergunta."
            )
        try:
            self._initialize()
            with self._memory.session(session_id) as session:
                snapshot = list(session.memory.chat_memory.messages)
                history = session.memory.load_memory_variables({})["history"]
                try:
                    conversation = create_conversation(
                        self._chat_llm, session.memory, self.context, self.system
                    )
                    answer = conversation.invoke({"input": query})["response"].strip()
                    if not answer:
                        raise ValueError("Resposta vazia.")
                    analysis = self._analysis.invoke(
                        {"consulta": query, "resposta": answer, "historico": history}
                    )
                    return Consultation(
                        answer, analysis, count_messages(session.memory.chat_memory.messages)
                    )
                except Exception:
                    # Se a segunda chain falhar, não conservar um turno invisível ao usuário.
                    session.memory.chat_memory.messages = snapshot
                    raise
        except ValueError as exc:
            if isinstance(exc, (MissingConfiguration, OutputParserException, ValidationError)):
                raise ConsultationError(public_error(exc)) from None
            raise ConsultationError(
                "Não foi possível processar o turno. Confira a entrada ou tente novamente."
            ) from None
        except Exception as exc:
            raise ConsultationError(public_error(exc)) from None

    def clear(self, session_id: str) -> None:
        if self._memory is not None:
            self._memory.clear(session_id)

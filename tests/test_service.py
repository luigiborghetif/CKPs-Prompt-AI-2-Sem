import json

import httpx
import pytest
from conftest import OfflineChatModel

from app.chain import create_llm
from app.config import Settings
from app.service import Assistant, ConsultationError, public_error


def test_two_chains_and_one_history_update(settings, models):
    assistant = Assistant(settings, chat_llm=models[0], analysis_llm=models[1])
    result = assistant.consult("Tenho entulho em São Paulo.", "A")
    assert result.analysis.tema == "ecopontos"
    assert result.answer
    assert result.history_tokens > 0
    with assistant._memory.session("A") as session:
        assert len(session.memory.chat_memory.messages) == 2


def test_parser_failure_rolls_back_and_retry_does_not_duplicate(settings, analysis_dict):
    chat = OfflineChatModel(responses=["resposta"])
    analysis = OfflineChatModel(responses=["não é JSON", json.dumps(analysis_dict)])
    assistant = Assistant(settings, chat_llm=chat, analysis_llm=analysis)
    with pytest.raises(ConsultationError, match="validação"):
        assistant.consult("Consulta", "A")
    assert assistant._memory.tokens("A") == 0
    assistant.consult("Consulta", "A")
    with assistant._memory.session("A") as session:
        assert len(session.memory.chat_memory.messages) == 2


def test_history_reaches_conversation_and_analysis(settings, analysis_dict):
    from langchain_core.callbacks import BaseCallbackHandler

    class Capture(BaseCallbackHandler):
        def __init__(self):
            self.prompts = []

        def on_chat_model_start(self, serialized, messages, **kwargs):
            self.prompts.append("\n".join(str(m.content) for m in messages[0]))

    chat_capture, analysis_capture = Capture(), Capture()
    chat = OfflineChatModel(responses=["resposta"], callbacks=[chat_capture])
    analysis = OfflineChatModel(responses=[json.dumps(analysis_dict)], callbacks=[analysis_capture])
    assistant = Assistant(settings, chat_llm=chat, analysis_llm=analysis)
    assistant.consult("O código do meu caso é CASO-7319.", "A")
    assistant.consult("Retome meu caso.", "A")
    assert "CASO-7319" in chat_capture.prompts[-1]
    assert "CASO-7319" in analysis_capture.prompts[-1]
    assistant.consult("Outra sessão.", "B")
    assert "CASO-7319" not in chat_capture.prompts[-1]


def test_missing_key_is_lazy_and_clear_does_not_initialize(settings):
    assistant = Assistant(settings)
    assistant.clear("A")
    assert assistant._memory is None
    with pytest.raises(ConsultationError, match="OLLAMA_API_KEY"):
        assistant.consult("Olá", "A")
    assert assistant._memory is None


@pytest.mark.parametrize("query", ["", "  ", "x" * 4001])
def test_invalid_input_does_not_initialize(settings, query):
    assistant = Assistant(settings)
    with pytest.raises(ConsultationError):
        assistant.consult(query, "A")
    assert assistant._memory is None


def test_errors_do_not_expose_details():
    secret = "sk-private-secret"
    assert secret not in public_error(httpx.ConnectError(secret))
    assert secret not in public_error(RuntimeError(secret))


def test_client_can_be_built_without_network_and_uses_local_counter(monkeypatch):
    import socket

    def forbid(*args, **kwargs):
        raise AssertionError("Tentativa de conexão externa no teste offline")

    monkeypatch.setattr(socket.socket, "connect", forbid)
    llm = create_llm(Settings(api_key="test-only"), structured=True)
    assert llm.model == "gemma4:cloud"
    assert llm.format == "json"
    assert llm.get_num_tokens_from_messages([]) == 0

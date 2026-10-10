"""Modelos determinísticos somente para testes offline, nunca usados pela UI."""

import json

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from app.config import Settings
from app.tokens import count_messages


class OfflineChatModel(FakeListChatModel):
    def get_num_tokens_from_messages(self, messages, tools=None):
        return count_messages(messages)


@pytest.fixture
def settings():
    return Settings()


@pytest.fixture
def analysis_dict():
    return {
        "tema": "ecopontos",
        "jurisdicao": "sao_paulo_capital",
        "resumo_consulta": "Consulta de entulho em São Paulo capital.",
        "tipo_residuo": "entulho",
        "volume_m3": 0.5,
        "informacoes_faltantes": [],
        "necessita_esclarecimento": False,
        "status": "orientacao",
        "limitacoes": ["Base inicial sem atualização em tempo real."],
    }


@pytest.fixture
def models(analysis_dict):
    return OfflineChatModel(responses=["Orientação de teste sobre Ecopontos."]), OfflineChatModel(
        responses=[json.dumps(analysis_dict)]
    )

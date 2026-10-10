"""Arquitetura obrigatória: ConversationChain e segunda chain LCEL tipada."""

from pathlib import Path

from langchain_classic.chains import ConversationChain
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_ollama import ChatOllama

from app.config import ROOT, Settings
from app.prompts import ANALYSIS_HUMAN, ANALYSIS_SYSTEM, SYSTEM_V1
from app.schemas import AnaliseConsulta
from app.tokens import count_messages, token_ids


class EstimatedChatOllama(ChatOllama):
    """Contador explícito para TokenBuffer, sem tokenizer remoto implícito."""

    def get_num_tokens_from_messages(self, messages, tools=None) -> int:
        return count_messages(messages)


class MissingConfiguration(ValueError):
    """A conexão só é exigida ao iniciar uma operação que usa o modelo."""


def create_llm(settings: Settings, *, structured: bool = False) -> ChatOllama:
    if not settings.configured:
        raise MissingConfiguration("Copie .env.example para .env e preencha OLLAMA_API_KEY.")
    return EstimatedChatOllama(
        model=settings.effective_model,
        base_url=settings.base_url,
        temperature=settings.temperature,
        num_predict=settings.max_output_tokens,
        format="json" if structured else "",
        validate_model_on_init=False,
        custom_get_token_ids=token_ids,
        client_kwargs={
            "headers": {"Authorization": f"Bearer {settings.api_key.get_secret_value()}"},
            "timeout": settings.timeout,
        },
    )


def load_context(path: Path | None = None) -> str:
    text = (path or ROOT / "data" / "contexto_oficial.md").read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError("O contexto oficial não pode estar vazio.")
    return text


def conversation_prompt(context: str, system: str = SYSTEM_V1) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "<historico>{history}</historico>\n<consulta>{input}</consulta>"),
        ]
    ).partial(contexto_oficial=context)


def create_conversation(
    llm: BaseChatModel, memory, context: str, system: str = SYSTEM_V1
) -> ConversationChain:
    return ConversationChain(
        llm=llm,
        memory=memory,
        prompt=conversation_prompt(context, system),
        verbose=False,
    )


def create_analysis(llm: BaseChatModel) -> Runnable:
    parser = PydanticOutputParser(pydantic_object=AnaliseConsulta)
    prompt = ChatPromptTemplate.from_messages(
        [("system", ANALYSIS_SYSTEM), ("human", ANALYSIS_HUMAN)]
    ).partial(format_instructions=parser.get_format_instructions())
    return prompt | llm | parser

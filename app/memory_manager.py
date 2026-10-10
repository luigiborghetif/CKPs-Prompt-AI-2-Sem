"""TokenBuffer gerenciado, isolado por sessão e com expiração."""

from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import RLock
from time import monotonic

from langchain_classic.memory import ConversationTokenBufferMemory
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage

from app.config import Settings
from app.tokens import count_messages


class TokenBuffer(ConversationTokenBufferMemory):
    """Usa poda da biblioteca e remove resposta órfã após a poda."""

    def save_context(self, inputs, outputs) -> None:
        super().save_context(inputs, outputs)
        while self.chat_memory.messages and isinstance(self.chat_memory.messages[0], AIMessage):
            self.chat_memory.messages.pop(0)


@dataclass
class Session:
    memory: TokenBuffer
    lock: RLock = field(default_factory=RLock)
    touched: float = field(default_factory=monotonic)
    active: int = 0


class MemoryManager:
    def __init__(self, llm: BaseChatModel, settings: Settings):
        self.llm = llm
        self.settings = settings
        self._sessions: dict[str, Session] = {}
        self._lock = RLock()

    def _new_memory(self) -> TokenBuffer:
        return TokenBuffer(
            llm=self.llm,
            max_token_limit=self.settings.memory_tokens,
            memory_key="history",
            input_key="input",
            output_key="response",
            return_messages=False,
        )

    @contextmanager
    def session(self, session_id: str):
        if not session_id.strip():
            raise ValueError("Sessão inválida.")
        with self._lock:
            now = monotonic()
            expired = [
                key
                for key, item in self._sessions.items()
                if item.active == 0 and now - item.touched > self.settings.session_ttl_seconds
            ]
            for key in expired:
                del self._sessions[key]
            if session_id not in self._sessions:
                if len(self._sessions) >= self.settings.max_sessions:
                    raise ValueError(
                        "Limite de sessões atingido. Aguarde a expiração ou limpe uma conversa."
                    )
                self._sessions[session_id] = Session(self._new_memory())
            item = self._sessions[session_id]
            item.active += 1
        try:
            with item.lock:
                yield item
        finally:
            with self._lock:
                item.active -= 1
                item.touched = monotonic()

    def clear(self, session_id: str) -> None:
        # Não remover a sessão enquanto um turno estiver em execução.
        with self._lock:
            item = self._sessions.get(session_id)
            if item is None:
                return
            item.active += 1
        try:
            with item.lock:
                item.memory.clear()
        finally:
            with self._lock:
                item.active -= 1
                item.touched = monotonic()

    def tokens(self, session_id: str) -> int:
        with self.session(session_id) as item:
            return count_messages(item.memory.chat_memory.messages)

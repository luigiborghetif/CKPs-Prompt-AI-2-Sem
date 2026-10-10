from langchain_core.messages import HumanMessage

from app.memory_manager import MemoryManager
from app.tokens import count_messages


def test_retains_six_turns_and_separates_sessions(models, settings):
    manager = MemoryManager(models[0], settings)
    with manager.session("A") as session:
        for turn in range(6):
            session.memory.save_context({"input": f"Fato {turn}"}, {"response": f"Resposta {turn}"})
        assert len(session.memory.chat_memory.messages) == 12
        assert "Fato 0" in session.memory.load_memory_variables({})["history"]
    with manager.session("B") as session:
        assert not session.memory.chat_memory.messages
    manager.clear("A")
    assert manager.tokens("A") == 0


def test_limit_prunes_old_turns_without_orphan_response(models, settings):
    manager = MemoryManager(models[0], settings)
    with manager.session("A") as session:
        for turn in range(20):
            session.memory.save_context(
                {"input": f"Turno {turn}: " + "registro " * 120}, {"response": "resposta " * 80}
            )
        messages = session.memory.chat_memory.messages
        assert count_messages(messages) <= settings.memory_tokens
        assert isinstance(messages[0], HumanMessage)
        assert "Turno 0:" not in session.memory.buffer
        assert "Turno 19:" in session.memory.buffer


def test_session_expiration_and_capacity(models, settings):
    from dataclasses import replace

    import pytest

    manager = MemoryManager(models[0], settings.model_copy(update={"max_sessions": 1}))
    with manager.session("A"):
        pass
    with pytest.raises(ValueError):
        with manager.session("B"):
            pass
    manager._sessions["A"] = replace(manager._sessions["A"], touched=0)
    with manager.session("B"):
        assert "A" not in manager._sessions

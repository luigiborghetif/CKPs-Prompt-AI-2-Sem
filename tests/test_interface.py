from app.main import build_app, handle_query
from app.service import Assistant


def test_ui_builds_without_key_and_does_not_initialize_client(settings):
    assistant = Assistant(settings)
    app = build_app(settings, assistant)
    assert app.blocks
    assert assistant._memory is None


def test_ui_error_retains_input_and_history(settings):
    assistant = Assistant(settings)
    history = [{"role": "user", "content": "anterior"}]
    query, messages, analysis, status = handle_query("Consulta", history, "A", assistant)
    assert query == "Consulta"
    assert messages == history
    assert analysis is None
    assert "OLLAMA_API_KEY" in status


def test_ui_success_and_reset(settings, models):
    assistant = Assistant(settings, chat_llm=models[0], analysis_llm=models[1])
    query, messages, analysis, status = handle_query("Consulta", [], "A", assistant)
    assert query == ""
    assert [message["role"] for message in messages] == ["user", "assistant"]
    assert analysis["tema"] == "ecopontos"
    assert "tokens" in status
    assistant.clear("A")
    assert assistant._memory.tokens("A") == 0

import pytest
from pydantic import ValidationError

from app.config import Settings, load_settings


def test_env_precedence_and_secret_representation(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("OLLAMA_API_KEY=private-value\nMEMORY_TOKEN_LIMIT=900\n")
    monkeypatch.setenv("MEMORY_TOKEN_LIMIT", "1100")
    settings = load_settings(env)
    assert settings.memory_tokens == 1100
    assert "private-value" not in repr(settings)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"memory_tokens": 1600},
        {"base_url": "http://remote.example"},
        {"base_url": "https://user:secret@example.com"},
        {"base_url": "https://ollama.com/api"},
        {"model": "outro-modelo"},
        {"api_model": "gpt-oss:120b"},
    ],
)
def test_invalid_config(kwargs):
    with pytest.raises(ValidationError):
        Settings(**kwargs)

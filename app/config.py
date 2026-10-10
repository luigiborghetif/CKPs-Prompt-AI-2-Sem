"""Configuração sem chamadas de rede e sem credenciais embutidas."""

from pathlib import Path
from typing import Literal

from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseModel):
    model_config = ConfigDict(frozen=True)

    api_key: SecretStr = SecretStr("")
    base_url: str = "https://ollama.com"
    model: str = "gemma4:cloud"
    api_model: str = ""
    temperature: float = Field(default=0, ge=0, le=1)
    memory_tokens: int = Field(default=1200, ge=800, le=1500)
    max_output_tokens: int = Field(default=900, ge=256, le=4096)
    timeout: float = Field(default=90, gt=0, le=600)
    max_input_chars: int = Field(default=4000, ge=100, le=20000)
    max_sessions: int = Field(default=100, ge=1)
    session_ttl_seconds: int = Field(default=3600, ge=60)
    host: Literal["127.0.0.1", "localhost", "0.0.0.0"] = "127.0.0.1"
    port: int = Field(default=7860, ge=1024, le=65535)

    @field_validator("base_url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        from urllib.parse import urlparse

        value = value.strip().rstrip("/")
        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("OLLAMA_BASE_URL deve ser uma URL HTTP(S).")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("Não inclua credenciais ou parâmetros na URL do Ollama.")
        if parsed.path:
            raise ValueError("Use a raiz do servidor Ollama, sem /api ou /v1.")
        if parsed.hostname not in {"localhost", "127.0.0.1", "::1"} and parsed.scheme != "https":
            raise ValueError("Endpoints remotos devem usar HTTPS para proteger a chave.")
        return value

    @field_validator("model")
    @classmethod
    def validate_model(cls, value: str) -> str:
        if value != "gemma4:cloud":
            raise ValueError("O modelo lógico da CKP01 deve ser gemma4:cloud.")
        return value

    @field_validator("api_model")
    @classmethod
    def validate_api_model(cls, value: str) -> str:
        value = value.strip()
        if value not in {"", "gemma4:cloud", "gemma4:31b"}:
            raise ValueError("O alias de API deve ser vazio, gemma4:cloud ou gemma4:31b.")
        return value

    @property
    def configured(self) -> bool:
        key = self.api_key.get_secret_value().strip()
        return bool(key) and key not in {"sua_chave_aqui", "cole_sua_chave_aqui"}

    @property
    def effective_model(self) -> str:
        return self.api_model or self.model


def load_settings(env_file: Path | None = None) -> Settings:
    """Variáveis do processo têm prioridade sobre o .env do projeto."""
    import os

    values = {**dotenv_values(env_file or ROOT / ".env"), **os.environ}
    mapping = {
        "OLLAMA_API_KEY": "api_key",
        "OLLAMA_BASE_URL": "base_url",
        "OLLAMA_MODEL": "model",
        "OLLAMA_API_MODEL": "api_model",
        "MODEL_TEMPERATURE": "temperature",
        "MEMORY_TOKEN_LIMIT": "memory_tokens",
        "MODEL_MAX_TOKENS": "max_output_tokens",
        "OLLAMA_TIMEOUT": "timeout",
        "MAX_INPUT_CHARS": "max_input_chars",
        "MAX_SESSIONS": "max_sessions",
        "SESSION_TTL_SECONDS": "session_ttl_seconds",
        "APP_HOST": "host",
        "APP_PORT": "port",
    }
    return Settings(
        **{field: values[key] for key, field in mapping.items() if values.get(key) is not None}
    )

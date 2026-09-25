"""src/core/config.py

Centralized, typed configuration utilizing an absolute path mapping registry
to ensure seamless .env ingestion across multi-nested runtime terminals.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# Dynamically locate the project root folder regardless of execution path context
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_ENV_FILE_PATH = _PROJECT_ROOT / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE_PATH), 
        env_file_encoding="utf-8", 
        extra="ignore"
    )

    openai_api_key: SecretStr = SecretStr("")
    openai_model: str = "meta-llama/llama-3-8b-instruct:free"
    openai_base_url: str = "https://openrouter.ai/api/v1"
    database_path: str = "compliance.db"
    
    # Restoring the critical orchestrator framework properties
    cache_ttl_seconds: int = 86400
    batch_concurrency_limit: int = 10


@lru_cache
def get_settings() -> Settings:
    return Settings()

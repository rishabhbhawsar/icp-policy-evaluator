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
def _find_project_root(start: Path, markers: tuple[str, ...] = (".env", ".git")) -> Path:
    """Walk upward from `start` until a directory containing one of `markers`
    is found. Falls back to `start` if nothing is found, so a missing .env
    fails with a clear 'not found' rather than silently resolving to the
    wrong directory."""
    current = start.resolve()
    for candidate in (current, *current.parents):
        if any((candidate / marker).exists() for marker in markers):
            return candidate
    return start


_PROJECT_ROOT = _find_project_root(Path(__file__).parent)
_ENV_FILE_PATH = _PROJECT_ROOT / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE_PATH), 
        env_file_encoding="utf-8", 
        extra="ignore"
    )

    openai_api_key: SecretStr = SecretStr("")
    openai_model: str = "openrouter/free"
    openai_base_url: str = "https://openrouter.ai/api/v1"
    database_path: str = "compliance.db"

    judge_max_tokens: int = 2000
    cache_ttl_seconds: int = 86400
    batch_concurrency_limit: int = 2


@lru_cache
def get_settings() -> Settings:
    return Settings()

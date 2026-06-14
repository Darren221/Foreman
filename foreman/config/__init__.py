"""Application configuration loaded from environment / `.env`."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

ProviderName = Literal["openai", "anthropic"]


class Settings(BaseSettings):
    """Runtime settings, read from environment variables or a local `.env`.

    Secrets live here and nowhere in source. See `.env.example` for the contract.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    default_provider: ProviderName = "anthropic"
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    tavily_api_key: str | None = None

    openai_model: str = "gpt-4o"
    anthropic_model: str = "claude-sonnet-4-6"

    embedding_model: str = "text-embedding-3-small"
    memory_path: Path = Path("./data/memory")
    checkpoint_path: Path = Path("./data/checkpoints.sqlite")
    approval_path: Path = Path("./data/approvals.sqlite")
    trace_path: Path = Path("./data/traces.sqlite")
    workspace_path: Path = Path("./data/workspace")
    database_dsn: str | None = None


__all__ = ["Settings", "ProviderName"]

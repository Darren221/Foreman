"""Application configuration loaded from environment / `.env`."""

from __future__ import annotations

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


__all__ = ["Settings", "ProviderName"]

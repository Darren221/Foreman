"""Provider selection: turn config into a concrete `LLMProvider`.

This is the multi-model routing seam. Today it picks by name; later it can route
per-agent or per-task complexity without any caller changing.
"""

from __future__ import annotations

from foreman.config import ProviderName, Settings
from foreman.llm.base import LLMProvider
from foreman.llm.providers import AnthropicProvider, OpenAIProvider


def select_provider(settings: Settings, provider: ProviderName | None = None) -> LLMProvider:
    """Return the provider named by `provider`, or the configured default.

    Raises `ValueError` if the chosen provider has no API key configured, so a
    misconfiguration fails at selection time rather than at the first live call.
    """
    chosen = provider or settings.default_provider

    if chosen == "openai":
        if not settings.openai_api_key:
            raise ValueError("openai selected but OPENAI_API_KEY is not set")
        return OpenAIProvider(settings.openai_api_key, settings.openai_model)

    if chosen == "anthropic":
        if not settings.anthropic_api_key:
            raise ValueError("anthropic selected but ANTHROPIC_API_KEY is not set")
        return AnthropicProvider(settings.anthropic_api_key, settings.anthropic_model)

    raise ValueError(f"unknown provider: {chosen}")

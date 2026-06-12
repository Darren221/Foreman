import pytest

from foreman.config import Settings
from foreman.llm import LLMProvider, select_provider


def _settings(**kwargs: object) -> Settings:
    # Disable .env loading so these tests don't pick up the developer's real keys.
    return Settings(_env_file=None, **kwargs)  # type: ignore[call-arg]


def test_router_selects_anthropic_by_config() -> None:
    settings = _settings(default_provider="anthropic", anthropic_api_key="x")
    provider = select_provider(settings)
    assert isinstance(provider, LLMProvider)
    assert provider.name == "anthropic"


def test_router_selects_openai_by_config() -> None:
    settings = _settings(default_provider="openai", openai_api_key="x")
    provider = select_provider(settings)
    assert provider.name == "openai"


def test_router_can_override_provider() -> None:
    settings = _settings(default_provider="anthropic", openai_api_key="x")
    provider = select_provider(settings, provider="openai")
    assert provider.name == "openai"


def test_router_rejects_provider_without_key() -> None:
    settings = _settings(default_provider="openai")
    with pytest.raises(ValueError):
        select_provider(settings)

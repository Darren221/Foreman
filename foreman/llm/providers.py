"""Concrete `LLMProvider` adapters for OpenAI and Anthropic.

Both lazily construct their SDK client on first use, so a provider can be
instantiated (and selected by the router) without a key or a network call — only
an actual completion needs credentials. Live calls are exercised by manual smoke
runs, not the test suite.
"""

from __future__ import annotations

from typing import Any, cast

from foreman.llm.base import LLMProvider, T


class OpenAIProvider(LLMProvider):
    name = "openai"

    def __init__(self, api_key: str, model: str) -> None:
        self._api_key = api_key
        self._model = model
        self._client: Any = None

    def _ensure_client(self) -> Any:
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=self._api_key)
        return self._client

    def structured_complete(self, prompt: str, schema: type[T]) -> T:
        client = self._ensure_client()
        completion = client.beta.chat.completions.parse(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            response_format=schema,
        )
        parsed = completion.choices[0].message.parsed
        if parsed is None:
            raise ValueError("OpenAI returned no parseable structured output")
        return cast(T, parsed)


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(self, api_key: str, model: str) -> None:
        self._api_key = api_key
        self._model = model
        self._client: Any = None

    def _ensure_client(self) -> Any:
        if self._client is None:
            from anthropic import Anthropic

            self._client = Anthropic(api_key=self._api_key)
        return self._client

    def structured_complete(self, prompt: str, schema: type[T]) -> T:
        client = self._ensure_client()
        # Force the model to "return" its answer by calling a tool whose input
        # schema is exactly the target model — the cleanest way to get strict
        # structured output from the Messages API.
        tool = {
            "name": "respond",
            "description": "Return the structured response.",
            "input_schema": schema.model_json_schema(),
        }
        message = client.messages.create(
            model=self._model,
            max_tokens=4096,
            tools=[tool],
            tool_choice={"type": "tool", "name": "respond"},
            messages=[{"role": "user", "content": prompt}],
        )
        for block in message.content:
            if getattr(block, "type", None) == "tool_use":
                return schema.model_validate(block.input)
        raise ValueError("Anthropic returned no tool_use block to parse")

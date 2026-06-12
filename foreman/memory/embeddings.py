"""The embedding interface: turn text into vectors for semantic recall.

Same dependency-inversion idea as `LLMProvider`: the memory store depends on an
`Embedder`, not on OpenAI directly, so the model is swappable and tests inject a
deterministic fake (no key, no network).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Embedder(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text."""


class OpenAIEmbedder(Embedder):
    """OpenAI embeddings. Lazily builds its client, so it can be constructed
    without a key or network — only `embed` actually calls out."""

    def __init__(self, api_key: str, model: str = "text-embedding-3-small") -> None:
        self._api_key = api_key
        self._model = model
        self._client: Any = None

    def _ensure_client(self) -> Any:
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=self._api_key)
        return self._client

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        client = self._ensure_client()
        response = client.embeddings.create(input=texts, model=self._model)
        return [d.embedding for d in response.data]

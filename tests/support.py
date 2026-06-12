"""Shared test doubles."""

from __future__ import annotations

from foreman.memory import Embedder, MemoryStore
from foreman.schemas import TaskMemory


class FakeEmbedder(Embedder):
    """Deterministic bag-of-words embedder: cosine similarity tracks word overlap,
    so recall is predictable without a model or network."""

    _VOCAB = ["bicycle", "history", "python", "cooking", "space", "music", "detail"]

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [
            [1.0 if word in text.lower() else 0.0 for word in self._VOCAB] for text in texts
        ]


class NullMemoryStore(MemoryStore):
    """A memory store that records nothing and recalls nothing — for tests that
    don't care about memory and must stay offline (no embedding calls)."""

    def remember(self, memory: TaskMemory) -> None:
        return None

    def recall(self, query: str, k: int = 5) -> list[TaskMemory]:
        return []

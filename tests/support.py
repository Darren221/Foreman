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

    def recall(self, query: str, k: int = 5, user_id: str | None = None) -> list[TaskMemory]:
        return []

    def delete(self, ids: list[str]) -> None:
        return None

    def delete_user(self, user_id: str) -> None:
        return None


class DictMemoryStore(MemoryStore):
    """An in-memory store that actually keeps memories — recall does substring match
    on the task description. Enough to verify remember/recall/delete wiring offline
    without Chroma or an embedder."""

    def __init__(self) -> None:
        self._items: dict[str, TaskMemory] = {}

    def remember(self, memory: TaskMemory) -> None:
        self._items[memory.id] = memory

    def recall(self, query: str, k: int = 5, user_id: str | None = None) -> list[TaskMemory]:
        hits = [
            m
            for m in self._items.values()
            if query.lower() in m.task_description.lower()
            and (user_id is None or m.user_id == user_id)
        ]
        return hits[:k]

    def delete(self, ids: list[str]) -> None:
        for memory_id in ids:
            self._items.pop(memory_id, None)

    def delete_user(self, user_id: str) -> None:
        self._items = {i: m for i, m in self._items.items() if m.user_id != user_id}

"""Shared test doubles."""

from __future__ import annotations

from foreman.memory import MemoryStore
from foreman.schemas import TaskMemory


class NullMemoryStore(MemoryStore):
    """A memory store that records nothing and recalls nothing — for tests that
    don't care about memory and must stay offline (no embedding calls)."""

    def remember(self, memory: TaskMemory) -> None:
        return None

    def recall(self, query: str, k: int = 5) -> list[TaskMemory]:
        return []

"""Long-term semantic memory, backed by an embedded ChromaDB.

Chroma is used as a pure vector store: we embed the memory's task description
ourselves (bring-your-own embeddings) and keep the full `TaskMemory` as a JSON
document, so recall can hand back the complete record. The store runs in-process
against a local directory — no server.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, cast

from foreman.memory.embeddings import Embedder
from foreman.schemas import TaskMemory

_COLLECTION = "task_memories"


class MemoryStore(ABC):
    @abstractmethod
    def remember(self, memory: TaskMemory) -> None:
        """Persist a task memory."""

    @abstractmethod
    def recall(self, query: str, k: int = 5) -> list[TaskMemory]:
        """Return up to `k` memories most similar to `query`, closest first."""


class ChromaMemoryStore(MemoryStore):
    def __init__(self, path: Path, embedder: Embedder) -> None:
        import chromadb

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self._embedder = embedder
        self._client = chromadb.PersistentClient(path=str(path))
        self._collection = self._client.get_or_create_collection(
            name=_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )

    def remember(self, memory: TaskMemory) -> None:
        vector = self._embedder.embed([memory.task_description])[0]
        # chromadb's type stubs want numpy arrays; plain float lists work fine.
        self._collection.upsert(
            ids=[memory.id],
            embeddings=cast(Any, [vector]),
            documents=[memory.model_dump_json()],
        )

    def recall(self, query: str, k: int = 5) -> list[TaskMemory]:
        count = self._collection.count()
        if count == 0:
            return []
        vector = self._embedder.embed([query])[0]
        result = self._collection.query(
            query_embeddings=cast(Any, [vector]),
            n_results=min(k, count),
        )
        documents = result.get("documents") or [[]]
        return [TaskMemory.model_validate_json(doc) for doc in documents[0]]

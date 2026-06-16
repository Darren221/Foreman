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
    def recall(self, query: str, k: int = 5, user_id: str | None = None) -> list[TaskMemory]:
        """Return up to `k` memories most similar to `query`, closest first. If
        `user_id` is given, only that user's memories are considered."""

    @abstractmethod
    def delete(self, ids: list[str]) -> None:
        """Purge the given memories by id."""

    @abstractmethod
    def delete_user(self, user_id: str) -> None:
        """Purge all of a user's memories (the user-data delete path; SPEC §7)."""


class ChromaMemoryStore(MemoryStore):
    def __init__(
        self, path: Path, embedder: Embedder, *, host: str | None = None, port: int = 8000
    ) -> None:
        import chromadb

        self._embedder = embedder
        # Server mode (a shared Chroma the API and workers both reach) when a host is
        # given; otherwise embedded against a local directory.
        if host:
            self._client = chromadb.HttpClient(host=host, port=port)
        else:
            path = Path(path)
            path.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(path))
        self._collection = self._client.get_or_create_collection(
            name=_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )

    def remember(self, memory: TaskMemory) -> None:
        vector = self._embedder.embed([memory.task_description])[0]
        # chromadb's type stubs want numpy arrays; plain float lists work fine.
        # `user_id` rides in metadata so recall/delete can filter on it server-side.
        self._collection.upsert(
            ids=[memory.id],
            embeddings=cast(Any, [vector]),
            documents=[memory.model_dump_json()],
            metadatas=[{"user_id": memory.user_id}],
        )

    def recall(self, query: str, k: int = 5, user_id: str | None = None) -> list[TaskMemory]:
        count = self._collection.count()
        if count == 0:
            return []
        vector = self._embedder.embed([query])[0]
        where = {"user_id": user_id} if user_id is not None else None
        result = self._collection.query(
            query_embeddings=cast(Any, [vector]),
            n_results=min(k, count),
            where=cast(Any, where),
        )
        documents = result.get("documents") or [[]]
        return [TaskMemory.model_validate_json(doc) for doc in documents[0]]

    def delete(self, ids: list[str]) -> None:
        if ids:
            self._collection.delete(ids=ids)

    def delete_user(self, user_id: str) -> None:
        self._collection.delete(where=cast(Any, {"user_id": user_id}))

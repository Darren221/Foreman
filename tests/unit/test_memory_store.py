from pathlib import Path
from typing import Any

from foreman.memory import ChromaMemoryStore, Embedder, OpenAIEmbedder
from foreman.schemas import TaskMemory


class FakeEmbedder(Embedder):
    """Deterministic bag-of-words embedder: cosine similarity tracks word overlap,
    so recall ranking is predictable without a model or network."""

    _VOCAB = ["bicycle", "history", "python", "cooking", "space", "music"]

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            lowered = text.lower()
            vectors.append([1.0 if word in lowered else 0.0 for word in self._VOCAB])
        return vectors


def _memory(description: str, **kw: object) -> TaskMemory:
    return TaskMemory(
        task_description=description,
        outcome=str(kw.get("outcome", "passed")),
        score=float(kw.get("score", 0.9)),  # type: ignore[arg-type]
        tools_used=list(kw.get("tools_used", [])),  # type: ignore[arg-type]
        user_id=str(kw.get("user_id", "default")),
    )


def test_recall_and_delete_are_scoped_per_user(tmp_path: Path) -> None:
    store = ChromaMemoryStore(tmp_path / "mem", FakeEmbedder())
    store.remember(_memory("history of the bicycle", user_id="alice"))
    store.remember(_memory("history of the bicycle", user_id="bob"))

    # recall is scoped to the asking user
    alice = store.recall("history of the bicycle", k=5, user_id="alice")
    assert [m.user_id for m in alice] == ["alice"]

    # deleting alice's data leaves bob's intact
    store.delete_user("alice")
    assert store.recall("history of the bicycle", k=5, user_id="alice") == []
    bob = store.recall("history of the bicycle", k=5, user_id="bob")
    assert [m.user_id for m in bob] == ["bob"]


def test_remember_and_recall_roundtrip(tmp_path: Path) -> None:
    store = ChromaMemoryStore(tmp_path / "mem", FakeEmbedder())
    store.remember(_memory("history of the bicycle", tools_used=["web_search"]))

    got = store.recall("history of the bicycle", k=1)
    assert len(got) == 1
    assert got[0].task_description == "history of the bicycle"
    assert got[0].tools_used == ["web_search"]


def test_server_mode_uses_http_client(monkeypatch: Any) -> None:
    import chromadb

    called: dict[str, object] = {}

    class _FakeCollection:
        def count(self) -> int:
            return 0

    class _FakeClient:
        def get_or_create_collection(self, **kwargs: object) -> _FakeCollection:
            return _FakeCollection()

    def _fake_http(host: str, port: int) -> _FakeClient:
        called["host"], called["port"] = host, port
        return _FakeClient()

    monkeypatch.setattr(chromadb, "HttpClient", _fake_http)
    ChromaMemoryStore(Path("/unused"), FakeEmbedder(), host="chroma", port=8000)
    assert called == {"host": "chroma", "port": 8000}


def test_delete_removes_a_memory_from_recall(tmp_path: Path) -> None:
    store = ChromaMemoryStore(tmp_path / "mem", FakeEmbedder())
    memory = _memory("history of the bicycle")
    store.remember(memory)
    assert store.recall("history of the bicycle", k=1)

    store.delete([memory.id])
    assert store.recall("history of the bicycle", k=1) == []


def test_recall_ranks_relevant_above_irrelevant(tmp_path: Path) -> None:
    store = ChromaMemoryStore(tmp_path / "mem", FakeEmbedder())
    store.remember(_memory("bicycle history facts"))
    store.remember(_memory("python cooking recipes"))

    got = store.recall("history of the bicycle", k=2)
    assert got[0].task_description == "bicycle history facts"


def test_recall_on_empty_store_returns_empty(tmp_path: Path) -> None:
    store = ChromaMemoryStore(tmp_path / "mem", FakeEmbedder())
    assert store.recall("anything") == []


def test_openai_embedder_constructs_without_key() -> None:
    # No key, no network — only embed() would call out.
    embedder = OpenAIEmbedder(api_key="", model="text-embedding-3-small")
    assert isinstance(embedder, Embedder)

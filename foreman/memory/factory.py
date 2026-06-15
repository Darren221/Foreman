"""Build the default memory store from configuration."""

from __future__ import annotations

from foreman.config import Settings
from foreman.memory.embeddings import OpenAIEmbedder
from foreman.memory.store import ChromaMemoryStore, MemoryStore


def build_default_memory_store(settings: Settings) -> MemoryStore:
    embedder = OpenAIEmbedder(settings.openai_api_key or "", settings.embedding_model)
    return ChromaMemoryStore(
        settings.memory_path,
        embedder,
        host=settings.chroma_host,
        port=settings.chroma_port,
    )

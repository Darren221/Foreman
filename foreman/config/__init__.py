"""Application configuration loaded from environment / `.env`."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

ProviderName = Literal["openai", "anthropic"]
StoreBackend = Literal["sqlite", "postgres"]


class Settings(BaseSettings):
    """Runtime settings, read from environment variables or a local `.env`.

    Secrets live here and nowhere in source. See `.env.example` for the contract.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    default_provider: ProviderName = "anthropic"
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    tavily_api_key: str | None = None

    openai_model: str = "gpt-4o"
    anthropic_model: str = "claude-sonnet-4-6"

    embedding_model: str = "text-embedding-3-small"
    memory_path: Path = Path("./data/memory")
    # Chroma runs embedded (a local dir) by default; set chroma_host to use a Chroma
    # server, which the distributed deployment needs so the API and workers (separate
    # processes) share one long-term memory store.
    chroma_host: str | None = None
    chroma_port: int = 8000
    checkpoint_path: Path = Path("./data/checkpoints.sqlite")
    approval_path: Path = Path("./data/approvals.sqlite")
    trace_path: Path = Path("./data/traces.sqlite")
    workspace_path: Path = Path("./data/workspace")

    # Durable state (checkpointer, approval queue, trace store) lives on one of two
    # backends. SQLite (embedded, default) keeps local single-process runs and the
    # offline test suite self-contained; Postgres is the shared backend once the API
    # and workers run as separate processes (Phase 5). `database_dsn` is required
    # when store_backend == "postgres".
    store_backend: StoreBackend = "sqlite"
    database_dsn: str | None = None

    # The analyst's read-only data source for `db_query`. This is a *separate*
    # database from the operational store above: the data the crew analyses (e.g. a
    # sales warehouse) is rarely the same DB that holds Foreman's checkpoints and
    # approvals. Falls back to `database_dsn` when unset, so a single-DB deployment
    # still works without extra config. The `db_query` tool is inert when neither
    # is set (no DSN -> any query fails fast), which is correct for offline runs.
    analyst_database_dsn: str | None = None

    # Specialist execution fans out to Redis-brokered Celery workers (Phase 5 C4).
    # task_always_eager runs tasks in-process (tests/local), bypassing the broker.
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/1"
    celery_task_always_eager: bool = False


__all__ = ["Settings", "ProviderName", "StoreBackend"]

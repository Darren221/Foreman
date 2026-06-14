"""Build durable-state stores from configuration.

One place decides the storage backend for the checkpointer, approval queue, and
trace store, so calling code never picks a backend itself ("the factory hides the
choice"). SQLite is the embedded default; Postgres is selected by
`settings.store_backend` and needs `settings.database_dsn`.

The Postgres drivers (psycopg, the langgraph Postgres saver) are an *optional*
dependency — a SQLite user shouldn't have to install them. So they're imported
only on the Postgres path, and a missing install is reported as the configuration
error it is, naming the fix, rather than surfacing as a bare ImportError later.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from foreman.config import Settings
from foreman.hitl.queue import ApprovalQueue
from foreman.observability.store import TraceStore
from foreman.storage.db import Conn


def build_approval_queue(settings: Settings) -> ApprovalQueue:
    return ApprovalQueue(_conn(settings, settings.approval_path))


def build_trace_store(settings: Settings) -> TraceStore:
    return TraceStore(_conn(settings, settings.trace_path))


def build_checkpointer(settings: Settings) -> Any:
    """The LangGraph checkpointer (saver) for the configured backend.

    Typed `Any`: `SqliteSaver` and `PostgresSaver` share LangGraph's
    `BaseCheckpointSaver` interface, but we won't import the Postgres one (an
    optional dep) unless it's selected, so there's no shared concrete type to name.
    """
    if settings.store_backend == "sqlite":
        from langgraph.checkpoint.sqlite import SqliteSaver

        settings.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        return SqliteSaver(
            sqlite3.connect(str(settings.checkpoint_path), check_same_thread=False)
        )

    postgres_saver, psycopg = _require_postgres_saver()
    # PostgresSaver expects an autocommit connection: setup() issues
    # CREATE INDEX CONCURRENTLY, which cannot run inside a transaction block, and
    # the saver manages its own transactions for normal operations.
    conn = psycopg.connect(_require_dsn(settings), autocommit=True)
    saver = postgres_saver(conn)
    saver.setup()  # create the checkpointer's tables if absent
    return saver


def _conn(settings: Settings, sqlite_path: Path) -> Conn:
    if settings.store_backend == "postgres":
        return Conn.postgres(_require_dsn(settings))
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    return Conn.sqlite(sqlite_path)


def _require_dsn(settings: Settings) -> str:
    if not settings.database_dsn:
        raise RuntimeError(
            "store_backend='postgres' requires database_dsn; set it in .env "
            "(e.g. postgresql://user:pass@host:5432/foreman)."
        )
    return settings.database_dsn


def _require_postgres_saver() -> tuple[Any, Any]:
    """Import the Postgres checkpointer + driver, or fail with an actionable message."""
    try:
        import psycopg
        from langgraph.checkpoint.postgres import PostgresSaver
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise RuntimeError(
            "Postgres backend selected but its drivers aren't installed; "
            "run `pip install foreman[postgres]`."
        ) from exc
    return PostgresSaver, psycopg


__all__ = ["build_approval_queue", "build_trace_store", "build_checkpointer"]

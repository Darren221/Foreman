"""Read-only database queries.

The tool depends on a `DatabaseBackend` interface, so tests use a fake (no DB) and
the real Postgres backend is one swappable implementation. A guard rejects anything
that isn't a SELECT — the analyst can read data, never modify it.
"""

from __future__ import annotations

from typing import Any, Protocol

from foreman.schemas import Specialist
from foreman.tools.base import Tool


class DatabaseBackend(Protocol):
    def query(self, sql: str) -> list[dict[str, Any]]:
        """Run `sql` and return rows as dicts."""
        ...


class PostgresBackend:
    """Lazily-connected Postgres client (psycopg arrives in C3). No connection is
    made until the first query, so the tool can be constructed without a DB."""

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn
        self._conn: Any = None

    def query(self, sql: str) -> list[dict[str, Any]]:
        if self._conn is None:
            import psycopg
            from psycopg.rows import dict_row

            self._conn = psycopg.connect(self._dsn, row_factory=dict_row)
        with self._conn.cursor() as cur:
            cur.execute(sql)
            rows: list[dict[str, Any]] = cur.fetchall()
        return rows


class FakeDatabase:
    """Deterministic stand-in for tests — returns canned rows for any query."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows

    def query(self, sql: str) -> list[dict[str, Any]]:
        return self._rows


class DatabaseQueryTool(Tool):
    name = "db_query"
    description = "Run a read-only SQL query and return rows."
    allowed_specialists = frozenset({Specialist.ANALYST})

    def __init__(self, backend: DatabaseBackend) -> None:
        self._backend = backend

    def run(self, **inputs: Any) -> dict[str, Any]:
        sql = inputs["query"]
        if not sql.strip().lower().startswith("select"):
            raise ValueError("only read-only SELECT queries are allowed")
        return {"rows": self._backend.query(sql)}

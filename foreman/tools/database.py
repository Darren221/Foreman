"""Read-only database queries.

The tool depends on a `DatabaseBackend` interface, so tests use a fake (no DB) and
the real Postgres backend is one swappable implementation. A guard rejects anything
that isn't a SELECT — the analyst can read data, never modify it.
"""

from __future__ import annotations

import re
from typing import Any, Protocol

from foreman.schemas import Specialist
from foreman.tools.base import Tool

# Single-quoted string literals (handling the '' escape), removed before the
# single-statement check so a semicolon *inside* a string isn't a false positive.
_STRING_LITERAL = re.compile(r"'(?:''|[^'])*'")


class DatabaseBackend(Protocol):
    def query(self, sql: str) -> list[dict[str, Any]]:
        """Run `sql` and return rows as dicts."""
        ...


class PostgresBackend:
    """Lazily-connected Postgres client (psycopg arrives in C3). No connection is
    made until the first query, so the tool can be constructed without a DB."""

    def __init__(
        self, dsn: str, statement_timeout_ms: int = 30_000, max_rows: int = 10_000
    ) -> None:
        self._dsn = dsn
        self._statement_timeout_ms = statement_timeout_ms
        # Cap the rows pulled into worker memory: a SELECT with no LIMIT over a huge
        # table would otherwise load the whole result set at once.
        self._max_rows = max_rows
        # One connection per backend, reused for the backend's lifetime; close() when
        # done. (A per-query connect would be safer but far slower.)
        self._conn: Any = None

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def _connect(self) -> Any:
        import psycopg
        from psycopg.rows import dict_row

        conn = psycopg.connect(self._dsn, row_factory=dict_row)
        # The real read-only guarantee: the connection itself refuses writes, so a
        # write that slips past the tool's string guard is rejected by Postgres.
        conn.read_only = True
        # Bound query runtime so a slow/looping query can't hang the connection.
        conn.execute(f"SET statement_timeout = {int(self._statement_timeout_ms)}")
        conn.commit()
        return conn

    def query(self, sql: str) -> list[dict[str, Any]]:
        if self._conn is None:
            self._conn = self._connect()
        try:
            with self._conn.cursor() as cur:
                cur.execute(sql)
                rows: list[dict[str, Any]] = cur.fetchmany(self._max_rows)
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
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
        # A friendly early fail. The binding guarantee is the read-only *connection*
        # in PostgresBackend; this just rejects the obvious cases before a round-trip.
        stripped = inputs["query"].strip().rstrip(";").strip()
        if not stripped.lower().startswith("select"):
            raise ValueError("only read-only SELECT queries are allowed")
        if ";" in _STRING_LITERAL.sub("", stripped):
            raise ValueError("only a single statement is allowed")
        sql = inputs["query"]
        return {"rows": self._backend.query(sql)}

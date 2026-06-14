"""The storage seam: one connection wrapper over SQLite or Postgres.

`ApprovalQueue` and `TraceStore` are just SQL talking to a database. To run on
either backend we'd otherwise fork every query, because the two drivers share no
parameter-placeholder style (SQLite wants ``?``, psycopg wants ``%s``). Rather
than duplicate the SQL per backend, the stores write each query *once* in qmark
(``?``) style and `Conn` rewrites ``?`` -> ``%s`` when the backend is Postgres.

That rewrite is naive on purpose, which makes it fragile in exactly two ways: a
literal ``%`` (psycopg's format character) and a ``?|`` / ``?&`` JSONB operator
(indistinguishable-ish from a placeholder) would be silently mangled. We do not
*document* that hazard and hope — `assert_translatable` enforces it, loudly, at
the offending query, on every backend (so the SQLite test suite, which runs every
query, polices the whole query set in CI).

The one thing the validator cannot catch is a *bare* ``?`` Postgres JSONB
existence operator: it is syntactically identical to a placeholder. That is not a
hole to patch but the documented signal that a store has outgrown the seam — give
it backend-specific SQL instead. We never hit it today: both stores store JSON in
``TEXT`` columns and filter only on scalar columns, never inside the JSON.
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Iterable
from pathlib import Path
from typing import Any

# ``?|`` and ``?&`` are Postgres JSONB existence operators, not placeholders; the
# ``?`` -> ``%s`` rewrite would corrupt them.
_JSONB_OPS = re.compile(r"\?[|&]")


def assert_translatable(sql: str) -> None:
    """Reject SQL the qmark->%s rewrite would silently mangle.

    Raises `ValueError` with the offending SQL. Run on every query, on both
    backends, so the invariant ("SQL is written in plain qmark style") is checked
    wherever the seam is exercised — not left as a comment to be violated later.
    """
    if "%" in sql:
        raise ValueError(
            f"storage SQL must not contain a literal '%' (psycopg format char): {sql!r}"
        )
    if _JSONB_OPS.search(sql):
        raise ValueError(
            f"storage SQL uses a JSONB ?|/?& operator the placeholder rewrite would "
            f"corrupt; this store has outgrown the seam — give it backend-specific "
            f"SQL: {sql!r}"
        )
    if _has_quoted_question_mark(sql):
        raise ValueError(
            f"storage SQL has a '?' inside a string literal, which the placeholder "
            f"rewrite would corrupt: {sql!r}"
        )


def _has_quoted_question_mark(sql: str) -> bool:
    """True if a ``?`` appears inside a single-quoted string literal."""
    in_quote = False
    for ch in sql:
        if ch == "'":
            in_quote = not in_quote
        elif ch == "?" and in_quote:
            return True
    return False


class Conn:
    """A DB-API connection over SQLite or Postgres, exposing the narrow surface the
    stores use (`execute` returning a cursor, `commit`, `close`) and normalizing the
    two differences they'd otherwise see: how you connect, and the parameter
    placeholder. Rows are dict-like (`row["col"]`) on both backends.
    """

    def __init__(self, backend: str, raw: Any) -> None:
        self._backend = backend
        self._raw = raw

    @classmethod
    def sqlite(cls, path: str | Path) -> Conn:
        raw = sqlite3.connect(str(path), check_same_thread=False)
        raw.row_factory = sqlite3.Row
        return cls("sqlite", raw)

    @classmethod
    def postgres(cls, dsn: str) -> Conn:
        psycopg = _require_psycopg()
        from psycopg.rows import dict_row

        raw = psycopg.connect(dsn, row_factory=dict_row)
        return cls("postgres", raw)

    @property
    def is_postgres(self) -> bool:
        return self._backend == "postgres"

    def execute(self, sql: str, params: Iterable[Any] = ()) -> Any:
        """Run a qmark-style query and return the driver cursor.

        The query is validated, then (for Postgres only) its ``?`` placeholders are
        rewritten to ``%s``. Validation runs on both backends so the invariant is
        enforced even on a SQLite-only run.
        """
        assert_translatable(sql)
        if self.is_postgres:
            sql = sql.replace("?", "%s")
        return self._raw.execute(sql, tuple(params))

    def commit(self) -> None:
        self._raw.commit()

    def close(self) -> None:
        self._raw.close()


def _require_psycopg() -> Any:
    """Import psycopg or fail with an actionable message.

    psycopg is an *optional* dependency: a SQLite user shouldn't need a Postgres
    driver. Selecting the Postgres backend without it installed is a configuration
    error, so we surface it as one — at connection time, naming the fix — rather
    than letting a bare ImportError surface deep in a later call.
    """
    try:
        import psycopg
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise RuntimeError(
            "Postgres backend selected but psycopg isn't installed; "
            "run `pip install foreman[postgres]`."
        ) from exc
    return psycopg


__all__ = ["Conn", "assert_translatable"]

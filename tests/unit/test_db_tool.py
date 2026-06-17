"""C2: the database query tool runs read-only SELECTs through a backend and refuses
anything that would write."""

from __future__ import annotations

import os

import pytest

from foreman.schemas import Specialist
from foreman.tools.database import DatabaseQueryTool, FakeDatabase, PostgresBackend


def test_runs_a_select_through_the_backend() -> None:
    tool = DatabaseQueryTool(FakeDatabase(rows=[{"n": 1}, {"n": 2}]))
    assert tool.run(query="SELECT n FROM t")["rows"] == [{"n": 1}, {"n": 2}]


def test_rejects_non_select_queries() -> None:
    tool = DatabaseQueryTool(FakeDatabase(rows=[]))
    with pytest.raises(ValueError, match="read-only"):
        tool.run(query="DELETE FROM t")


def test_rejects_multi_statement_queries() -> None:
    # The classic bypass: a write smuggled after a leading SELECT.
    tool = DatabaseQueryTool(FakeDatabase(rows=[{"n": 1}]))
    with pytest.raises(ValueError):
        tool.run(query="SELECT 1; DELETE FROM t")


@pytest.mark.requires_postgres
def test_postgres_backend_refuses_writes() -> None:
    # The real guarantee: the connection itself is read-only, so even a write that
    # slips past the tool guard is refused by the database.
    import psycopg

    backend = PostgresBackend(os.environ["FOREMAN_TEST_POSTGRES_DSN"])
    assert backend.query("SELECT 1 AS n") == [{"n": 1}]
    with pytest.raises(psycopg.errors.ReadOnlySqlTransaction):
        backend.query("CREATE TEMP TABLE evil (x int)")
    backend.close()


@pytest.mark.requires_postgres
def test_postgres_backend_enforces_statement_timeout() -> None:
    import psycopg

    backend = PostgresBackend(os.environ["FOREMAN_TEST_POSTGRES_DSN"], statement_timeout_ms=200)
    with pytest.raises(psycopg.errors.QueryCanceled):
        backend.query("SELECT pg_sleep(2)")
    backend.close()


def test_db_tool_is_for_the_analyst() -> None:
    assert DatabaseQueryTool(FakeDatabase([])).allowed_specialists == frozenset(
        {Specialist.ANALYST}
    )

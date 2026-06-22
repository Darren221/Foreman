"""C2: the default registry wires up all the crew's tools with their allow-lists."""

from __future__ import annotations

from foreman.config import Settings
from foreman.schemas import Specialist
from foreman.tools import build_default_registry


def test_default_registry_has_the_crew_tools() -> None:
    registry = build_default_registry(Settings(_env_file=None))  # type: ignore[call-arg]

    expected = {"web_search", "code_execution", "file_io", "api_call"}
    assert {registry.get(n).name for n in expected} == expected

    assert registry.get("code_execution").allowed_specialists == frozenset({Specialist.ANALYST})
    assert Specialist.WRITER in registry.get("file_io").allowed_specialists


def test_db_query_registered_only_when_a_data_source_is_configured() -> None:
    # No DSN: the analyst has no database, so db_query is not wired (its calls would
    # only ever fail). `has` is the honest signal of whether live data is available.
    no_db = build_default_registry(Settings(_env_file=None))  # type: ignore[call-arg]
    assert not no_db.has("db_query")

    # A configured analyst DSN wires db_query (read-only, analyst-only).
    with_db = build_default_registry(
        Settings(_env_file=None, analyst_database_dsn="postgresql://x/y")  # type: ignore[call-arg]
    )
    assert with_db.has("db_query")
    assert with_db.get("db_query").allowed_specialists == frozenset({Specialist.ANALYST})


def test_analyst_dsn_falls_back_to_the_operational_dsn() -> None:
    # A single-database deployment sets only database_dsn; the analyst reuses it.
    registry = build_default_registry(
        Settings(_env_file=None, database_dsn="postgresql://x/y")  # type: ignore[call-arg]
    )
    assert registry.has("db_query")

"""C2: the default registry wires up all the crew's tools with their allow-lists."""

from __future__ import annotations

from foreman.config import Settings
from foreman.schemas import Specialist
from foreman.tools import build_default_registry


def test_default_registry_has_the_crew_tools() -> None:
    registry = build_default_registry(Settings(_env_file=None))  # type: ignore[call-arg]

    names = {registry.get(n).name for n in ("web_search", "code_execution", "file_io", "db_query")}
    assert names == {"web_search", "code_execution", "file_io", "db_query"}

    assert registry.get("code_execution").allowed_specialists == frozenset({Specialist.ANALYST})
    assert registry.get("db_query").allowed_specialists == frozenset({Specialist.ANALYST})
    assert Specialist.WRITER in registry.get("file_io").allowed_specialists

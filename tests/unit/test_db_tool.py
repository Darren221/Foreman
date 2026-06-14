"""C2: the database query tool runs read-only SELECTs through a backend and refuses
anything that would write."""

from __future__ import annotations

import pytest

from foreman.schemas import Specialist
from foreman.tools.database import DatabaseQueryTool, FakeDatabase


def test_runs_a_select_through_the_backend() -> None:
    tool = DatabaseQueryTool(FakeDatabase(rows=[{"n": 1}, {"n": 2}]))
    assert tool.run(query="SELECT n FROM t")["rows"] == [{"n": 1}, {"n": 2}]


def test_rejects_non_select_queries() -> None:
    tool = DatabaseQueryTool(FakeDatabase(rows=[]))
    with pytest.raises(ValueError, match="read-only"):
        tool.run(query="DELETE FROM t")


def test_db_tool_is_for_the_analyst() -> None:
    assert DatabaseQueryTool(FakeDatabase([])).allowed_specialists == frozenset(
        {Specialist.ANALYST}
    )

"""C2: the file I/O tool reads and writes within a per-task workspace, and refuses
paths that escape it."""

from __future__ import annotations

from pathlib import Path

import pytest

from foreman.schemas import Specialist
from foreman.tools.files import FileTool


def test_write_then_read_round_trips(tmp_path: Path) -> None:
    tool = FileTool(tmp_path)
    tool.run(operation="write", path="notes/draft.md", content="hello")
    assert tool.run(operation="read", path="notes/draft.md")["content"] == "hello"


def test_path_escaping_the_workspace_is_rejected(tmp_path: Path) -> None:
    tool = FileTool(tmp_path)
    with pytest.raises(ValueError, match="escapes"):
        tool.run(operation="write", path="../outside.txt", content="x")


def test_file_tool_is_for_writer_and_analyst(tmp_path: Path) -> None:
    assert FileTool(tmp_path).allowed_specialists == frozenset(
        {Specialist.WRITER, Specialist.ANALYST}
    )

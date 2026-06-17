"""File I/O confined to a per-task workspace.

Every path is resolved and checked to stay inside the workspace directory, so a
specialist can't read or write outside its sandbox (e.g. via `../`). The writer and
analyst use it to pass artifacts between steps.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from foreman.schemas import Specialist
from foreman.tools.base import Tool
from foreman.tools.limits import read_capped


class FileTool(Tool):
    name = "file_io"
    description = "Read and write files within the task workspace."
    allowed_specialists = frozenset({Specialist.WRITER, Specialist.ANALYST})

    def __init__(self, workspace: str | Path) -> None:
        self._workspace = Path(workspace)

    def run(self, **inputs: Any) -> dict[str, Any]:
        operation = inputs["operation"]
        target = self._resolve(inputs["path"])
        if operation == "write":
            target.parent.mkdir(parents=True, exist_ok=True)
            content = inputs["content"]
            target.write_text(content)
            written = str(target.relative_to(self._workspace.resolve()))
            return {"path": written, "bytes": len(content)}
        if operation == "read":
            with target.open("rb") as handle:
                raw, truncated = read_capped(handle)
            return {"content": raw.decode("utf-8", "replace"), "truncated": truncated}
        raise ValueError(f"unknown file operation {operation!r}")

    def _resolve(self, relative: str) -> Path:
        base = self._workspace.resolve()
        target = (base / relative).resolve()
        if target != base and base not in target.parents:
            raise ValueError(f"path {relative!r} escapes the workspace")
        return target

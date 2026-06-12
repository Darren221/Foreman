"""H1: with a checkpointer, a run can pause at an interrupt and resume later —
even from a freshly built graph over the same SQLite file (cross-process)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from foreman.graph.builder import build_graph
from foreman.llm.base import LLMProvider, T
from foreman.schemas import (
    Plan,
    ResearchFindings,
    ReviewResult,
    Specialist,
    Subtask,
    Synthesis,
    Task,
)
from foreman.tools import ToolRegistry, WebSearchTool
from tests.support import NullMemoryStore


class CannedProvider(LLMProvider):
    name = "canned"

    def __init__(self, plan: Plan) -> None:
        self._plan = plan

    def structured_complete(self, prompt: str, schema: type[T]) -> T:
        if schema is ResearchFindings:
            return ResearchFindings(content="findings")  # type: ignore[return-value]
        if schema is ReviewResult:
            return ReviewResult(passed=True, score=0.9, feedback="ok")  # type: ignore[return-value]
        if schema is Synthesis:
            return Synthesis(result="the answer")  # type: ignore[return-value]
        return self._plan  # type: ignore[return-value]


class FakeBackend:
    def search(self, query: str, max_results: int) -> list[dict[str, Any]]:
        return [{"title": "t", "url": "u", "content": "c"}]


def _registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(WebSearchTool(FakeBackend()))
    return reg


def _plan() -> Plan:
    return Plan(
        task_id="t1",
        subtasks=[
            Subtask(
                id="s1",
                description="research bicycles",
                assigned_specialist=Specialist.RESEARCHER,
                expected_output="findings",
                complexity=1,
            )
        ],
    )


def test_run_pauses_at_approval_and_resumes_from_fresh_graph(tmp_path: Path) -> None:
    db = str(tmp_path / "checkpoints.sqlite")
    config = {"configurable": {"thread_id": "run-1"}}
    task = Task(description="research the bicycle", require_approval=True)

    # First graph instance: run until it interrupts for approval.
    with SqliteSaver.from_conn_string(db) as saver:
        graph = build_graph(
            CannedProvider(_plan()), _registry(), NullMemoryStore(), checkpointer=saver
        )
        paused = graph.invoke({"task": task}, config)
        assert "__interrupt__" in paused
        assert not paused.get("result")  # not finished

    # A fresh graph instance over the same DB resumes and completes.
    with SqliteSaver.from_conn_string(db) as saver2:
        graph2 = build_graph(
            CannedProvider(_plan()), _registry(), NullMemoryStore(), checkpointer=saver2
        )
        final = graph2.invoke(Command(resume="approved"), config)
        assert final.get("result")

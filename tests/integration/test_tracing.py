"""O1: a task run under the tracer records a real OpenTelemetry span tree — a root
`run` span with the node spans nested beneath it — into the SQLite trace store,
fully offline. With no tracer the pipeline is unchanged (covered elsewhere)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from foreman.graph import run_task
from foreman.llm.base import LLMProvider, T
from foreman.observability import OTelTracer, TraceStore
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


def test_traced_run_records_a_span_tree(tmp_path: Path) -> None:
    store = TraceStore(tmp_path / "traces.sqlite")
    tracer = OTelTracer(store)

    run_task(
        CannedProvider(_plan()),
        Task(description="research the bicycle"),
        registry=_registry(),
        memory_store=NullMemoryStore(),
        tracer=tracer,
    )

    runs = store.list_runs()
    assert len(runs) == 1
    root = store.get_trace(runs[0].trace_id)
    assert root is not None
    assert root.span.kind == "run"
    assert root.span.attributes["foreman.task"] == "research the bicycle"

    node_names = {child.span.name for child in root.children}
    assert "node:plan" in node_names  # node spans nest under the run span
    assert "node:synthesize" in node_names
    store.close()

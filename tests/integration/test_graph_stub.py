"""The graph must run end-to-end as stubs are replaced by real agents.

As of T3 the `plan` node calls the supervisor (LLM) and `execute` runs the
researcher (web search); review/synthesize remain stubs. A canned provider and a
fake search backend keep the test free and offline while proving the full
pipeline still reaches a result.
"""

from __future__ import annotations

from typing import Any

from foreman.graph import run_task
from foreman.llm.base import LLMProvider, T
from foreman.schemas import Plan, ReviewResult, Specialist, Subtask, Synthesis, Task
from foreman.tools import ToolRegistry, WebSearchTool


class CannedProvider(LLMProvider):
    """Returns a canned Plan, an accepting verdict, and a synthesised result."""

    name = "canned"

    def __init__(self, plan: Plan) -> None:
        self._plan = plan

    def structured_complete(self, prompt: str, schema: type[T]) -> T:
        if schema is ReviewResult:
            return ReviewResult(passed=True, score=1.0, feedback="ok")  # type: ignore[return-value]
        if schema is Synthesis:
            return Synthesis(result="synthesised result")  # type: ignore[return-value]
        return self._plan  # type: ignore[return-value]


def _canned_plan() -> Plan:
    return Plan(
        task_id="t1",
        subtasks=[
            Subtask(
                id="s1",
                description="research the topic",
                assigned_specialist=Specialist.RESEARCHER,
                expected_output="findings",
                complexity=1,
            )
        ],
    )


class FakeBackend:
    def search(self, query: str, max_results: int) -> list[dict[str, Any]]:
        return [{"title": "Bicycle", "url": "http://x", "content": "Invented in 1817."}]


def _registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(WebSearchTool(FakeBackend()))
    return reg


def test_graph_runs_end_to_end() -> None:
    provider = CannedProvider(_canned_plan())
    state = run_task(
        provider,
        Task(description="research the history of the bicycle"),
        registry=_registry(),
    )

    assert state["plan"] is not None
    assert state["plan"].subtasks[0].assigned_specialist is Specialist.RESEARCHER
    assert state["outputs"], "a specialist should have produced output"
    assert state["review"] is not None and state["review"].passed
    assert isinstance(state["result"], str) and state["result"]

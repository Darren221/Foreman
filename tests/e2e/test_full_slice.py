"""End-to-end Phase 1 slice: a research task flows supervisor -> researcher
(web search) -> reviewer -> synthesis and returns a written result. Also proves
the pipeline recovers from a specialist failure instead of crashing.

LLM and search are faked so the slice runs offline and deterministically; a
single live run is reserved for the demo.
"""

from __future__ import annotations

from typing import Any

import pytest

from foreman.graph import run_task
from foreman.llm.base import LLMProvider, T
from foreman.schemas import Plan, ReviewResult, Specialist, Subtask, Synthesis, Task
from foreman.tools import ToolRegistry, WebSearchTool


class SliceProvider(LLMProvider):
    name = "slice"

    def __init__(self, plan: Plan, review: ReviewResult, synthesis: str) -> None:
        self._plan = plan
        self._review = review
        self._synthesis = synthesis

    def structured_complete(self, prompt: str, schema: type[T]) -> T:
        if schema is Plan:
            return self._plan  # type: ignore[return-value]
        if schema is ReviewResult:
            return self._review  # type: ignore[return-value]
        if schema is Synthesis:
            return Synthesis(result=self._synthesis)  # type: ignore[return-value]
        raise AssertionError(f"unexpected schema {schema}")


class FakeBackend:
    def search(self, query: str, max_results: int) -> list[dict[str, Any]]:
        return [{"title": "Bicycle", "url": "http://x", "content": "Invented in 1817."}]


class BrokenBackend:
    def search(self, query: str, max_results: int) -> list[dict[str, Any]]:
        raise RuntimeError("search backend is down")


def _registry(backend: Any) -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(WebSearchTool(backend))
    return reg


def _plan() -> Plan:
    return Plan(
        task_id="t1",
        subtasks=[
            Subtask(
                id="s1",
                description="history of the bicycle",
                assigned_specialist=Specialist.RESEARCHER,
                expected_output="a written summary",
                complexity=2,
            )
        ],
    )


def test_research_slice_produces_synthesized_result() -> None:
    provider = SliceProvider(
        _plan(),
        ReviewResult(passed=True, score=0.95, feedback="good"),
        synthesis="The bicycle was invented in 1817.",
    )
    state = run_task(
        provider,
        Task(description="Research the history of the bicycle"),
        registry=_registry(FakeBackend()),
    )

    assert state["review"].passed is True
    assert state["result"] == "The bicycle was invented in 1817."


def test_pipeline_recovers_from_specialist_failure() -> None:
    reg = _registry(BrokenBackend())
    provider = SliceProvider(
        _plan(),
        ReviewResult(passed=False, score=0.0, feedback="no findings"),
        synthesis="Could not complete the research.",
    )

    # Must not raise despite the tool failing on every attempt.
    state = run_task(provider, Task(description="anything"), registry=reg)

    assert isinstance(state["result"], str) and state["result"]
    # the search failure was recorded, and retries were capped
    assert all(not inv.success for inv in reg.invocations)
    assert len(reg.invocations) == 2


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__])

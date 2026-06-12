"""The reviewer's verdict drives conditional routing: pass -> synthesize,
fail -> retry the specialist with feedback, up to a cap then proceed."""

from __future__ import annotations

from typing import Any

from foreman.graph import run_task
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


class ScriptedProvider(LLMProvider):
    """Returns a canned object per requested schema. ReviewResult is a queue so a
    test can script a sequence of verdicts; the last is reused if it runs dry."""

    name = "scripted"

    def __init__(self, plan: Plan, reviews: list[ReviewResult]) -> None:
        self._plan = plan
        self._reviews = reviews

    def structured_complete(self, prompt: str, schema: type[T]) -> T:
        if schema is Plan:
            return self._plan  # type: ignore[return-value]
        if schema is ResearchFindings:
            return ResearchFindings(content="researched findings")  # type: ignore[return-value]
        if schema is ReviewResult:
            verdict = self._reviews[0] if len(self._reviews) == 1 else self._reviews.pop(0)
            return verdict  # type: ignore[return-value]
        if schema is Synthesis:
            return Synthesis(result="synthesised result")  # type: ignore[return-value]
        raise AssertionError(f"unexpected schema {schema}")


class FakeBackend:
    def search(self, query: str, max_results: int) -> list[dict[str, Any]]:
        return [{"title": "Bicycle", "url": "http://x", "content": "Invented in 1817."}]


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
                description="history of the bicycle",
                assigned_specialist=Specialist.RESEARCHER,
                expected_output="findings",
                complexity=1,
            )
        ],
    )


def _passed() -> ReviewResult:
    return ReviewResult(passed=True, score=0.9, feedback="good")


def _failed() -> ReviewResult:
    return ReviewResult(passed=False, score=0.1, feedback="add dates")


def test_passing_review_does_not_retry() -> None:
    reg = _registry()
    provider = ScriptedProvider(_plan(), [_passed()])
    state = run_task(
        provider, Task(description="x"), registry=reg, memory_store=NullMemoryStore()
    )

    assert state["review"].passed is True
    # researcher ran exactly once — no retry
    assert sum(i.tool == "web_search" for i in reg.invocations) == 1


def test_failed_review_retries_then_passes() -> None:
    reg = _registry()
    provider = ScriptedProvider(_plan(), [_failed(), _passed()])
    state = run_task(
        provider, Task(description="x"), registry=reg, memory_store=NullMemoryStore()
    )

    assert state["review"].passed is True
    assert sum(i.tool == "web_search" for i in reg.invocations) == 2  # one retry


def test_retry_cap_halts_always_failing_loop() -> None:
    reg = _registry()
    provider = ScriptedProvider(_plan(), [_failed()])  # always fails
    state = run_task(
        provider, Task(description="x"), registry=reg, memory_store=NullMemoryStore()
    )

    assert state["review"].passed is False
    # capped: at most 2 attempts, and the pipeline still produced a result
    assert sum(i.tool == "web_search" for i in reg.invocations) == 2
    assert isinstance(state["result"], str) and state["result"]

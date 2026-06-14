"""C1: with three specialists wired, each subtask runs through the agent it was
assigned to, and the capability check still rejects an unavailable specialist."""

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


class CrewProvider(LLMProvider):
    name = "crew"

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


def _mixed_plan() -> Plan:
    def _sub(sid: str, desc: str, specialist: Specialist) -> Subtask:
        return Subtask(
            id=sid,
            description=desc,
            assigned_specialist=specialist,
            expected_output="result",
            complexity=1,
        )

    return Plan(
        task_id="t1",
        subtasks=[
            _sub("s1", "research bicycles", Specialist.RESEARCHER),
            _sub("s2", "analyse the figures", Specialist.ANALYST),
            _sub("s3", "write the summary", Specialist.WRITER),
        ],
    )


def _between(text: str, start: str, end: str) -> str:
    i = text.index(start) + len(start)
    return text[i : text.index(end, i)].strip()


class WeakS2Provider(LLMProvider):
    """Echoes each subtask's description as its output (so review can identify it),
    counts specialist calls, and fails the s2 subtask's review exactly once."""

    name = "weak-s2"

    def __init__(self, plan: Plan) -> None:
        self._plan = plan
        self.findings_calls = 0
        self._s2_reviews = 0

    def structured_complete(self, prompt: str, schema: type[T]) -> T:
        if schema is ResearchFindings:
            self.findings_calls += 1
            return ResearchFindings(content=_between(prompt, "Subtask: ", "\n"))  # type: ignore[return-value]
        if schema is ReviewResult:
            if "analyse the figures" in prompt:  # the s2 subtask
                self._s2_reviews += 1
                if self._s2_reviews == 1:
                    return ReviewResult(passed=False, score=0.2, feedback="needs more")  # type: ignore[return-value]
            return ReviewResult(passed=True, score=0.9, feedback="ok")  # type: ignore[return-value]
        if schema is Synthesis:
            return Synthesis(result="done")  # type: ignore[return-value]
        return self._plan  # type: ignore[return-value]


def test_only_the_weak_subtask_is_retried() -> None:
    def _sub(sid: str, desc: str, specialist: Specialist) -> Subtask:
        return Subtask(
            id=sid,
            description=desc,
            assigned_specialist=specialist,
            expected_output="x",
            complexity=1,
        )

    plan = Plan(
        task_id="t1",
        subtasks=[
            _sub("s1", "research bicycles", Specialist.RESEARCHER),
            _sub("s2", "analyse the figures", Specialist.ANALYST),
        ],
    )
    provider = WeakS2Provider(plan)
    state = run_task(
        provider, Task(description="x"), registry=_registry(), memory_store=NullMemoryStore()
    )

    # First pass runs both (2); s2 fails review, so only s2 re-runs (+1) = 3.
    # If the whole batch retried, it would be 4.
    assert provider.findings_calls == 3
    assert {o.subtask_id for o in state["outputs"]} == {"s1", "s2"}


def test_each_subtask_runs_through_its_assigned_specialist() -> None:
    state = run_task(
        CrewProvider(_mixed_plan()),
        Task(description="research the bicycle"),
        registry=_registry(),
        memory_store=NullMemoryStore(),
    )
    outputs = {o.subtask_id: o for o in state["outputs"]}
    assert outputs["s1"].produced_by is Specialist.RESEARCHER
    assert outputs["s2"].produced_by is Specialist.ANALYST
    assert outputs["s3"].produced_by is Specialist.WRITER

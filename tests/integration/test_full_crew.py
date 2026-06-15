"""C1: with three specialists wired, each subtask runs through the agent it was
assigned to, and the capability check still rejects an unavailable specialist."""

from __future__ import annotations

from typing import Any

import pytest

from foreman.graph import run_task
from foreman.llm.base import LLMProvider, T
from foreman.schemas import (
    AnalysisCode,
    Plan,
    ResearchFindings,
    ReviewResult,
    Specialist,
    Subtask,
    Synthesis,
    Task,
)
from foreman.tools import CodeExecutionTool, ToolRegistry, WebSearchTool
from foreman.tools.code_exec import FakeSandbox
from tests.support import NullMemoryStore


class CrewProvider(LLMProvider):
    name = "crew"

    def __init__(self, plan: Plan) -> None:
        self._plan = plan

    def structured_complete(self, prompt: str, schema: type[T]) -> T:
        if schema is AnalysisCode:
            return AnalysisCode(code="print(42)")  # type: ignore[return-value]
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
    reg.register(CodeExecutionTool(FakeSandbox(stdout="42\n")))
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
        if schema is AnalysisCode:
            return AnalysisCode(code="print(1)")  # type: ignore[return-value]
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


class UpstreamEchoProvider(LLMProvider):
    """s1 (researcher) emits a marker; s2 (writer, depends on s1) reports whether it
    received s1's output as upstream — which is only true if the wave ran s1 first and
    injected its output into s2's payload."""

    name = "upstream-echo"

    def __init__(self, plan: Plan) -> None:
        self._plan = plan

    def structured_complete(self, prompt: str, schema: type[T]) -> T:
        if schema is ResearchFindings:
            if "You are a writer" in prompt:  # the dependent s2 subtask
                seen = "MARKER_S1" in prompt
                return ResearchFindings(content=f"s2_saw_s1={seen}")  # type: ignore[return-value]
            return ResearchFindings(content="MARKER_S1")  # s1
        if schema is ReviewResult:
            return ReviewResult(passed=True, score=0.9, feedback="ok")  # type: ignore[return-value]
        if schema is Synthesis:
            return Synthesis(result="done")  # type: ignore[return-value]
        return self._plan  # type: ignore[return-value]


def _dependent_plan() -> Plan:
    def _sub(sid: str, desc: str, spec: Specialist, deps: list[str]) -> Subtask:
        return Subtask(
            id=sid,
            description=desc,
            assigned_specialist=spec,
            expected_output="x",
            complexity=1,
            dependencies=deps,
        )

    return Plan(
        task_id="t1",
        subtasks=[
            _sub("s1", "research bicycles", Specialist.RESEARCHER, []),
            _sub("s2", "write the summary", Specialist.WRITER, ["s1"]),
        ],
    )


def test_dependent_subtask_receives_its_upstream_output() -> None:
    state = run_task(
        UpstreamEchoProvider(_dependent_plan()),
        Task(description="x"),
        registry=_registry(),
        memory_store=NullMemoryStore(),
    )
    outputs = {o.subtask_id: o for o in state["outputs"]}
    assert outputs["s1"].content == "MARKER_S1"
    # s2 ran in a later wave and saw s1's output injected as upstream.
    assert outputs["s2"].content == "s2_saw_s1=True"


@pytest.mark.requires_redis
def test_fan_out_over_a_real_redis_broker() -> None:
    """The same fan-out, but eager off and dispatched through a real Redis broker to
    an actual worker — proving the JSON task payloads round-trip over the wire and the
    group gathers. (A same-process worker thread still picks up the in-process context,
    so this exercises the broker without needing live providers.)"""
    from celery.contrib.testing.worker import start_worker

    from foreman.workers.celery_app import app

    app.conf.task_always_eager = False
    with start_worker(app, perform_ping_check=False, shutdown_timeout=30):
        state = run_task(
            CrewProvider(_mixed_plan()),
            Task(description="research the bicycle"),
            registry=_registry(),
            memory_store=NullMemoryStore(),
        )
    outputs = {o.subtask_id: o for o in state["outputs"]}
    assert set(outputs) == {"s1", "s2", "s3"}
    assert outputs["s2"].produced_by is Specialist.ANALYST

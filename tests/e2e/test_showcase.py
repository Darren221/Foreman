"""C7 checkpoint (headless): the Phase 5 showcase end to end in eager mode.

A dependent three-step plan — research -> analyse (via the sandbox) -> write — runs
through the C-orchestrated waves: each specialist sees its upstream's output, the
analyst computes via the code sandbox, the reviewer catches the writer's section and
only that subtask is retried, and a memory is written so a repeat run recalls it.

This is the CI proof of the distributed slice (eager Celery, fakes, real Chroma with a
deterministic embedder). The live, paid run across the real stack is the operator demo.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from foreman.graph import run_task
from foreman.llm.base import LLMProvider, T
from foreman.memory import ChromaMemoryStore
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
from tests.support import FakeEmbedder


class ShowcaseProvider(LLMProvider):
    """Drives the showcase: emits the dependent plan, threads upstream markers so we
    can prove each step saw the previous one's output, and fails the writer's review
    exactly once so the reviewer-catch-and-retry path runs."""

    name = "showcase"

    def __init__(self, plan: Plan) -> None:
        self._plan = plan
        self.prompts: list[str] = []
        self.analyst_saw_research = False
        self.writer_review_count = 0

    def structured_complete(self, prompt: str, schema: type[T]) -> T:
        self.prompts.append(prompt)
        if schema is Plan:
            return self._plan  # type: ignore[return-value]
        if schema is AnalysisCode:
            # The analyst's code prompt carries its upstream (the research output).
            self.analyst_saw_research = "S1_RESEARCH" in prompt
            return AnalysisCode(code="print('GROWTH')")  # type: ignore[return-value]
        if schema is ResearchFindings:
            if "Search results:" in prompt:  # the researcher (s1)
                return ResearchFindings(content="S1_RESEARCH")  # type: ignore[return-value]
            if "Program output:" in prompt:  # the analyst's write-up (s2)
                return ResearchFindings(content="S2_ANALYSIS")  # type: ignore[return-value]
            # the writer (s3): its upstream is the analyst's output
            saw = "S2_ANALYSIS" in prompt
            return ResearchFindings(content=f"S3_WRITEUP(saw_analysis={saw})")  # type: ignore[return-value]
        if schema is ReviewResult:
            if "write the recommendation" in prompt:  # the writer's subtask
                self.writer_review_count += 1
                if self.writer_review_count == 1:
                    return ReviewResult(passed=False, score=0.2, feedback="too thin")  # type: ignore[return-value]
            return ReviewResult(passed=True, score=0.9, feedback="ok")  # type: ignore[return-value]
        if schema is Synthesis:
            return Synthesis(result="final recommendation")  # type: ignore[return-value]
        raise AssertionError(f"unexpected schema {schema!r}")


class FakeBackend:
    def search(self, query: str, max_results: int) -> list[dict[str, Any]]:
        return [{"title": "t", "url": "u", "content": "c"}]


def _registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(WebSearchTool(FakeBackend()))
    reg.register(CodeExecutionTool(FakeSandbox(stdout="GROWTH\n")))
    return reg


def _showcase_plan() -> Plan:
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
            _sub("s1", "research the frameworks", Specialist.RESEARCHER, []),
            _sub("s2", "analyse the growth figures", Specialist.ANALYST, ["s1"]),
            _sub("s3", "write the recommendation", Specialist.WRITER, ["s2"]),
        ],
    )


def test_showcase_runs_end_to_end(tmp_path: Path) -> None:
    store = ChromaMemoryStore(tmp_path / "mem", FakeEmbedder())
    provider = ShowcaseProvider(_showcase_plan())

    state = run_task(
        provider,
        Task(description="recommend a python web framework"),
        registry=_registry(),
        memory_store=store,
    )
    outputs = {o.subtask_id: o for o in state["outputs"]}

    # All three specialists ran, each through its assigned agent.
    assert {o.subtask_id for o in state["outputs"]} == {"s1", "s2", "s3"}
    assert outputs["s2"].produced_by is Specialist.ANALYST

    # Dependency data-flow across the waves: analyst saw the research, writer saw the
    # analysis — the C-orchestrated chain end to end.
    assert provider.analyst_saw_research is True
    assert outputs["s3"].content == "S3_WRITEUP(saw_analysis=True)"

    # The analyst computed via the sandbox.
    assert "code_execution" in outputs["s2"].tools_used

    # The reviewer caught the writer's section and only that subtask was retried.
    assert provider.writer_review_count >= 2
    assert state["result"] == "final recommendation"


def test_showcase_repeat_recalls_memory(tmp_path: Path) -> None:
    store = ChromaMemoryStore(tmp_path / "mem", FakeEmbedder())
    run_task(
        ShowcaseProvider(_showcase_plan()),
        Task(description="recommend a python web framework"),
        registry=_registry(),
        memory_store=store,
    )

    second = ShowcaseProvider(_showcase_plan())
    state = run_task(
        second,
        Task(description="recommend a python web framework in detail"),
        registry=_registry(),
        memory_store=store,
    )
    assert state["retrieved_memories"]
    assert any("recommend a python web framework" in p for p in second.prompts)

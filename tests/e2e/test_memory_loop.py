"""Phase 2 checkpoint: a repeat task recalls the earlier task's memory.

Run task A through the real Chroma store (with a deterministic fake embedder),
then run a similar task B sharing that store and assert B recalled A's memory and
it reached the planning prompt. This exercises the full write->read loop end to
end, not a stubbed store.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from foreman.graph import run_task
from foreman.llm.base import LLMProvider, T
from foreman.memory import ChromaMemoryStore
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
from tests.support import FakeEmbedder


class RecordingProvider(LLMProvider):
    name = "recording"

    def __init__(self, plan: Plan) -> None:
        self._plan = plan
        self.prompts: list[str] = []

    def structured_complete(self, prompt: str, schema: type[T]) -> T:
        self.prompts.append(prompt)
        if schema is ResearchFindings:
            return ResearchFindings(content="findings")  # type: ignore[return-value]
        if schema is ReviewResult:
            return ReviewResult(passed=True, score=0.9, feedback="ok")  # type: ignore[return-value]
        if schema is Synthesis:
            return Synthesis(result="answer")  # type: ignore[return-value]
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


def test_repeat_task_recalls_prior_memory(tmp_path: Path) -> None:
    store = ChromaMemoryStore(tmp_path / "mem", FakeEmbedder())

    # First run writes a memory of the bicycle task.
    run_task(
        RecordingProvider(_plan()),
        Task(description="history of the bicycle"),
        registry=_registry(),
        memory_store=store,
    )

    # A similar second run should recall it and feed it to planning.
    second = RecordingProvider(_plan())
    state = run_task(
        second,
        Task(description="history of the bicycle in detail"),
        registry=_registry(),
        memory_store=store,
    )

    assert state["retrieved_memories"]
    assert state["retrieved_memories"][0].task_description == "history of the bicycle"
    assert any("history of the bicycle" in p for p in second.prompts)

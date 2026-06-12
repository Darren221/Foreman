"""After a task completes, the pipeline records a TaskMemory of it."""

from __future__ import annotations

from typing import Any

from foreman.graph import run_task
from foreman.llm.base import LLMProvider, T
from foreman.memory import MemoryStore
from foreman.schemas import (
    Plan,
    ResearchFindings,
    ReviewResult,
    Specialist,
    Subtask,
    Synthesis,
    Task,
    TaskMemory,
)
from foreman.tools import ToolRegistry, WebSearchTool


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
            return Synthesis(result="the final answer")  # type: ignore[return-value]
        return self._plan  # type: ignore[return-value]


class FakeBackend:
    def search(self, query: str, max_results: int) -> list[dict[str, Any]]:
        return [{"title": "t", "url": "u", "content": "c"}]


class FakeMemoryStore(MemoryStore):
    def __init__(self) -> None:
        self.memories: list[TaskMemory] = []

    def remember(self, memory: TaskMemory) -> None:
        self.memories.append(memory)

    def recall(self, query: str, k: int = 5) -> list[TaskMemory]:
        return list(self.memories)[:k]


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


def test_completed_task_is_remembered() -> None:
    store = FakeMemoryStore()
    run_task(
        CannedProvider(_plan()),
        Task(description="research the history of the bicycle"),
        registry=_registry(),
        memory_store=store,
    )

    assert len(store.memories) == 1
    memory = store.memories[0]
    assert memory.task_description == "research the history of the bicycle"
    assert memory.outcome == "passed"
    assert memory.score == 0.9
    assert "web_search" in memory.tools_used
    assert memory.result_snippet

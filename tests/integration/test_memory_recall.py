"""Before planning, the pipeline recalls similar past tasks and feeds them to
the supervisor — so memory actually influences the plan."""

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


class SeededMemoryStore(MemoryStore):
    def __init__(self, memories: list[TaskMemory]) -> None:
        self._memories = memories

    def remember(self, memory: TaskMemory) -> None:
        self._memories.append(memory)

    def recall(self, query: str, k: int = 5) -> list[TaskMemory]:
        return self._memories[:k]


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


def test_recalled_memory_reaches_the_planning_prompt() -> None:
    seeded = TaskMemory(
        task_description="earlier: full history of the bicycle",
        outcome="passed",
        score=0.95,
        tools_used=["web_search"],
    )
    provider = RecordingProvider(_plan())
    store = SeededMemoryStore([seeded])

    state = run_task(
        provider,
        Task(description="history of bikes"),
        registry=_registry(),
        memory_store=store,
    )

    assert state["retrieved_memories"]
    assert state["retrieved_memories"][0].task_description == seeded.task_description
    # the recalled memory was injected into the planning prompt
    assert any("earlier: full history of the bicycle" in p for p in provider.prompts)

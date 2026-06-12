import pytest

from foreman.agents import Supervisor
from foreman.llm.base import LLMProvider, T
from foreman.schemas import Plan, Specialist, Subtask, Task, TaskMemory


class CannedProvider(LLMProvider):
    """Returns a preset object and records the prompts it was given."""

    name = "canned"

    def __init__(self, response: object) -> None:
        self._response = response
        self.prompts: list[str] = []

    def structured_complete(self, prompt: str, schema: type[T]) -> T:
        self.prompts.append(prompt)
        return self._response  # type: ignore[return-value]


def _plan_with(specialist: Specialist) -> Plan:
    return Plan(
        task_id="t1",
        subtasks=[
            Subtask(
                id="s1",
                description="do the thing",
                assigned_specialist=specialist,
                expected_output="result",
                complexity=1,
            )
        ],
    )


def test_supervisor_returns_valid_plan() -> None:
    expected = _plan_with(Specialist.RESEARCHER)
    supervisor = Supervisor(CannedProvider(expected))
    plan = supervisor.plan(Task(description="research bicycles"))
    assert plan.subtasks[0].assigned_specialist is Specialist.RESEARCHER


def test_supervisor_rejects_unavailable_specialist() -> None:
    # writer is a valid enum member but not wired in Phase 1.
    supervisor = Supervisor(
        CannedProvider(_plan_with(Specialist.WRITER)),
        available_specialists={Specialist.RESEARCHER},
    )
    with pytest.raises(ValueError, match="not available"):
        supervisor.plan(Task(description="anything"))


def test_retrieved_memories_are_injected_into_the_planning_prompt() -> None:
    provider = CannedProvider(_plan_with(Specialist.RESEARCHER))
    memory = TaskMemory(
        task_description="prior task: history of the bicycle",
        outcome="passed",
        score=0.9,
        tools_used=["web_search"],
    )
    Supervisor(provider).plan(Task(description="history of bikes"), memories=[memory])

    assert any("history of the bicycle" in p for p in provider.prompts)


def test_planning_works_with_no_memories() -> None:
    provider = CannedProvider(_plan_with(Specialist.RESEARCHER))
    plan = Supervisor(provider).plan(Task(description="anything"), memories=[])
    assert plan.subtasks[0].assigned_specialist is Specialist.RESEARCHER

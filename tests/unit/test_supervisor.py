import pytest

from foreman.agents import Supervisor
from foreman.llm.base import LLMProvider, T
from foreman.schemas import Plan, Specialist, Subtask, Task


class CannedProvider(LLMProvider):
    """Returns a preset object, ignoring prompt and schema."""

    name = "canned"

    def __init__(self, response: object) -> None:
        self._response = response

    def structured_complete(self, prompt: str, schema: type[T]) -> T:
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

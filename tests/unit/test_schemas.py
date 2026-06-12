import pytest
from pydantic import ValidationError

from foreman.schemas import Plan, Specialist, Subtask, Task


def _subtask(sid: str, deps: list[str] | None = None) -> Subtask:
    return Subtask(
        id=sid,
        description=f"do {sid}",
        assigned_specialist=Specialist.RESEARCHER,
        required_inputs=[],
        expected_output="a result",
        complexity=1,
        dependencies=deps or [],
    )


def test_valid_plan_constructs() -> None:
    task = Task(description="research the topic")
    plan = Plan(
        task_id=task.id,
        subtasks=[_subtask("a"), _subtask("b", deps=["a"])],
    )
    assert len(plan.subtasks) == 2


def test_plan_rejects_unknown_dependency() -> None:
    with pytest.raises(ValidationError):
        Plan(task_id="t1", subtasks=[_subtask("a", deps=["ghost"])])


def test_plan_rejects_cyclic_dependencies() -> None:
    with pytest.raises(ValidationError):
        Plan(
            task_id="t1",
            subtasks=[_subtask("a", deps=["b"]), _subtask("b", deps=["a"])],
        )


def test_unknown_specialist_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Subtask(
            id="a",
            description="x",
            assigned_specialist="astronaut",  # type: ignore[arg-type]
            required_inputs=[],
            expected_output="y",
            complexity=1,
            dependencies=[],
        )

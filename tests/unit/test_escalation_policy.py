"""H2: the escalation policy decides, from graph state, whether to hand control
to a human — which trigger fired, at what approval level, with what context.

A table of (state -> expected escalation/level or None) exercises all 5 triggers
and proves a clean run escalates to no one."""

from __future__ import annotations

from foreman.graph.state import GraphState
from foreman.hitl import ApprovalLevel, Escalation, EscalationPolicy, EscalationTrigger
from foreman.schemas import (
    Plan,
    ReviewResult,
    Specialist,
    SpecialistOutput,
    Subtask,
    Task,
    TaskMemory,
)


def _plan(confidence: float = 1.0) -> Plan:
    return Plan(
        task_id="t1",
        confidence=confidence,
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


def _state(**overrides: object) -> GraphState:
    """A clean, non-escalating baseline state; override one key per trigger."""
    state: GraphState = {
        "task": Task(description="research the bicycle"),
        "plan": _plan(),
        "outputs": [SpecialistOutput(subtask_id="s1", content="found it")],
        "review": ReviewResult(passed=True, score=0.95, feedback="ok"),
        "attempts": 1,
        "retrieved_memories": [],
    }
    state.update(overrides)  # type: ignore[typeddict-item]
    return state


def test_clean_run_does_not_escalate() -> None:
    assert EscalationPolicy().evaluate(_state()) is None


def test_explicit_require_approval_escalates_to_approve_plan() -> None:
    esc = EscalationPolicy().evaluate(_state(task=Task(description="x", require_approval=True)))
    assert esc is not None
    assert esc.trigger is EscalationTrigger.REQUIRE_APPROVAL
    assert esc.level is ApprovalLevel.APPROVE_PLAN


def test_sensitive_task_escalates_to_approve_action() -> None:
    esc = EscalationPolicy().evaluate(_state(task=Task(description="x", sensitive=True)))
    assert esc is not None
    assert esc.trigger is EscalationTrigger.SENSITIVE
    assert esc.level is ApprovalLevel.APPROVE_ACTION


def test_low_plan_confidence_escalates_to_approve_plan() -> None:
    esc = EscalationPolicy(confidence_threshold=0.5).evaluate(_state(plan=_plan(confidence=0.2)))
    assert esc is not None
    assert esc.trigger is EscalationTrigger.LOW_CONFIDENCE
    assert esc.level is ApprovalLevel.APPROVE_PLAN


def test_retry_cap_exhausted_escalates_to_take_over() -> None:
    esc = EscalationPolicy(max_attempts=2).evaluate(
        _state(review=ReviewResult(passed=False, score=0.3, feedback="weak"), attempts=2)
    )
    assert esc is not None
    assert esc.trigger is EscalationTrigger.RETRY_EXHAUSTED
    assert esc.level is ApprovalLevel.TAKE_OVER


def test_marginal_review_score_escalates_to_notify() -> None:
    # Passed the bar but below the comfort floor: flag it, don't block.
    esc = EscalationPolicy(review_floor=0.75).evaluate(
        _state(review=ReviewResult(passed=True, score=0.6, feedback="thin"))
    )
    assert esc is not None
    assert esc.trigger is EscalationTrigger.LOW_REVIEW_SCORE
    assert esc.level is ApprovalLevel.NOTIFY


def test_failing_review_below_cap_does_not_escalate() -> None:
    # Still has retries left — the loop handles it, no human needed yet.
    assert (
        EscalationPolicy(max_attempts=2).evaluate(
            _state(review=ReviewResult(passed=False, score=0.3, feedback="weak"), attempts=1)
        )
        is None
    )


def test_escalation_carries_a_complete_context_packet() -> None:
    memory = TaskMemory(task_description="past bike task", outcome="passed", score=0.9)
    task = Task(description="research bikes", sensitive=True)
    esc = EscalationPolicy().evaluate(_state(task=task, retrieved_memories=[memory]))
    assert isinstance(esc, Escalation)
    assert esc.task == task
    assert esc.plan is not None
    assert esc.completed_outputs and esc.completed_outputs[0].subtask_id == "s1"
    assert esc.memories == [memory]
    assert esc.proposed_action
    assert esc.reason


def test_most_severe_trigger_wins_when_several_apply() -> None:
    # A run that is sensitive, low-confidence, and explicitly gated should report
    # the most blocking trigger (sensitive action) — precedence is deterministic.
    esc = EscalationPolicy(confidence_threshold=0.5).evaluate(
        _state(
            task=Task(description="x", sensitive=True, require_approval=True),
            plan=_plan(confidence=0.1),
        )
    )
    assert esc is not None
    assert esc.trigger is EscalationTrigger.SENSITIVE

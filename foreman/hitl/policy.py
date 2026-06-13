"""The escalation policy: decide, from graph state, when to involve a human.

This is the *when* and *what level* of human-in-the-loop — a pure function over
graph state, with no side effects. The runner (H3) carries the resulting
`Escalation` into the approval queue and the interrupt; the policy itself only
classifies. Keeping it pure makes the five triggers exhaustively unit-testable.

Five triggers, mapped to four approval levels:

- sensitive operation        -> APPROVE_ACTION  (confirm the specific action)
- retry cap exhausted        -> TAKE_OVER       (agents are stuck; human supplies it)
- explicit `require_approval` -> APPROVE_PLAN    (user asked to gate the work)
- low plan confidence        -> APPROVE_PLAN    (shaky plan; review before work)
- marginal review score      -> NOTIFY          (passed but thin; flag, don't block)

When several apply, the most blocking one wins; precedence is the order above.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

from foreman.graph.state import GraphState
from foreman.schemas import Plan, SpecialistOutput, Task, TaskMemory


class ApprovalLevel(StrEnum):
    """How much control the human takes, in increasing order of intervention."""

    NOTIFY = "notify"
    APPROVE_ACTION = "approve_action"
    APPROVE_PLAN = "approve_plan"
    TAKE_OVER = "take_over"


class EscalationTrigger(StrEnum):
    """What caused the escalation."""

    SENSITIVE = "sensitive"
    RETRY_EXHAUSTED = "retry_exhausted"
    REQUIRE_APPROVAL = "require_approval"
    LOW_CONFIDENCE = "low_confidence"
    LOW_REVIEW_SCORE = "low_review_score"


class Escalation(BaseModel):
    """A request for human input, with everything the operator needs to decide."""

    trigger: EscalationTrigger
    level: ApprovalLevel
    reason: str
    proposed_action: str
    task: Task
    plan: Plan | None = None
    completed_outputs: list[SpecialistOutput] = Field(default_factory=list)
    memories: list[TaskMemory] = Field(default_factory=list)


class EscalationPolicy:
    """Evaluates graph state against the escalation triggers.

    Thresholds are constructor knobs (like the reviewer's pass threshold) so they
    can be tuned without touching the trigger logic. `review_floor` sits *above*
    the reviewer's pass bar: a run that passed review but landed below the floor
    is worth notifying a human about without blocking it.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.5,
        review_floor: float = 0.75,
        max_attempts: int = 2,
    ) -> None:
        self._confidence_threshold = confidence_threshold
        self._review_floor = review_floor
        self._max_attempts = max_attempts

    def evaluate(self, state: GraphState) -> Escalation | None:
        task = state["task"]
        plan = state.get("plan")
        review = state.get("review")
        attempts = state.get("attempts", 0)

        if task.sensitive:
            return self._build(
                state,
                EscalationTrigger.SENSITIVE,
                ApprovalLevel.APPROVE_ACTION,
                "task is flagged as a sensitive operation",
                "perform a sensitive operation",
            )

        if review is not None and not review.passed and attempts >= self._max_attempts:
            return self._build(
                state,
                EscalationTrigger.RETRY_EXHAUSTED,
                ApprovalLevel.TAKE_OVER,
                f"review still failing after {attempts} attempt(s)",
                "deliver a best-effort result after exhausting retries",
            )

        if task.require_approval:
            return self._build(
                state,
                EscalationTrigger.REQUIRE_APPROVAL,
                ApprovalLevel.APPROVE_PLAN,
                "task explicitly requires approval",
                self._execute_action(plan),
            )

        if plan is not None and plan.confidence < self._confidence_threshold:
            return self._build(
                state,
                EscalationTrigger.LOW_CONFIDENCE,
                ApprovalLevel.APPROVE_PLAN,
                f"plan confidence {plan.confidence:.2f} below "
                f"threshold {self._confidence_threshold:.2f}",
                self._execute_action(plan),
            )

        if review is not None and review.passed and review.score < self._review_floor:
            return self._build(
                state,
                EscalationTrigger.LOW_REVIEW_SCORE,
                ApprovalLevel.NOTIFY,
                f"review score {review.score:.2f} below floor {self._review_floor:.2f}",
                "deliver the result despite a marginal review score",
            )

        return None

    @staticmethod
    def _execute_action(plan: Plan | None) -> str:
        count = len(plan.subtasks) if plan else 0
        return f"execute the plan ({count} subtask(s))"

    @staticmethod
    def _build(
        state: GraphState,
        trigger: EscalationTrigger,
        level: ApprovalLevel,
        reason: str,
        proposed_action: str,
    ) -> Escalation:
        return Escalation(
            trigger=trigger,
            level=level,
            reason=reason,
            proposed_action=proposed_action,
            task=state["task"],
            plan=state.get("plan"),
            completed_outputs=state.get("outputs") or [],
            memories=state.get("retrieved_memories") or [],
        )

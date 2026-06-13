"""The logic beneath the review UI, kept free of Streamlit so it can be tested.

Two concerns live here: which decisions an operator may make at a given approval
level, and turning a chosen decision into a `Decision` that resumes the run. The
Streamlit page renders these; it owns no decision logic of its own.
"""

from __future__ import annotations

from foreman.hitl.policy import ApprovalLevel
from foreman.hitl.queue import Decision, DecisionKind
from foreman.hitl.runner import Runner, RunResult
from foreman.schemas import Plan

# The controls each level exposes. Approve-style levels can be approved, rejected
# (redo with feedback), or have their plan replaced; a take-over asks for the
# human's own output, or accepts the agents' best effort.
_DECISIONS_BY_LEVEL: dict[ApprovalLevel, tuple[DecisionKind, ...]] = {
    ApprovalLevel.APPROVE_PLAN: (DecisionKind.APPROVE, DecisionKind.REJECT, DecisionKind.MODIFY),
    ApprovalLevel.APPROVE_ACTION: (DecisionKind.APPROVE, DecisionKind.REJECT),
    ApprovalLevel.TAKE_OVER: (DecisionKind.TAKE_OVER, DecisionKind.APPROVE),
    ApprovalLevel.NOTIFY: (DecisionKind.APPROVE,),
}


def available_decisions(level: ApprovalLevel) -> tuple[DecisionKind, ...]:
    """The decision kinds an operator may make for an escalation at `level`."""
    return _DECISIONS_BY_LEVEL[level]


def build_decision(
    kind: DecisionKind,
    *,
    feedback: str = "",
    output: str = "",
    plan: Plan | None = None,
) -> Decision:
    """Build a Decision carrying only the field its kind uses, so a stray value
    left in a form field can't leak into an unrelated decision."""
    if kind is DecisionKind.REJECT:
        return Decision(kind=kind, feedback=feedback)
    if kind is DecisionKind.TAKE_OVER:
        return Decision(kind=kind, output=output)
    if kind is DecisionKind.MODIFY:
        return Decision(kind=kind, plan=plan)
    return Decision(kind=kind)


def submit_decision(runner: Runner, approval_id: str, decision: Decision) -> RunResult:
    """Apply the operator's decision and resume the run from its checkpoint."""
    return runner.resume(approval_id, decision)

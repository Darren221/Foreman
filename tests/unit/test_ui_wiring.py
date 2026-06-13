"""H4: the testable seam under the Streamlit review UI — which decisions a level
offers, how a form maps to a Decision, and that submitting resumes the run. The
Streamlit page itself is a thin renderer, verified by manual demo."""

from __future__ import annotations

from foreman.hitl import ApprovalLevel, Decision, DecisionKind, RunResult
from foreman.ui.wiring import available_decisions, build_decision, submit_decision


def test_approve_levels_offer_approve_and_reject() -> None:
    for level in (ApprovalLevel.APPROVE_PLAN, ApprovalLevel.APPROVE_ACTION):
        kinds = available_decisions(level)
        assert DecisionKind.APPROVE in kinds
        assert DecisionKind.REJECT in kinds


def test_take_over_level_offers_take_over() -> None:
    assert DecisionKind.TAKE_OVER in available_decisions(ApprovalLevel.TAKE_OVER)


def test_build_decision_populates_only_the_relevant_field() -> None:
    assert build_decision(DecisionKind.REJECT, feedback="fix it").feedback == "fix it"
    assert build_decision(DecisionKind.TAKE_OVER, output="mine").output == "mine"
    # An approve decision ignores feedback/output the form may have lying around.
    approve = build_decision(DecisionKind.APPROVE, feedback="x", output="y")
    assert approve.kind is DecisionKind.APPROVE
    assert approve.feedback == "" and approve.output == ""


class _FakeRunner:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Decision]] = []

    def resume(self, approval_id: str, decision: Decision) -> RunResult:
        self.calls.append((approval_id, decision))
        return RunResult(status="completed", result="done")


def test_submit_decision_resumes_the_run() -> None:
    runner = _FakeRunner()
    decision = Decision(kind=DecisionKind.APPROVE)
    result = submit_decision(runner, "a1", decision)  # type: ignore[arg-type]
    assert runner.calls == [("a1", decision)]
    assert result.status == "completed"

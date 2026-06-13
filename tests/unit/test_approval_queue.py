"""H3: the approval queue persists pending escalations and the human's decision,
so a run paused for approval can be found and resolved later — across processes
(a reopened queue still sees the pending item)."""

from __future__ import annotations

from pathlib import Path

import pytest

from foreman.hitl import (
    ApprovalLevel,
    ApprovalQueue,
    Decision,
    DecisionKind,
    Escalation,
    EscalationTrigger,
)
from foreman.schemas import Task


def _escalation() -> Escalation:
    return Escalation(
        trigger=EscalationTrigger.SENSITIVE,
        level=ApprovalLevel.APPROVE_ACTION,
        reason="sensitive operation",
        proposed_action="perform a sensitive operation",
        task=Task(description="wire the funds", sensitive=True),
    )


def test_enqueue_then_pending_round_trips(tmp_path: Path) -> None:
    queue = ApprovalQueue(tmp_path / "approvals.sqlite")
    approval_id = queue.enqueue(_escalation(), thread_id="run-1")
    pending = queue.pending()
    assert len(pending) == 1
    assert pending[0].id == approval_id
    assert pending[0].thread_id == "run-1"
    assert pending[0].escalation.trigger is EscalationTrigger.SENSITIVE
    assert pending[0].resolved is False
    queue.close()


def test_resolve_removes_from_pending_and_records_decision(tmp_path: Path) -> None:
    queue = ApprovalQueue(tmp_path / "approvals.sqlite")
    approval_id = queue.enqueue(_escalation(), thread_id="run-1")
    queue.resolve(approval_id, Decision(kind=DecisionKind.REJECT, feedback="too risky"))
    assert queue.pending() == []
    resolved = queue.get(approval_id)
    assert resolved is not None
    assert resolved.resolved is True
    assert resolved.decision is not None
    assert resolved.decision.kind is DecisionKind.REJECT
    assert resolved.decision.feedback == "too risky"
    queue.close()


def test_pending_survives_reopen(tmp_path: Path) -> None:
    path = tmp_path / "approvals.sqlite"
    queue = ApprovalQueue(path)
    approval_id = queue.enqueue(_escalation(), thread_id="run-1")
    queue.close()

    reopened = ApprovalQueue(path)
    assert [p.id for p in reopened.pending()] == [approval_id]
    reopened.close()


def test_resolving_unknown_approval_raises(tmp_path: Path) -> None:
    queue = ApprovalQueue(tmp_path / "approvals.sqlite")
    with pytest.raises(ValueError):
        queue.resolve("does-not-exist", Decision(kind=DecisionKind.APPROVE))
    queue.close()

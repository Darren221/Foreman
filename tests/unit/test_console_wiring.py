"""The submit-and-watch console's wiring: parse an API task response into the few
display values the page renders, with no Streamlit and no live API."""

from __future__ import annotations

from foreman.ui.console_wiring import (
    TaskView,
    body_text,
    is_pending,
    task_view_from_payload,
)


def test_parses_a_completed_task() -> None:
    view = task_view_from_payload({"id": "t1", "status": "completed", "result": "the answer"})
    assert view == TaskView(id="t1", status="completed", result="the answer")
    assert not is_pending(view)
    assert body_text(view) == "the answer"


def test_parses_a_pending_task_and_summarises_the_escalation() -> None:
    view = task_view_from_payload(
        {
            "id": "t2",
            "status": "pending",
            "approval_id": "a9",
            "escalation": {
                "trigger": "sensitive_action",
                "level": "approve_action",
                "reason": "writes to prod",
            },
        }
    )
    assert is_pending(view)
    assert view.approval_id == "a9"
    assert view.escalation_summary == "sensitive_action → approve_action: writes to prod"
    assert "review UI" in body_text(view)


def test_pending_without_an_escalation_still_renders() -> None:
    # Defensive: a pending status with no escalation payload must not blow up the page.
    view = task_view_from_payload({"id": "t3", "status": "pending"})
    assert is_pending(view)
    assert view.escalation_summary is None
    assert "Paused for approval" in body_text(view)


def test_completed_with_no_result_text_has_a_fallback() -> None:
    view = task_view_from_payload({"id": "t4", "status": "completed", "result": None})
    assert "no result text" in body_text(view)

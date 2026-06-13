"""Streamlit review surface: the operator's window into paused runs.

Thin by design. It lists the approval queue, shows the context behind each paused
run, and on a decision calls the runner to resume from the checkpoint. Every
non-UI concern lives in `wiring.py` and the runner/queue, so this file is just
rendering and the wiring between widgets and that logic.

Run it with:  streamlit run foreman/ui/review.py
"""

from __future__ import annotations

import sqlite3

import streamlit as st
from langgraph.checkpoint.sqlite import SqliteSaver

from foreman.config import Settings
from foreman.hitl import ApprovalQueue, DecisionKind, Runner
from foreman.hitl.queue import PendingApproval
from foreman.llm import select_provider
from foreman.memory import build_default_memory_store
from foreman.schemas import Plan
from foreman.tools import build_default_registry
from foreman.ui.wiring import available_decisions, build_decision, submit_decision


@st.cache_resource  # type: ignore[untyped-decorator, unused-ignore]  # untyped only when streamlit absent
def _runner_and_queue() -> tuple[Runner, ApprovalQueue]:
    """Build the runner and queue once and reuse them across reruns. The queue
    and checkpointer both point at the SQLite files the headless runner wrote, so
    this UI resumes the very runs another process paused."""
    settings = Settings()
    queue = ApprovalQueue(settings.approval_path)
    saver = SqliteSaver(
        sqlite3.connect(str(settings.checkpoint_path), check_same_thread=False)
    )
    runner = Runner(
        provider=select_provider(settings),
        registry=build_default_registry(settings),
        memory_store=build_default_memory_store(settings),
        checkpointer=saver,
        queue=queue,
    )
    return runner, queue


def _render_context(item: PendingApproval) -> None:
    escalation = item.escalation
    st.subheader(f"{escalation.trigger.value} → {escalation.level.value}")
    st.write(f"**Task:** {escalation.task.description}")
    st.write(f"**Why escalated:** {escalation.reason}")
    st.write(f"**Proposed action:** {escalation.proposed_action}")
    if escalation.plan is not None:
        st.write("**Plan**")
        for subtask in escalation.plan.subtasks:
            st.write(f"- `{subtask.assigned_specialist.value}` — {subtask.description}")
    if escalation.completed_outputs:
        st.write("**Work so far**")
        for output in escalation.completed_outputs:
            st.write(f"- {output.content}")
    if escalation.memories:
        st.write("**Recalled memories**")
        for memory in escalation.memories:
            st.write(f"- {memory.task_description} ({memory.outcome})")


def _render_controls(runner: Runner, item: PendingApproval) -> None:
    kind = st.selectbox(
        "Decision",
        available_decisions(item.escalation.level),
        format_func=lambda k: k.value,
        key=f"kind-{item.id}",
    )
    feedback = ""
    output = ""
    plan: Plan | None = None
    if kind is DecisionKind.REJECT:
        feedback = st.text_area("Feedback for the redo", key=f"fb-{item.id}")
    elif kind is DecisionKind.TAKE_OVER:
        output = st.text_area("Your result (the agents stand down)", key=f"out-{item.id}")
    elif kind is DecisionKind.MODIFY:
        raw = st.text_area("Replacement plan (JSON)", key=f"plan-{item.id}")
        if raw.strip():
            try:
                plan = Plan.model_validate_json(raw)
            except ValueError as exc:
                st.error(f"Invalid plan JSON: {exc}")
                return

    if st.button("Submit decision", key=f"submit-{item.id}"):
        decision = build_decision(kind, feedback=feedback, output=output, plan=plan)
        result = submit_decision(runner, item.id, decision)
        st.success(f"Run {result.status}.")
        _runner_and_queue.clear()  # drop the cached queue so the list refreshes
        st.rerun()


def main() -> None:
    st.title("Foreman — approval queue")
    runner, queue = _runner_and_queue()
    pending = queue.pending()
    if not pending:
        st.info("No pending approvals.")
        return
    for item in pending:
        with st.container(border=True):
            _render_context(item)
            _render_controls(runner, item)


main()

"""Phase 3 checkpoint: the full human-in-the-loop loop, end to end and headless.

A task escalates and the run pauses; an operator *discovers* it through the
queue (not via an id handed back at submit time, the way the UI sees it) and
resolves it; the run resumes from its checkpoint and finishes, and the queue
drains. The approve case runs the resume in a *separate* runner/queue over the
same SQLite files — proving the decoupled-via-persistence topology: one process
can pause a run and another can pick it up.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver

from foreman.hitl import ApprovalLevel, ApprovalQueue, Decision, DecisionKind, Runner
from foreman.llm.base import LLMProvider, T
from foreman.schemas import (
    Plan,
    ResearchFindings,
    ReviewResult,
    Specialist,
    Subtask,
    Synthesis,
    Task,
)
from foreman.tools import ToolRegistry, WebSearchTool
from tests.support import NullMemoryStore


class CannedProvider(LLMProvider):
    name = "canned"

    def __init__(self, plan: Plan, review_score: float = 0.9) -> None:
        self._plan = plan
        self._review_score = review_score

    def structured_complete(self, prompt: str, schema: type[T]) -> T:
        if schema is ResearchFindings:
            return ResearchFindings(content="findings")  # type: ignore[return-value]
        if schema is ReviewResult:
            return ReviewResult(passed=True, score=self._review_score, feedback="ok")  # type: ignore[return-value]
        if schema is Synthesis:
            return Synthesis(result="the answer")  # type: ignore[return-value]
        return self._plan  # type: ignore[return-value]


class FakeBackend:
    def search(self, query: str, max_results: int) -> list[dict[str, Any]]:
        return [{"title": "t", "url": "u", "content": "c"}]


def _registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(WebSearchTool(FakeBackend()))
    return reg


def _plan() -> Plan:
    return Plan(
        task_id="t1",
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


def _runner(saver: SqliteSaver, queue: ApprovalQueue, review_score: float = 0.9) -> Runner:
    return Runner(
        provider=CannedProvider(_plan(), review_score=review_score),
        registry=_registry(),
        memory_store=NullMemoryStore(),
        checkpointer=saver,
        queue=queue,
    )


def test_operator_approves_a_paused_run_across_processes(tmp_path: Path) -> None:
    db = str(tmp_path / "checkpoints.sqlite")
    queue_path = tmp_path / "approvals.sqlite"

    # Process 1: submit a sensitive task; it pauses at the pre-execution gate.
    with SqliteSaver.from_conn_string(db) as saver:
        queue = ApprovalQueue(queue_path)
        submitted = _runner(saver, queue).submit(
            Task(description="wire the funds", sensitive=True)
        )
        assert submitted.status == "pending"
        queue.close()

    # Process 2: a fresh runner and queue over the same files. The operator finds
    # the approval by listing the queue, approves it, and the run completes.
    with SqliteSaver.from_conn_string(db) as saver:
        queue = ApprovalQueue(queue_path)
        runner = _runner(saver, queue)
        pending = queue.pending()
        assert len(pending) == 1
        assert pending[0].escalation.level is ApprovalLevel.APPROVE_ACTION

        done = runner.resume(pending[0].id, Decision(kind=DecisionKind.APPROVE))
        assert done.status == "completed"
        assert done.result == "the answer"
        assert queue.pending() == []  # the resolved item left the queue
        queue.close()


def test_operator_takes_over_a_run_that_exhausted_its_retries(tmp_path: Path) -> None:
    db = str(tmp_path / "checkpoints.sqlite")
    queue_path = tmp_path / "approvals.sqlite"

    with SqliteSaver.from_conn_string(db) as saver:
        queue = ApprovalQueue(queue_path)
        runner = _runner(saver, queue, review_score=0.2)  # work never clears review

        runner.submit(Task(description="research bicycles"))
        pending = queue.pending()
        assert len(pending) == 1
        assert pending[0].escalation.level is ApprovalLevel.TAKE_OVER

        done = runner.resume(
            pending[0].id, Decision(kind=DecisionKind.TAKE_OVER, output="the operator's answer")
        )
        assert done.status == "completed"
        assert done.result == "the operator's answer"
        assert queue.pending() == []
        queue.close()

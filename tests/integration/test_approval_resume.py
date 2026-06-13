"""H3 end-to-end (headless): a task that triggers escalation pauses with a pending
approval; resolving it resumes the run, and each decision type changes the
outcome as specified — approve continues, modify swaps the plan, reject loops the
work with feedback, take-over injects the human's output and stands the agents
down."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver

from foreman.hitl import (
    ApprovalLevel,
    ApprovalQueue,
    Decision,
    DecisionKind,
    EscalationTrigger,
    Runner,
)
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


def _between(text: str, start: str, end: str) -> str:
    i = text.index(start) + len(start)
    return text[i : text.index(end, i)].strip()


class EchoProvider(LLMProvider):
    """Echoes the work back through the result so a decision's effect is visible:
    findings carry the subtask topic and any feedback; synthesis returns the
    findings verbatim."""

    name = "echo"

    def __init__(self, plan: Plan, review_score: float = 0.9) -> None:
        self._plan = plan
        self._review_score = review_score

    def structured_complete(self, prompt: str, schema: type[T]) -> T:
        if schema is Plan:
            return self._plan  # type: ignore[return-value]
        if schema is ResearchFindings:
            topic = _between(prompt, "Subtask: ", "\n")
            feedback = _between(prompt, "feedback to address (if any): ", "\n")
            return ResearchFindings(content=f"[topic={topic}|fb={feedback}]")  # type: ignore[return-value]
        if schema is ReviewResult:
            # The reviewer recomputes `passed` from the score against its
            # threshold, so the score alone drives pass/fail (and retry exhaustion).
            return ReviewResult(passed=True, score=self._review_score, feedback="ok")  # type: ignore[return-value]
        if schema is Synthesis:
            return Synthesis(result=prompt.split("Findings:\n", 1)[-1].strip())  # type: ignore[return-value]
        raise AssertionError(f"unexpected schema {schema!r}")


class FakeBackend:
    def search(self, query: str, max_results: int) -> list[dict[str, Any]]:
        return [{"title": "t", "url": "u", "content": "c"}]


def _registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(WebSearchTool(FakeBackend()))
    return reg


def _plan(topic: str = "research bicycles") -> Plan:
    return Plan(
        task_id="t1",
        subtasks=[
            Subtask(
                id="s1",
                description=topic,
                assigned_specialist=Specialist.RESEARCHER,
                expected_output="findings",
                complexity=1,
            )
        ],
    )


def _runner(saver: SqliteSaver, queue: ApprovalQueue, plan: Plan) -> Runner:
    return Runner(
        provider=EchoProvider(plan),
        registry=_registry(),
        memory_store=NullMemoryStore(),
        checkpointer=saver,
        queue=queue,
    )


def test_submit_sensitive_task_lands_one_pending(tmp_path: Path) -> None:
    with SqliteSaver.from_conn_string(str(tmp_path / "c.sqlite")) as saver:
        queue = ApprovalQueue(tmp_path / "q.sqlite")
        runner = _runner(saver, queue, _plan())
        res = runner.submit(Task(description="research the bicycle", sensitive=True))
        assert res.status == "pending"
        assert res.escalation is not None
        assert res.escalation.level is ApprovalLevel.APPROVE_ACTION
        assert len(queue.pending()) == 1
        queue.close()


def test_approve_resumes_to_completion(tmp_path: Path) -> None:
    with SqliteSaver.from_conn_string(str(tmp_path / "c.sqlite")) as saver:
        queue = ApprovalQueue(tmp_path / "q.sqlite")
        runner = _runner(saver, queue, _plan())
        pending = runner.submit(Task(description="x", sensitive=True))
        done = runner.resume(pending.approval_id, Decision(kind=DecisionKind.APPROVE))
        assert done.status == "completed"
        assert "fb=none" in (done.result or "")
        assert queue.pending() == []
        queue.close()


def test_reject_loops_the_work_with_feedback(tmp_path: Path) -> None:
    with SqliteSaver.from_conn_string(str(tmp_path / "c.sqlite")) as saver:
        queue = ApprovalQueue(tmp_path / "q.sqlite")
        runner = _runner(saver, queue, _plan())
        pending = runner.submit(Task(description="x", sensitive=True))
        done = runner.resume(
            pending.approval_id, Decision(kind=DecisionKind.REJECT, feedback="needs-dates")
        )
        assert done.status == "completed"
        assert "needs-dates" in (done.result or "")  # feedback reached the redo
        queue.close()


def test_modify_runs_the_modified_plan(tmp_path: Path) -> None:
    with SqliteSaver.from_conn_string(str(tmp_path / "c.sqlite")) as saver:
        queue = ApprovalQueue(tmp_path / "q.sqlite")
        runner = _runner(saver, queue, _plan())
        pending = runner.submit(Task(description="x", sensitive=True))
        done = runner.resume(
            pending.approval_id,
            Decision(kind=DecisionKind.MODIFY, plan=_plan(topic="MODIFIED-TOPIC")),
        )
        assert done.status == "completed"
        assert "MODIFIED-TOPIC" in (done.result or "")
        queue.close()


def test_take_over_injects_human_output(tmp_path: Path) -> None:
    with SqliteSaver.from_conn_string(str(tmp_path / "c.sqlite")) as saver:
        queue = ApprovalQueue(tmp_path / "q.sqlite")
        runner = _runner(saver, queue, _plan())
        pending = runner.submit(Task(description="x", sensitive=True))
        done = runner.resume(
            pending.approval_id, Decision(kind=DecisionKind.TAKE_OVER, output="HUMAN ANSWER")
        )
        assert done.status == "completed"
        assert done.result == "HUMAN ANSWER"
        queue.close()


def test_retry_exhaustion_pauses_post_review_for_take_over(tmp_path: Path) -> None:
    # A plain task (no pre-execution trigger) whose work keeps failing review hits
    # the retry cap and escalates at the *post-review* gate for a human take-over.
    with SqliteSaver.from_conn_string(str(tmp_path / "c.sqlite")) as saver:
        queue = ApprovalQueue(tmp_path / "q.sqlite")
        runner = Runner(
            provider=EchoProvider(_plan(), review_score=0.2),
            registry=_registry(),
            memory_store=NullMemoryStore(),
            checkpointer=saver,
            queue=queue,
        )
        pending = runner.submit(Task(description="research the bicycle"))
        assert pending.status == "pending"
        assert pending.escalation is not None
        assert pending.escalation.trigger is EscalationTrigger.RETRY_EXHAUSTED
        assert pending.escalation.level is ApprovalLevel.TAKE_OVER

        done = runner.resume(
            pending.approval_id, Decision(kind=DecisionKind.TAKE_OVER, output="HUMAN FALLBACK")
        )
        assert done.status == "completed"
        assert done.result == "HUMAN FALLBACK"
        queue.close()


def test_retry_exhaustion_approve_accepts_best_effort(tmp_path: Path) -> None:
    with SqliteSaver.from_conn_string(str(tmp_path / "c.sqlite")) as saver:
        queue = ApprovalQueue(tmp_path / "q.sqlite")
        runner = Runner(
            provider=EchoProvider(_plan(), review_score=0.2),
            registry=_registry(),
            memory_store=NullMemoryStore(),
            checkpointer=saver,
            queue=queue,
        )
        pending = runner.submit(Task(description="x"))
        done = runner.resume(pending.approval_id, Decision(kind=DecisionKind.APPROVE))
        assert done.status == "completed"
        assert done.result  # best-effort synthesis of the agents' work proceeds
        queue.close()


def test_marginal_review_notifies_without_pausing(tmp_path: Path) -> None:
    # Score passes the bar (0.5) but sits below the comfort floor (0.75): NOTIFY is
    # non-blocking, so the run completes without ever pausing.
    with SqliteSaver.from_conn_string(str(tmp_path / "c.sqlite")) as saver:
        queue = ApprovalQueue(tmp_path / "q.sqlite")
        runner = Runner(
            provider=EchoProvider(_plan(), review_score=0.6),
            registry=_registry(),
            memory_store=NullMemoryStore(),
            checkpointer=saver,
            queue=queue,
        )
        res = runner.submit(Task(description="x"))
        assert res.status == "completed"
        assert queue.pending() == []
        queue.close()


def test_resume_unknown_approval_raises(tmp_path: Path) -> None:
    with SqliteSaver.from_conn_string(str(tmp_path / "c.sqlite")) as saver:
        queue = ApprovalQueue(tmp_path / "q.sqlite")
        runner = _runner(saver, queue, _plan())
        with pytest.raises(ValueError):
            runner.resume("nope", Decision(kind=DecisionKind.APPROVE))
        queue.close()

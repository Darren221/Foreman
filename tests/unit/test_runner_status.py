"""`Runner.status` must distinguish a finished run from one that is still executing.

Regression guard for the interaction with the claim-before-invoke fix: because an
approval is now resolved *before* the resume's graph work finishes, the queue no longer
flags the run as pending while it runs. `status` must then fall back to the checkpoint's
`next` to tell "done" from "still running", rather than reporting a resultless
"completed" (the empty-result symptom seen in the console)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from foreman.hitl.queue import ApprovalQueue
from foreman.hitl.runner import Runner
from tests.support import NullMemoryStore


class _Snapshot:
    def __init__(self, values: dict[str, Any], nxt: tuple[str, ...]) -> None:
        self.values = values
        self.next = nxt


class _FakeGraph:
    def __init__(self, snapshot: _Snapshot) -> None:
        self._snapshot = snapshot

    def get_state(self, config: Any) -> _Snapshot:
        return self._snapshot


def _runner_with_snapshot(tmp_path: Path, snapshot: _Snapshot) -> Runner:
    runner = Runner(
        provider=None,  # type: ignore[arg-type]  # status() touches neither provider nor registry
        registry=None,  # type: ignore[arg-type]
        memory_store=NullMemoryStore(),
        checkpointer=None,
        queue=ApprovalQueue(tmp_path / "q.sqlite"),
    )
    runner._graph = lambda: _FakeGraph(snapshot)  # type: ignore[method-assign]
    return runner


def test_unknown_run_is_none(tmp_path: Path) -> None:
    runner = _runner_with_snapshot(tmp_path, _Snapshot(values={}, nxt=()))
    assert runner.status("nope") is None


def test_in_flight_run_reports_running_not_completed(tmp_path: Path) -> None:
    # A checkpoint exists (the run was resumed) but the graph still has a node to run
    # and no result yet: it is NOT done. It must not be reported as completed.
    runner = _runner_with_snapshot(
        tmp_path, _Snapshot(values={"task": {"id": "t1"}}, nxt=("synthesize",))
    )
    result = runner.status("t1")
    assert result is not None
    assert result.status == "running"
    assert result.result is None


def test_finished_run_reports_completed_with_result(tmp_path: Path) -> None:
    runner = _runner_with_snapshot(
        tmp_path, _Snapshot(values={"result": "the answer"}, nxt=())
    )
    result = runner.status("t1")
    assert result is not None
    assert result.status == "completed"
    assert result.result == "the answer"

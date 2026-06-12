"""The graph must run end-to-end before any node does real work.

In Phase 1 every node is a stub; this test pins the contract that a task flows
intake -> plan -> execute -> review -> synthesize -> done and produces a result.
A mocked provider stands in for any future LLM use so the test is free and
deterministic.
"""

from __future__ import annotations

from foreman.graph import run_task
from foreman.llm.base import LLMProvider, T
from foreman.schemas import Task


class FakeProvider(LLMProvider):
    name = "fake"

    def structured_complete(self, prompt: str, schema: type[T]) -> T:  # pragma: no cover
        raise AssertionError("stub nodes must not call the LLM in Phase 1")


def test_graph_runs_end_to_end() -> None:
    task = Task(description="research the history of the bicycle")
    state = run_task(FakeProvider(), task)

    assert state["plan"] is not None
    assert state["outputs"], "a specialist should have produced output"
    assert state["review"] is not None and state["review"].passed
    assert isinstance(state["result"], str) and state["result"]

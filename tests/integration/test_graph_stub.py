"""The graph must run end-to-end as stubs are replaced by real agents.

As of T2 the `plan` node calls the supervisor (LLM); execute/review/synthesize
are still stubs. A canned provider supplies a deterministic `Plan` so the test
stays free and offline while proving the full pipeline still reaches a result.
"""

from __future__ import annotations

from foreman.graph import run_task
from foreman.llm.base import LLMProvider, T
from foreman.schemas import Plan, Specialist, Subtask, Task


class CannedProvider(LLMProvider):
    name = "canned"

    def __init__(self, response: object) -> None:
        self._response = response

    def structured_complete(self, prompt: str, schema: type[T]) -> T:
        return self._response  # type: ignore[return-value]


def _canned_plan() -> Plan:
    return Plan(
        task_id="t1",
        subtasks=[
            Subtask(
                id="s1",
                description="research the topic",
                assigned_specialist=Specialist.RESEARCHER,
                expected_output="findings",
                complexity=1,
            )
        ],
    )


def test_graph_runs_end_to_end() -> None:
    provider = CannedProvider(_canned_plan())
    state = run_task(provider, Task(description="research the history of the bicycle"))

    assert state["plan"] is not None
    assert state["plan"].subtasks[0].assigned_specialist is Specialist.RESEARCHER
    assert state["outputs"], "a specialist should have produced output"
    assert state["review"] is not None and state["review"].passed
    assert isinstance(state["result"], str) and state["result"]

"""C5: the FastAPI surface over the Runner. A submitted task that triggers
escalation lands a pending approval; resolving it via the API resumes the run to
completion. The user-data delete endpoint purges a memory so recall no longer
returns it. Driven offline via TestClient with fakes (Celery runs eager)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
from langgraph.checkpoint.sqlite import SqliteSaver

from foreman.api.app import create_app
from foreman.hitl import ApprovalQueue, Runner
from foreman.llm.base import LLMProvider, T
from foreman.schemas import (
    Plan,
    ResearchFindings,
    ReviewResult,
    Specialist,
    Subtask,
    Synthesis,
    TaskMemory,
)
from foreman.tools import ToolRegistry, WebSearchTool
from tests.support import DictMemoryStore, NullMemoryStore


class EchoProvider(LLMProvider):
    name = "echo"

    def __init__(self, plan: Plan) -> None:
        self._plan = plan

    def structured_complete(self, prompt: str, schema: type[T]) -> T:
        if schema is Plan:
            return self._plan  # type: ignore[return-value]
        if schema is ResearchFindings:
            return ResearchFindings(content="findings")  # type: ignore[return-value]
        if schema is ReviewResult:
            return ReviewResult(passed=True, score=0.9, feedback="ok")  # type: ignore[return-value]
        if schema is Synthesis:
            return Synthesis(result="the answer")  # type: ignore[return-value]
        raise AssertionError(f"unexpected schema {schema!r}")


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


def test_submit_then_resolve_completes_the_run(tmp_path: Path) -> None:
    with SqliteSaver.from_conn_string(str(tmp_path / "c.sqlite")) as saver:
        queue = ApprovalQueue(tmp_path / "q.sqlite")
        runner = Runner(
            provider=EchoProvider(_plan()),
            registry=_registry(),
            memory_store=NullMemoryStore(),
            checkpointer=saver,
            queue=queue,
        )
        client = TestClient(create_app(runner, queue, DictMemoryStore()))

        submitted = client.post(
            "/tasks", json={"description": "research the bicycle", "sensitive": True}
        ).json()
        run_id = submitted["id"]
        assert submitted["status"] == "pending"
        assert submitted["approval_id"]
        assert client.get(f"/tasks/{run_id}").json()["status"] == "pending"

        pending = client.get("/approvals").json()
        assert len(pending) == 1
        approval_id = pending[0]["id"]

        resolved = client.post(f"/approvals/{approval_id}", json={"kind": "approve"}).json()
        assert resolved["status"] == "completed"

        done = client.get(f"/tasks/{run_id}").json()
        assert done["status"] == "completed"
        assert done["result"]
        queue.close()


def test_unknown_task_is_404(tmp_path: Path) -> None:
    with SqliteSaver.from_conn_string(str(tmp_path / "c.sqlite")) as saver:
        queue = ApprovalQueue(tmp_path / "q.sqlite")
        runner = Runner(
            provider=EchoProvider(_plan()),
            registry=_registry(),
            memory_store=NullMemoryStore(),
            checkpointer=saver,
            queue=queue,
        )
        client = TestClient(create_app(runner, queue, DictMemoryStore()))
        assert client.get("/tasks/nope").status_code == 404
        queue.close()


def test_delete_memory_endpoint_purges_from_recall(tmp_path: Path) -> None:
    memory = DictMemoryStore()
    record = TaskMemory(task_description="history of the bicycle", outcome="passed", score=0.9)
    memory.remember(record)
    with SqliteSaver.from_conn_string(str(tmp_path / "c.sqlite")) as saver:
        queue = ApprovalQueue(tmp_path / "q.sqlite")
        runner = Runner(
            provider=EchoProvider(_plan()),
            registry=_registry(),
            memory_store=memory,
            checkpointer=saver,
            queue=queue,
        )
        client = TestClient(create_app(runner, queue, memory))

        assert memory.recall("history of the bicycle")
        assert client.delete(f"/memory/{record.id}").status_code == 204
        assert memory.recall("history of the bicycle") == []
        queue.close()

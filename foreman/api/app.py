"""The orchestration API: an HTTP surface over the `Runner`.

Thin by design — the same split as the Streamlit UIs. Every endpoint translates a
request into a `Runner`/queue/memory call and serializes the result; no orchestration
logic lives here. `create_app` takes its dependencies so tests drive it with fakes
over a `TestClient`; `build_app` wires the real ones from config for `uvicorn`.

Run it:  uvicorn --factory foreman.api.app:build_app

Run tracking is in-process: `Runner.submit` runs synchronously to completion-or-pause,
so the outcome is known when the request returns; we keep the last `RunResult` per run
id for `GET /tasks/{id}`. Durable status (reading the checkpointer) is a follow-up.
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from foreman.hitl.policy import Escalation
from foreman.hitl.queue import ApprovalQueue, Decision, DecisionKind, PendingApproval
from foreman.hitl.runner import Runner, RunResult
from foreman.memory import MemoryStore
from foreman.schemas import Plan, Task


class TaskRequest(BaseModel):
    description: str
    require_approval: bool = False
    sensitive: bool = False


class TaskResponse(BaseModel):
    id: str
    status: str
    result: str | None = None
    approval_id: str | None = None
    escalation: Escalation | None = None


class DecisionRequest(BaseModel):
    kind: DecisionKind
    feedback: str = ""
    plan: Plan | None = None
    output: str = ""


def _response(run_id: str, result: RunResult) -> TaskResponse:
    return TaskResponse(
        id=run_id,
        status=result.status,
        result=result.result,
        approval_id=result.approval_id,
        escalation=result.escalation,
    )


def create_app(runner: Runner, queue: ApprovalQueue, memory_store: MemoryStore) -> FastAPI:
    app = FastAPI(title="Foreman")
    # run id -> its latest outcome (see the module docstring on run tracking).
    runs: dict[str, RunResult] = {}

    @app.post("/tasks")
    def submit_task(request: TaskRequest) -> TaskResponse:
        task = Task(
            description=request.description,
            require_approval=request.require_approval,
            sensitive=request.sensitive,
        )
        result = runner.submit(task)
        runs[task.id] = result
        return _response(task.id, result)

    @app.get("/tasks/{run_id}")
    def get_task(run_id: str) -> TaskResponse:
        result = runs.get(run_id)
        if result is None:
            raise HTTPException(status_code=404, detail="unknown task")
        return _response(run_id, result)

    @app.get("/approvals")
    def list_approvals() -> list[PendingApproval]:
        return queue.pending()

    @app.post("/approvals/{approval_id}")
    def resolve_approval(approval_id: str, request: DecisionRequest) -> TaskResponse:
        pending = queue.get(approval_id)
        if pending is None or pending.resolved:
            raise HTTPException(status_code=404, detail="unknown or resolved approval")
        decision = Decision(
            kind=request.kind,
            feedback=request.feedback,
            plan=request.plan,
            output=request.output,
        )
        result = runner.resume(approval_id, decision)
        runs[pending.thread_id] = result
        return _response(pending.thread_id, result)

    @app.delete("/memory/{memory_id}", status_code=204)
    def delete_memory(memory_id: str) -> None:
        # The user-data delete path (SPEC §7). Memories aren't user-keyed yet, so the
        # scope is purge-by-id; per-user scoping is a follow-up.
        memory_store.delete([memory_id])

    return app


def build_app() -> FastAPI:
    """Wire the real dependencies from config (the `uvicorn --factory` entrypoint)."""
    from foreman.config import Settings
    from foreman.llm import select_provider
    from foreman.memory import build_default_memory_store
    from foreman.storage.factory import build_approval_queue, build_checkpointer
    from foreman.tools import build_default_registry

    settings = Settings()
    memory_store = build_default_memory_store(settings)
    queue = build_approval_queue(settings)
    runner = Runner(
        provider=select_provider(settings),
        registry=build_default_registry(settings),
        memory_store=memory_store,
        checkpointer=build_checkpointer(settings),
        queue=queue,
    )
    return create_app(runner, queue, memory_store)

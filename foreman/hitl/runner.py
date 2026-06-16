"""The runner: drive a task through the graph, pausing for human approval.

`submit` runs until the graph either finishes or interrupts for approval; on an
interrupt it persists the escalation to the queue and returns control. `resume`
applies the human's decision and continues the same run from its checkpoint. The
runner is the seam between the autonomous graph and the human: it owns neither
the decision logic (the policy does) nor the storage (the queue and checkpointer
do), it just carries state across the pause.
"""

from __future__ import annotations

from typing import Any, Literal

from langgraph.types import Command
from pydantic import BaseModel

from foreman.hitl.policy import Escalation, EscalationPolicy
from foreman.hitl.queue import ApprovalQueue, Decision
from foreman.llm.base import LLMProvider
from foreman.memory import MemoryStore
from foreman.schemas import Task
from foreman.tools import ToolRegistry


class RunResult(BaseModel):
    """The outcome of a submit/resume: either a finished result, or a pending
    approval the human must resolve before the run can continue."""

    status: Literal["completed", "pending"]
    result: str | None = None
    approval_id: str | None = None
    escalation: Escalation | None = None


class Runner:
    def __init__(
        self,
        *,
        provider: LLMProvider,
        registry: ToolRegistry,
        memory_store: MemoryStore,
        checkpointer: Any,
        queue: ApprovalQueue,
        policy: EscalationPolicy | None = None,
    ) -> None:
        self._provider = provider
        self._registry = registry
        self._memory_store = memory_store
        self._checkpointer = checkpointer
        self._queue = queue
        self._policy = policy or EscalationPolicy()

    def submit(self, task: Task) -> RunResult:
        config = {"configurable": {"thread_id": task.id}}
        state = self._graph().invoke({"task": task}, config)
        return self._after(state, task.id)

    def resume(self, approval_id: str, decision: Decision) -> RunResult:
        pending = self._queue.get(approval_id)
        if pending is None or pending.resolved:
            raise ValueError(f"no pending approval with id {approval_id!r}")
        config = {"configurable": {"thread_id": pending.thread_id}}
        state = self._graph().invoke(Command(resume=decision.model_dump()), config)
        self._queue.resolve(approval_id, decision)
        return self._after(state, pending.thread_id)

    def status(self, run_id: str) -> RunResult | None:
        """The durable status of a run, recovered from the approval queue and the
        checkpointer rather than from process memory, so it survives a restart. A
        pending approval for the run means it's paused; otherwise the checkpointed
        state carries the result. None if the run is unknown (no pending approval and
        no checkpoint)."""
        for pending in self._queue.pending():
            if pending.thread_id == run_id:
                return RunResult(
                    status="pending", approval_id=pending.id, escalation=pending.escalation
                )
        config = {"configurable": {"thread_id": run_id}}
        snapshot = self._graph().get_state(config)
        if not snapshot.values:
            return None
        return RunResult(status="completed", result=snapshot.values.get("result"))

    def _graph(self) -> Any:
        # Imported lazily: the graph builder imports the hitl policy/queue, so a
        # module-level import here would close a cycle (builder -> hitl -> runner).
        from foreman.graph.builder import build_graph

        return build_graph(
            self._provider,
            self._registry,
            self._memory_store,
            checkpointer=self._checkpointer,
            policy=self._policy,
        )

    def _after(self, state: Any, thread_id: str) -> RunResult:
        interrupts = state.get("__interrupt__")
        if interrupts:
            escalation = Escalation.model_validate(interrupts[0].value)
            approval_id = self._queue.enqueue(escalation, thread_id)
            return RunResult(status="pending", approval_id=approval_id, escalation=escalation)
        return RunResult(status="completed", result=state.get("result"))

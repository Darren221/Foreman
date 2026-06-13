"""Assembles the agent pipeline as a LangGraph state machine.

The pipeline runs retrieve -> plan -> approval -> execute -> review ->
post_review -> synthesize -> remember. `retrieve` recalls similar past tasks to
inform planning; `remember` stores a memory of this run. Two human-in-the-loop
gates bracket the work: `approval` (before execution) and `post_review` (after
review); each pauses only when the escalation policy says so. The review node
feeds a conditional edge: a passing verdict moves on to the post-review gate, a
failing one routes back to execute (with the reviewer's feedback) until a retry
cap, after which the pipeline proceeds rather than looping forever. A specialist
failure is captured as output rather than crashing.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from foreman.agents import Researcher, Reviewer, Supervisor
from foreman.graph.state import GraphState
from foreman.hitl.policy import ApprovalLevel, EscalationPolicy, Stage
from foreman.hitl.queue import Decision, DecisionKind
from foreman.llm.base import LLMProvider
from foreman.memory import MemoryStore
from foreman.observability import NoOpTracer, Tracer
from foreman.schemas import ReviewResult, SpecialistOutput, Task, TaskMemory
from foreman.tools import ToolRegistry

MAX_ATTEMPTS = 2
_RESULT_SNIPPET_LEN = 500


def _apply_decision(decision: Decision, state: GraphState) -> GraphState:
    """Translate a human's approval decision into a state update.

    Take-over delivers the human's own output and marks the run resolved, so the
    agents stand down. Modify swaps in the human's plan. Reject re-runs the work
    carrying the feedback (the same channel the reviewer uses on a retry), so the
    next pass can address it. Approve is a no-op — the run just proceeds.
    """
    if decision.kind is DecisionKind.TAKE_OVER:
        return {
            "result": decision.output,
            "review": ReviewResult(passed=True, score=1.0, feedback="human take-over"),
        }
    if decision.kind is DecisionKind.MODIFY and decision.plan is not None:
        return {"plan": decision.plan}
    if decision.kind is DecisionKind.REJECT:
        return {"review": ReviewResult(passed=False, score=0.0, feedback=decision.feedback)}
    return {}


def build_graph(
    provider: LLMProvider,
    registry: ToolRegistry,
    memory_store: MemoryStore,
    checkpointer: Any = None,
    policy: EscalationPolicy | None = None,
    tracer: Tracer | None = None,
) -> Any:
    """Wire the nodes into a compiled graph. `provider`, `registry`, and
    `memory_store` are the dependencies the nodes need. A `checkpointer` enables
    durable pause/resume (required for the human-in-the-loop interrupt); `policy`
    decides when that interrupt fires (defaults to the standard thresholds); a
    `tracer` records a span per node (no-op by default)."""

    supervisor = Supervisor(provider)
    researcher = Researcher(registry, provider)
    reviewer = Reviewer(provider)
    policy = policy or EscalationPolicy()
    tracer = tracer or NoOpTracer()
    # The gates can only pause if there's somewhere to pause *to*. With no
    # checkpointer there's no human channel, so the gates are inert and the
    # pipeline runs autonomously (degrading to best-effort at the retry cap).
    human_in_loop = checkpointer is not None

    def retrieve_node(state: GraphState) -> GraphState:
        memories = memory_store.recall(state["task"].description)
        return {"retrieved_memories": memories}

    def plan_node(state: GraphState) -> GraphState:
        memories = state.get("retrieved_memories")
        return {"plan": supervisor.plan(state["task"], memories)}

    def approval_node(state: GraphState) -> GraphState:
        # Human-in-the-loop gate, sited after planning and before execution. The
        # policy decides whether to pause; a clean run produces no escalation and
        # passes straight through, so the default pipeline is unchanged. When it
        # does pause, the resume value is the human's Decision, applied below.
        if not human_in_loop:
            return {}
        escalation = policy.evaluate(state, stage=Stage.PRE_EXECUTION)
        if escalation is None or escalation.level is ApprovalLevel.NOTIFY:
            return {}
        decision = Decision.model_validate(interrupt(escalation.model_dump()))
        return _apply_decision(decision, state)

    def route_after_approval(state: GraphState) -> str:
        # A take-over decision sets the result directly; skip the agents and go
        # straight to recording the run. Every other decision runs the plan.
        return "remember" if state.get("result") is not None else "execute"

    def execute_node(state: GraphState) -> GraphState:
        plan = state["plan"]
        assert plan is not None
        review = state.get("review")
        feedback = review.feedback if review and not review.passed else None
        outputs = []
        for subtask in plan.subtasks:
            try:
                outputs.append(researcher.execute(subtask, feedback=feedback))
            except Exception as exc:
                # Degrade gracefully: capture the failure as output so the
                # reviewer can reject it and the pipeline keeps moving.
                outputs.append(
                    SpecialistOutput(
                        subtask_id=subtask.id,
                        content=f"[execution failed: {exc}]",
                    )
                )
        return {"outputs": outputs}

    def review_node(state: GraphState) -> GraphState:
        plan = state["plan"]
        assert plan is not None
        subtasks = {s.id: s for s in plan.subtasks}
        attempts = state.get("attempts", 0) + 1
        verdict = ReviewResult(passed=True, score=1.0, feedback="")
        for output in state.get("outputs") or []:
            verdict = reviewer.review(subtasks[output.subtask_id], output)
            if not verdict.passed:
                break
        return {"review": verdict, "attempts": attempts}

    def route_after_review(state: GraphState) -> str:
        review = state["review"]
        assert review is not None
        if review.passed or state.get("attempts", 0) >= MAX_ATTEMPTS:
            return "post_review"
        return "execute"

    def post_review_node(state: GraphState) -> GraphState:
        # Second human-in-the-loop gate, after the work is reviewed. It catches
        # the outcome-based triggers: a TAKE_OVER when retries are exhausted (the
        # human supplies the result), or a non-blocking NOTIFY on a marginal pass.
        if not human_in_loop:
            return {}
        escalation = policy.evaluate(state, stage=Stage.POST_REVIEW)
        if escalation is None or escalation.level is ApprovalLevel.NOTIFY:
            return {}
        decision = Decision.model_validate(interrupt(escalation.model_dump()))
        return _apply_decision(decision, state)

    def route_after_post_review(state: GraphState) -> str:
        # A take-over supplies the result outright; otherwise synthesise it.
        return "remember" if state.get("result") is not None else "synthesize"

    def synthesize_node(state: GraphState) -> GraphState:
        outputs = state.get("outputs") or []
        return {"result": supervisor.synthesize(state["task"], outputs)}

    def remember_node(state: GraphState) -> GraphState:
        # Distil the finished task into a memory for future recall.
        review = state.get("review")
        outputs = state.get("outputs") or []
        tools_used = sorted({tool for o in outputs for tool in o.tools_used})
        memory = TaskMemory(
            task_description=state["task"].description,
            outcome="passed" if review and review.passed else "failed",
            score=review.score if review else 0.0,
            tools_used=tools_used,
            result_snippet=(state.get("result") or "")[:_RESULT_SNIPPET_LEN],
        )
        memory_store.remember(memory)
        return {}

    def traced(name: str, fn: Callable[[GraphState], GraphState]) -> Any:
        # Wrap a node so each execution is a `node:<name>` span, nested under the
        # run span via the tracer's context. No-op tracer -> no overhead. Returns
        # Any: LangGraph's add_node overloads accept a concrete callable here.
        def run(state: GraphState) -> GraphState:
            with tracer.span(f"node:{name}", kind="node"):
                return fn(state)

        return run

    graph = StateGraph(GraphState)
    graph.add_node("retrieve", traced("retrieve", retrieve_node))
    graph.add_node("plan", traced("plan", plan_node))
    graph.add_node("approval", traced("approval", approval_node))
    graph.add_node("execute", traced("execute", execute_node))
    graph.add_node("review", traced("review", review_node))
    graph.add_node("post_review", traced("post_review", post_review_node))
    graph.add_node("synthesize", traced("synthesize", synthesize_node))
    graph.add_node("remember", traced("remember", remember_node))

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "plan")
    graph.add_edge("plan", "approval")
    graph.add_conditional_edges("approval", route_after_approval, ["execute", "remember"])
    graph.add_edge("execute", "review")
    graph.add_conditional_edges("review", route_after_review, ["execute", "post_review"])
    graph.add_conditional_edges("post_review", route_after_post_review, ["synthesize", "remember"])
    graph.add_edge("synthesize", "remember")
    graph.add_edge("remember", END)

    return graph.compile(checkpointer=checkpointer)


def run_task(
    provider: LLMProvider,
    task: Task,
    registry: ToolRegistry | None = None,
    memory_store: MemoryStore | None = None,
    tracer: Tracer | None = None,
) -> GraphState:
    """Run a task through the compiled graph and return the final state.

    Dependencies can be injected (tests pass fakes); otherwise the defaults are
    built from settings. A `tracer` opens the root run span the node spans nest
    under (no-op by default).
    """
    if registry is None or memory_store is None:
        from foreman.config import Settings
        from foreman.memory import build_default_memory_store
        from foreman.tools import build_default_registry

        settings = Settings()
        registry = registry or build_default_registry(settings)
        memory_store = memory_store or build_default_memory_store(settings)
    tracer = tracer or NoOpTracer()
    graph = build_graph(provider, registry, memory_store, tracer=tracer)
    with tracer.start_run(task.id, task.description):
        return cast(GraphState, graph.invoke({"task": task}))

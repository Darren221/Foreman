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

from celery import group
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from opentelemetry import trace
from opentelemetry.propagate import inject

from foreman.agents import Reviewer, Supervisor
from foreman.graph.state import GraphState
from foreman.hitl.policy import ApprovalLevel, Escalation, EscalationPolicy, Stage
from foreman.hitl.queue import Decision, DecisionKind
from foreman.llm.base import LLMProvider
from foreman.memory import MemoryStore
from foreman.observability import NoOpTracer, Tracer, TracingProvider
from foreman.schemas import ReviewResult, SpecialistOutput, Subtask, Task, TaskMemory
from foreman.tools import ToolRegistry
from foreman.workers import celery_app as _celery_app  # noqa: F401  (build + default the app)
from foreman.workers.tasks import run_specialist, use_worker_context

MAX_ATTEMPTS = 2
_RESULT_SNIPPET_LEN = 500


def _record_escalation_event(escalation: Escalation) -> None:
    # Attach the escalation to the active node span as an event. Reads the current
    # span from OTel context, so it nests correctly and no-ops when untraced.
    trace.get_current_span().add_event(
        "escalation",
        {"foreman.trigger": escalation.trigger.value, "foreman.level": escalation.level.value},
    )


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
        # New plan: drop the per-subtask reviews so every new subtask re-runs.
        return {"plan": decision.plan, "subtask_reviews": {}}
    if decision.kind is DecisionKind.REJECT:
        # Human reject: re-run the whole plan with the human's feedback (the
        # cleared per-subtask reviews send execute down the aggregate-feedback path).
        return {
            "review": ReviewResult(passed=False, score=0.0, feedback=decision.feedback),
            "subtask_reviews": {},
        }
    return {}


def _output_from_result(subtask: Subtask, result: Any) -> SpecialistOutput:
    """Map a worker result into a `SpecialistOutput`. A gather with `propagate=False`
    yields the output JSON for a success and the exception for a failure; we degrade a
    failure into a failure-output so the reviewer can reject it and the run moves on."""
    if isinstance(result, str):
        return SpecialistOutput.model_validate_json(result)
    return SpecialistOutput(
        subtask_id=subtask.id,
        content=f"[execution failed: {result}]",
        produced_by=subtask.assigned_specialist,
    )


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

    policy = policy or EscalationPolicy()
    tracer = tracer or NoOpTracer()
    # Instrument the dependencies: llm spans come from the wrapped provider, tool
    # spans from the registry. No-op tracer -> no spans, no behaviour change.
    traced_provider = TracingProvider(provider, tracer)
    registry.tracer = tracer
    supervisor = Supervisor(traced_provider)
    reviewer = Reviewer(traced_provider)
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
        _record_escalation_event(escalation)
        decision = Decision.model_validate(interrupt(escalation.model_dump()))
        return _apply_decision(decision, state)

    def route_after_approval(state: GraphState) -> str:
        # A take-over decision sets the result directly; skip the agents and go
        # straight to recording the run. Every other decision runs the plan.
        return "remember" if state.get("result") is not None else "execute"

    def execute_node(state: GraphState) -> GraphState:
        plan = state["plan"]
        assert plan is not None
        # Per-output retry: keep a subtask's output if its last review passed; only
        # re-run the ones that failed (or that never ran). Each re-run gets its own
        # subtask's feedback; a human reject (no per-subtask reviews) falls back to
        # the aggregate review feedback, re-running everything.
        reviews = state.get("subtask_reviews") or {}
        review = state.get("review")
        aggregate_feedback = review.feedback if review and not review.passed else None
        outputs = {o.subtask_id: o for o in (state.get("outputs") or [])}

        def feedback_for(subtask: Subtask) -> str | None:
            verdict = reviews.get(subtask.id)
            return verdict.feedback if verdict is not None else aggregate_feedback

        # C-orchestrated fan-out: dispatch in dependency-ordered waves. `done` holds
        # the subtasks whose outputs are available (kept-passing ones to start with);
        # each wave is the subtasks still to run whose dependencies are all done. A
        # wave fans out to the workers in parallel; we join it back into `outputs`,
        # then its dependents become runnable. Workers stay stateless — `GraphState`
        # (these outputs) bridges the waves, which is why no Redis store is needed.
        remaining = [s for s in plan.subtasks if not (reviews.get(s.id) and reviews[s.id].passed)]
        done = set(outputs)

        # Only propagate a trace context when actually recording, so an untraced run
        # doesn't spuriously start tracing in a remote worker.
        traceparent: str | None = None
        if trace.get_current_span().is_recording():
            carrier: dict[str, str] = {}
            inject(carrier)
            traceparent = carrier.get("traceparent")

        with use_worker_context(provider, registry, tracer):
            while remaining:
                wave = [s for s in remaining if all(d in done for d in s.dependencies)]
                if not wave:
                    break  # unsatisfiable deps — impossible for a validated DAG
                signatures = [
                    run_specialist.s(
                        subtask.model_dump_json(),
                        feedback_for(subtask),
                        [outputs[dep].model_dump_json() for dep in subtask.dependencies],
                        traceparent,
                    )
                    for subtask in wave
                ]
                results = group(signatures).apply_async().get(propagate=False)
                for subtask, result in zip(wave, results, strict=True):
                    outputs[subtask.id] = _output_from_result(subtask, result)
                    done.add(subtask.id)
                ran = {s.id for s in wave}
                remaining = [s for s in remaining if s.id not in ran]

        return {"outputs": [outputs[s.id] for s in plan.subtasks]}

    def review_node(state: GraphState) -> GraphState:
        plan = state["plan"]
        assert plan is not None
        subtasks = {s.id: s for s in plan.subtasks}
        attempts = state.get("attempts", 0) + 1
        # Judge every output, not just up to the first weak one, so the next
        # execute can retry only the subtasks that actually failed.
        reviews = {
            output.subtask_id: reviewer.review(subtasks[output.subtask_id], output)
            for output in state.get("outputs") or []
        }
        verdicts = list(reviews.values())
        aggregate = ReviewResult(
            passed=all(v.passed for v in verdicts),
            score=min((v.score for v in verdicts), default=1.0),
            feedback=" | ".join(v.feedback for v in verdicts if not v.passed),
        )
        return {"review": aggregate, "subtask_reviews": reviews, "attempts": attempts}

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
        _record_escalation_event(escalation)
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

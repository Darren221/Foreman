"""Assembles the agent pipeline as a LangGraph state machine.

The pipeline runs intake -> plan -> execute -> review -> synthesize. The review
node feeds a conditional edge: a passing verdict goes to synthesis, a failing one
routes back to execute (with the reviewer's feedback) until a retry cap, after
which the pipeline proceeds rather than looping forever. A specialist failure is
captured as output rather than crashing the run.
"""

from __future__ import annotations

from typing import Any, cast

from langgraph.graph import END, START, StateGraph

from foreman.agents import Researcher, Reviewer, Supervisor
from foreman.graph.state import GraphState
from foreman.llm.base import LLMProvider
from foreman.memory import MemoryStore
from foreman.schemas import ReviewResult, SpecialistOutput, Task, TaskMemory
from foreman.tools import ToolRegistry

MAX_ATTEMPTS = 2
_RESULT_SNIPPET_LEN = 500


def build_graph(
    provider: LLMProvider, registry: ToolRegistry, memory_store: MemoryStore
) -> Any:
    """Wire the nodes into a compiled graph. `provider`, `registry`, and
    `memory_store` are the dependencies the nodes need."""

    supervisor = Supervisor(provider)
    researcher = Researcher(registry, provider)
    reviewer = Reviewer(provider)

    def plan_node(state: GraphState) -> GraphState:
        return {"plan": supervisor.plan(state["task"])}

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
            return "synthesize"
        return "execute"

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

    graph = StateGraph(GraphState)
    graph.add_node("plan", plan_node)
    graph.add_node("execute", execute_node)
    graph.add_node("review", review_node)
    graph.add_node("synthesize", synthesize_node)
    graph.add_node("remember", remember_node)

    graph.add_edge(START, "plan")
    graph.add_edge("plan", "execute")
    graph.add_edge("execute", "review")
    graph.add_conditional_edges("review", route_after_review, ["execute", "synthesize"])
    graph.add_edge("synthesize", "remember")
    graph.add_edge("remember", END)

    return graph.compile()


def run_task(
    provider: LLMProvider,
    task: Task,
    registry: ToolRegistry | None = None,
    memory_store: MemoryStore | None = None,
) -> GraphState:
    """Run a task through the compiled graph and return the final state.

    Dependencies can be injected (tests pass fakes); otherwise the defaults are
    built from settings.
    """
    if registry is None or memory_store is None:
        from foreman.config import Settings
        from foreman.memory import build_default_memory_store
        from foreman.tools import build_default_registry

        settings = Settings()
        registry = registry or build_default_registry(settings)
        memory_store = memory_store or build_default_memory_store(settings)
    graph = build_graph(provider, registry, memory_store)
    return cast(GraphState, graph.invoke({"task": task}))

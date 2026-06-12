"""Assembles the agent pipeline as a LangGraph state machine.

Phase 1: every node is a stub that produces placeholder values, so the wiring
(state passing, edge order) can be proven end-to-end before any node does real
work. Tasks T2-T5 replace these stubs one at a time; the graph shape stays put.
"""

from __future__ import annotations

from typing import Any, cast

from langgraph.graph import END, START, StateGraph

from foreman.agents import Researcher, Supervisor
from foreman.graph.state import GraphState
from foreman.llm.base import LLMProvider
from foreman.schemas import ReviewResult, Task
from foreman.tools import ToolRegistry


def build_graph(provider: LLMProvider, registry: ToolRegistry) -> Any:
    """Wire the nodes into a compiled graph. `provider` and `registry` are the
    dependencies the real agents need; stubs that remain ignore them."""

    supervisor = Supervisor(provider)
    researcher = Researcher(registry)

    def plan_node(state: GraphState) -> GraphState:
        return {"plan": supervisor.plan(state["task"])}

    def execute_node(state: GraphState) -> GraphState:
        plan = state["plan"]
        assert plan is not None
        outputs = [researcher.execute(s) for s in plan.subtasks]
        return {"outputs": outputs}

    def review_node(state: GraphState) -> GraphState:
        return {"review": ReviewResult(passed=True, score=1.0, feedback="stub: accepted")}

    def synthesize_node(state: GraphState) -> GraphState:
        outputs = state.get("outputs") or []
        joined = "\n".join(o.content for o in outputs)
        return {"result": f"[stub synthesis]\n{joined}"}

    graph = StateGraph(GraphState)
    graph.add_node("plan", plan_node)
    graph.add_node("execute", execute_node)
    graph.add_node("review", review_node)
    graph.add_node("synthesize", synthesize_node)

    graph.add_edge(START, "plan")
    graph.add_edge("plan", "execute")
    graph.add_edge("execute", "review")
    graph.add_edge("review", "synthesize")
    graph.add_edge("synthesize", END)

    return graph.compile()


def run_task(
    provider: LLMProvider, task: Task, registry: ToolRegistry | None = None
) -> GraphState:
    """Run a task through the compiled graph and return the final state.

    A registry can be injected (tests pass one with a fake search backend);
    otherwise the default registry is built from settings.
    """
    if registry is None:
        from foreman.config import Settings
        from foreman.tools import build_default_registry

        registry = build_default_registry(Settings())
    graph = build_graph(provider, registry)
    return cast(GraphState, graph.invoke({"task": task}))

"""Assembles the agent pipeline as a LangGraph state machine.

Phase 1: every node is a stub that produces placeholder values, so the wiring
(state passing, edge order) can be proven end-to-end before any node does real
work. Tasks T2-T5 replace these stubs one at a time; the graph shape stays put.
"""

from __future__ import annotations

from typing import Any, cast

from langgraph.graph import END, START, StateGraph

from foreman.graph.state import GraphState
from foreman.llm.base import LLMProvider
from foreman.schemas import (
    Plan,
    ReviewResult,
    Specialist,
    SpecialistOutput,
    Subtask,
    Task,
)


def build_graph(provider: LLMProvider) -> Any:
    """Wire the nodes into a compiled graph. `provider` is captured for the
    real agents that replace these stubs in later tasks."""

    def plan_node(state: GraphState) -> GraphState:
        task = state["task"]
        plan = Plan(
            task_id=task.id,
            subtasks=[
                Subtask(
                    id="s1",
                    description=f"research: {task.description}",
                    assigned_specialist=Specialist.RESEARCHER,
                    expected_output="findings",
                    complexity=1,
                )
            ],
        )
        return {"plan": plan}

    def execute_node(state: GraphState) -> GraphState:
        plan = state["plan"]
        assert plan is not None
        outputs = [
            SpecialistOutput(subtask_id=s.id, content=f"[stub output for {s.id}]")
            for s in plan.subtasks
        ]
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


def run_task(provider: LLMProvider, task: Task) -> GraphState:
    """Run a task through the compiled graph and return the final state."""
    graph = build_graph(provider)
    return cast(GraphState, graph.invoke({"task": task}))

"""O5: counterfactual forking — re-run a finished run from a chosen node with an
edited input, under a new trace, and compare the divergence. Reuses the same
checkpointer primitive as HITL. Non-deterministic by design; here a deterministic
echo provider makes the edit's effect visible in the result."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver

from foreman.graph.builder import build_graph
from foreman.llm.base import LLMProvider, T
from foreman.observability import OTelTracer, SpanNode, TraceStore
from foreman.observability.replay import compare_runs, fork_run
from foreman.schemas import (
    Plan,
    ResearchFindings,
    ReviewResult,
    Specialist,
    Subtask,
    Synthesis,
    Task,
)
from foreman.tools import ToolRegistry, WebSearchTool
from tests.support import NullMemoryStore


def _between(text: str, start: str, end: str) -> str:
    i = text.index(start) + len(start)
    return text[i : text.index(end, i)].strip()


class EchoProvider(LLMProvider):
    """Findings echo the subtask topic; synthesis echoes the findings — so a
    different plan yields a different result."""

    name = "echo"
    model = "gpt-4o"

    def __init__(self, plan: Plan) -> None:
        self._plan = plan

    def structured_complete(self, prompt: str, schema: type[T]) -> T:
        if schema is Plan:
            return self._plan  # type: ignore[return-value]
        if schema is ResearchFindings:
            return ResearchFindings(content=_between(prompt, "Subtask: ", "\n"))  # type: ignore[return-value]
        if schema is ReviewResult:
            return ReviewResult(passed=True, score=0.9, feedback="ok")  # type: ignore[return-value]
        if schema is Synthesis:
            return Synthesis(result=prompt.split("Findings:\n", 1)[-1].strip())  # type: ignore[return-value]
        raise AssertionError(schema)


class FakeBackend:
    def search(self, query: str, max_results: int) -> list[dict[str, Any]]:
        return [{"title": "t", "url": "u", "content": "c"}]


def _registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(WebSearchTool(FakeBackend()))
    return reg


def _plan(topic: str) -> Plan:
    return Plan(
        task_id="t1",
        subtasks=[
            Subtask(
                id="s1",
                description=topic,
                assigned_specialist=Specialist.RESEARCHER,
                expected_output="findings",
                complexity=1,
            )
        ],
    )


def _node_names(root: SpanNode | None) -> set[str]:
    if root is None:
        return set()
    names = {root.span.name} if root.span.kind == "node" else set()
    for child in root.children:
        names |= _node_names(child)
    return names


def test_fork_resumes_from_node_and_diverges(tmp_path: Path) -> None:
    store = TraceStore(tmp_path / "traces.sqlite")
    tracer = OTelTracer(store)
    with SqliteSaver.from_conn_string(str(tmp_path / "c.sqlite")) as saver:
        graph = build_graph(
            EchoProvider(_plan("original-topic")),
            _registry(),
            NullMemoryStore(),
            checkpointer=saver,
            tracer=tracer,
        )
        task = Task(description="research")
        config = {"configurable": {"thread_id": task.id}}
        with tracer.start_run(task.id, task.description) as span:
            original = graph.invoke({"task": task}, config)
            original_trace_id = f"{span.get_span_context().trace_id:032x}"

        forked = fork_run(
            graph,
            thread_id=task.id,
            at_node="execute",
            edits={"plan": _plan("MODIFIED-topic")},
            tracer=tracer,
        )

    # The fork re-ran from execute onward; the prefix was inherited, not re-run.
    fork_nodes = _node_names(store.get_trace(forked.fork_trace_id))
    assert "node:execute" in fork_nodes
    assert "node:plan" not in fork_nodes

    # The edit diverged the outcome.
    assert "MODIFIED-topic" in (forked.result or "")
    assert "MODIFIED-topic" not in (original.get("result") or "")

    comparison = compare_runs(store, original_trace_id, forked.fork_trace_id, "execute")
    assert "node:plan" in comparison.inherited
    assert "node:execute" in comparison.replayed
    store.close()


def test_fork_at_unknown_node_raises(tmp_path: Path) -> None:
    store = TraceStore(tmp_path / "traces.sqlite")
    tracer = OTelTracer(store)
    with SqliteSaver.from_conn_string(str(tmp_path / "c.sqlite")) as saver:
        graph = build_graph(
            EchoProvider(_plan("x")),
            _registry(),
            NullMemoryStore(),
            checkpointer=saver,
            tracer=tracer,
        )
        task = Task(description="research")
        with tracer.start_run(task.id, task.description):
            graph.invoke({"task": task}, {"configurable": {"thread_id": task.id}})
        try:
            fork_run(graph, thread_id=task.id, at_node="nope", edits={}, tracer=tracer)
            raise AssertionError("expected ValueError")
        except ValueError:
            pass
    store.close()

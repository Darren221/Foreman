"""Phase 4 checkpoint: one run, the whole observability surface.

A task runs under the tracer; its trace is recorded and reconstructable (run /
node / tool / llm spans, including the memory nodes); the cost summary totals
match the spans; and the run can be forked at a node with an edited input to
diverge — all from the same recorded run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver

from foreman.graph.builder import build_graph
from foreman.llm.base import LLMProvider, T, Usage
from foreman.observability import OTelTracer, SpanNode, TraceStore, fork_run, summarize
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


class EchoUsageProvider(LLMProvider):
    """Echoes the plan's topic into the result (so an edit is visible) and reports
    token usage (so the run can be costed)."""

    name = "echo-usage"
    model = "gpt-4o"

    def __init__(self, plan: Plan) -> None:
        self._plan = plan

    def structured_complete(self, prompt: str, schema: type[T]) -> T:
        self.last_usage = Usage(input_tokens=13, output_tokens=9)
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


def _all(root: SpanNode) -> list[SpanNode]:
    nodes = [root]
    for child in root.children:
        nodes.extend(_all(child))
    return nodes


def test_observability_end_to_end(tmp_path: Path) -> None:
    store = TraceStore(tmp_path / "traces.sqlite")
    tracer = OTelTracer(store)
    with SqliteSaver.from_conn_string(str(tmp_path / "c.sqlite")) as saver:
        graph = build_graph(
            EchoUsageProvider(_plan("bicycles")),
            _registry(),
            NullMemoryStore(),
            checkpointer=saver,
            tracer=tracer,
        )
        task = Task(description="research bicycles")
        config = {"configurable": {"thread_id": task.id}}
        with tracer.start_run(task.id, task.description) as span:
            original = graph.invoke({"task": task}, config)
            original_trace_id = f"{span.get_span_context().trace_id:032x}"

        # Recorded and reconstructable: every span kind, including memory nodes.
        root = store.get_trace(original_trace_id)
        assert root is not None
        spans = _all(root)
        assert {s.span.kind for s in spans} >= {"run", "node", "tool", "llm"}
        node_names = {s.span.name for s in spans if s.span.kind == "node"}
        assert {"node:retrieve", "node:remember"} <= node_names

        # Costed: totals match the recorded usage.
        cost = summarize(root)
        assert cost.tool_calls == 1
        assert cost.total.input_tokens > 0
        assert cost.total.cost_usd > 0
        assert "gpt-4o" in cost.by_model

        # Forkable: re-run from execute with an edited plan and diverge.
        forked = fork_run(
            graph,
            thread_id=task.id,
            at_node="execute",
            edits={"plan": _plan("MODIFIED")},
            tracer=tracer,
        )
        assert "MODIFIED" in (forked.result or "")
        assert "MODIFIED" not in (original.get("result") or "")
        assert len(store.list_runs()) == 2  # original + fork, side by side
    store.close()

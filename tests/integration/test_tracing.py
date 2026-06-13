"""O1: a task run under the tracer records a real OpenTelemetry span tree — a root
`run` span with the node spans nested beneath it — into the SQLite trace store,
fully offline. With no tracer the pipeline is unchanged (covered elsewhere)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver

from foreman.graph import run_task
from foreman.graph.builder import build_graph
from foreman.llm.base import LLMProvider, T, Usage
from foreman.observability import OTelTracer, SpanNode, TraceStore, summarize
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


class CannedProvider(LLMProvider):
    name = "canned"

    def __init__(self, plan: Plan) -> None:
        self._plan = plan

    def structured_complete(self, prompt: str, schema: type[T]) -> T:
        if schema is ResearchFindings:
            return ResearchFindings(content="findings")  # type: ignore[return-value]
        if schema is ReviewResult:
            return ReviewResult(passed=True, score=0.9, feedback="ok")  # type: ignore[return-value]
        if schema is Synthesis:
            return Synthesis(result="the answer")  # type: ignore[return-value]
        return self._plan  # type: ignore[return-value]


class UsageProvider(CannedProvider):
    """A canned provider that reports model + token usage, the way a real adapter
    surfaces it, so llm spans can carry `gen_ai.*` attributes."""

    name = "usage"

    def __init__(self, plan: Plan, model: str = "fake-model") -> None:
        super().__init__(plan)
        self.model = model

    def structured_complete(self, prompt: str, schema: type[T]) -> T:
        self.last_usage = Usage(input_tokens=11, output_tokens=7)
        return super().structured_complete(prompt, schema)


def _find(node: SpanNode, name: str) -> SpanNode | None:
    if node.span.name == name:
        return node
    for child in node.children:
        found = _find(child, name)
        if found is not None:
            return found
    return None


def _collect(node: SpanNode, kind: str) -> list[SpanNode]:
    found = [node] if node.span.kind == kind else []
    for child in node.children:
        found.extend(_collect(child, kind))
    return found


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


def test_traced_run_records_a_span_tree(tmp_path: Path) -> None:
    store = TraceStore(tmp_path / "traces.sqlite")
    tracer = OTelTracer(store)

    run_task(
        CannedProvider(_plan()),
        Task(description="research the bicycle"),
        registry=_registry(),
        memory_store=NullMemoryStore(),
        tracer=tracer,
    )

    runs = store.list_runs()
    assert len(runs) == 1
    root = store.get_trace(runs[0].trace_id)
    assert root is not None
    assert root.span.kind == "run"
    assert root.span.attributes["foreman.task"] == "research the bicycle"

    node_names = {child.span.name for child in root.children}
    assert "node:plan" in node_names  # node spans nest under the run span
    assert "node:synthesize" in node_names
    store.close()


def test_tool_and_llm_spans_nest_under_their_nodes(tmp_path: Path) -> None:
    store = TraceStore(tmp_path / "traces.sqlite")
    run_task(
        UsageProvider(_plan()),
        Task(description="research the bicycle"),
        registry=_registry(),
        memory_store=NullMemoryStore(),
        tracer=OTelTracer(store),
    )

    root = store.get_trace(store.list_runs()[0].trace_id)
    assert root is not None

    execute = _find(root, "node:execute")
    assert execute is not None
    tool_spans = [c for c in execute.children if c.span.kind == "tool"]
    assert [s.span.name for s in tool_spans] == ["tool:web_search"]

    llm_spans = _collect(root, "llm")
    assert llm_spans  # the supervisor/researcher/reviewer calls are traced
    assert all(s.span.attributes.get("gen_ai.request.model") == "fake-model" for s in llm_spans)
    assert any(s.span.attributes.get("gen_ai.usage.input_tokens") == 11 for s in llm_spans)
    store.close()


def test_summarize_costs_a_real_recorded_run(tmp_path: Path) -> None:
    store = TraceStore(tmp_path / "traces.sqlite")
    run_task(
        UsageProvider(_plan(), model="gpt-4o"),
        Task(description="research the bicycle"),
        registry=_registry(),
        memory_store=NullMemoryStore(),
        tracer=OTelTracer(store),
    )

    root = store.get_trace(store.list_runs()[0].trace_id)
    assert root is not None
    cost = summarize(root)
    assert cost.tool_calls == 1
    assert cost.total.input_tokens > 0
    assert cost.total.cost_usd > 0
    assert "node:plan" in cost.by_agent
    store.close()


def test_escalation_is_recorded_as_a_span_event(tmp_path: Path) -> None:
    store = TraceStore(tmp_path / "traces.sqlite")
    tracer = OTelTracer(store)
    with SqliteSaver.from_conn_string(str(tmp_path / "c.sqlite")) as saver:
        graph = build_graph(
            UsageProvider(_plan()),
            _registry(),
            NullMemoryStore(),
            checkpointer=saver,
            tracer=tracer,
        )
        task = Task(description="wire the funds", sensitive=True)
        with tracer.start_run(task.id, task.description):
            graph.invoke({"task": task}, {"configurable": {"thread_id": "r1"}})

    root = store.get_trace(store.list_runs()[0].trace_id)
    assert root is not None
    approval = _find(root, "node:approval")
    assert approval is not None
    event_names = {event["name"] for event in approval.span.events}
    assert "escalation" in event_names
    store.close()

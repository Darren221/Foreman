"""O3: pricing an llm span from token usage, and aggregating a trace into a per-run
cost/performance summary (totals, by model, by agent, tool calls, wall-clock)."""

from __future__ import annotations

import logging

from foreman.observability import cost_of, summarize
from foreman.observability.semconv import (
    GEN_AI_REQUEST_MODEL,
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
)
from foreman.observability.store import SpanNode, SpanRecord


def _span(
    name: str, kind: str, attrs: dict[str, object] | None = None, end_ns: int = 0
) -> SpanRecord:
    return SpanRecord(
        span_id=name,
        trace_id="t",
        parent_id=None,
        name=name,
        kind=kind,
        start_ns=0,
        end_ns=end_ns,
        status="ok",
        attributes=attrs or {},
    )


def _llm(model: str, input_tokens: int, output_tokens: int) -> SpanNode:
    return SpanNode(
        _span(
            f"llm:{model}",
            "llm",
            {
                GEN_AI_REQUEST_MODEL: model,
                GEN_AI_USAGE_INPUT_TOKENS: input_tokens,
                GEN_AI_USAGE_OUTPUT_TOKENS: output_tokens,
            },
        )
    )


def test_cost_of_known_model() -> None:
    # gpt-4o is priced at 0.0025/1k input + 0.01/1k output.
    assert cost_of("gpt-4o", 1000, 1000) == 0.0025 + 0.01


def test_unknown_model_costs_zero_and_warns(caplog: object) -> None:
    with caplog.at_level(logging.WARNING):  # type: ignore[attr-defined]
        assert cost_of("mystery-model", 1000, 1000) == 0.0
    assert "mystery-model" in caplog.text  # type: ignore[attr-defined]


def test_summarize_aggregates_a_trace() -> None:
    root = SpanNode(
        _span("run", "run", end_ns=5_000_000),
        children=[
            SpanNode(_span("node:plan", "node"), children=[_llm("gpt-4o", 100, 50)]),
            SpanNode(
                _span("node:execute", "node"),
                children=[SpanNode(_span("tool:web_search", "tool")), _llm("gpt-4o", 200, 100)],
            ),
        ],
    )

    cost = summarize(root)
    assert cost.tool_calls == 1
    assert cost.wall_clock_ms == 5.0
    assert cost.total.input_tokens == 300
    assert cost.total.output_tokens == 150
    assert cost.total.cost_usd > 0
    assert cost.by_model["gpt-4o"].input_tokens == 300
    assert set(cost.by_agent) == {"node:plan", "node:execute"}
    assert cost.by_agent["node:plan"].input_tokens == 100
    assert cost.by_agent["node:execute"].output_tokens == 100

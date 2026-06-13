"""O4: the Streamlit-free logic under the trace explorer — flattening the span
tree for display, status colouring, latency/cost formatting, and pulling an llm
span's prompt/response. The Streamlit page is a thin renderer over these."""

from __future__ import annotations

from foreman.observability.semconv import (
    GEN_AI_REQUEST_MODEL,
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
)
from foreman.observability.store import SpanNode, SpanRecord
from foreman.observability.tracer import PROMPT_ATTR, RESPONSE_ATTR
from foreman.ui.explorer_wiring import flatten, format_detail, span_content, status_icon


def _rec(
    name: str,
    kind: str,
    status: str = "ok",
    attrs: dict[str, object] | None = None,
    events: list[dict[str, object]] | None = None,
    end_ns: int = 0,
) -> SpanRecord:
    return SpanRecord(
        span_id=f"id-{name}",
        trace_id="t",
        parent_id=None,
        name=name,
        kind=kind,
        start_ns=0,
        end_ns=end_ns,
        status=status,
        attributes=attrs or {},
        events=events or [],
    )


def test_flatten_is_depth_first_with_depth() -> None:
    root = SpanNode(
        _rec("run", "run"),
        children=[
            SpanNode(_rec("node:plan", "node"), children=[SpanNode(_rec("llm:Plan", "llm"))]),
            SpanNode(_rec("node:execute", "node")),
        ],
    )
    assert [(row.span.name, row.depth) for row in flatten(root)] == [
        ("run", 0),
        ("node:plan", 1),
        ("llm:Plan", 2),
        ("node:execute", 1),
    ]


def test_status_icon_reflects_status_and_escalation() -> None:
    assert status_icon(_rec("a", "node")) == "🟢"
    assert status_icon(_rec("a", "node", status="error")) == "🔴"
    escalated = _rec("a", "node", events=[{"name": "escalation", "attributes": {}}])
    assert status_icon(escalated) == "🟡"


def test_format_detail_includes_tokens_and_cost_for_llm() -> None:
    span = _rec(
        "llm:Plan",
        "llm",
        end_ns=2_000_000,
        attrs={
            GEN_AI_REQUEST_MODEL: "gpt-4o",
            GEN_AI_USAGE_INPUT_TOKENS: 1000,
            GEN_AI_USAGE_OUTPUT_TOKENS: 0,
        },
    )
    detail = format_detail(span)
    assert "2.0 ms" in detail
    assert "1000+0 tok" in detail
    assert "$" in detail


def test_span_content_only_for_llm_spans() -> None:
    llm = _rec("llm:Plan", "llm", attrs={PROMPT_ATTR: "the prompt", RESPONSE_ATTR: "{}"})
    assert span_content(llm) == ("the prompt", "{}")
    assert span_content(_rec("node:plan", "node")) == (None, None)

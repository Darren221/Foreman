"""The logic beneath the trace explorer, kept free of Streamlit so it can be
tested: flattening the span tree, status colouring, latency/cost formatting, and
pulling an llm span's captured prompt/response. The page just renders these.
"""

from __future__ import annotations

from dataclasses import dataclass

from foreman.observability.cost import cost_of
from foreman.observability.semconv import (
    GEN_AI_REQUEST_MODEL,
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
)
from foreman.observability.store import SpanNode, SpanRecord
from foreman.observability.tracer import PROMPT_ATTR, RESPONSE_ATTR


@dataclass
class Row:
    """A span placed in the display tree at a given indentation depth."""

    span: SpanRecord
    depth: int


def flatten(root: SpanNode, depth: int = 0) -> list[Row]:
    """Depth-first preorder, so the tree renders as an indented list."""
    rows = [Row(root.span, depth)]
    for child in root.children:
        rows.extend(flatten(child, depth + 1))
    return rows


def status_icon(span: SpanRecord) -> str:
    """Colour a span by outcome: failed, escalated, or ok."""
    if span.status == "error":
        return "🔴"
    if any(event.get("name") == "escalation" for event in span.events):
        return "🟡"
    return "🟢"


def format_detail(span: SpanRecord) -> str:
    """A one-line metric for the row: latency, plus tokens and cost for llm spans."""
    parts = [f"{span.duration_ms:.1f} ms"]
    if span.kind == "llm":
        model = str(span.attributes.get(GEN_AI_REQUEST_MODEL, ""))
        input_tokens = int(span.attributes.get(GEN_AI_USAGE_INPUT_TOKENS, 0) or 0)
        output_tokens = int(span.attributes.get(GEN_AI_USAGE_OUTPUT_TOKENS, 0) or 0)
        parts.append(f"{input_tokens}+{output_tokens} tok")
        parts.append(f"${cost_of(model, input_tokens, output_tokens):.4f}")
    return " · ".join(parts)


def span_content(span: SpanRecord) -> tuple[str | None, str | None]:
    """The captured (prompt, response) for an llm span, else (None, None)."""
    if span.kind != "llm":
        return (None, None)
    return (span.attributes.get(PROMPT_ATTR), span.attributes.get(RESPONSE_ATTR))

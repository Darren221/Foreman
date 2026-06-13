"""Cost and performance: turn a recorded trace into a per-run summary.

Pricing is derived after the fact from the token counts on the `llm` spans, so a
trace recorded once can be re-costed if prices change. The price table is small
and illustrative — a production system would source live prices — and an unknown
model degrades to zero cost with a warning rather than guessing or crashing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from foreman.observability.semconv import (
    GEN_AI_REQUEST_MODEL,
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
)
from foreman.observability.store import SpanNode

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelPrice:
    """USD per 1K tokens."""

    input_per_1k: float
    output_per_1k: float


# Illustrative list prices (USD / 1K tokens). Not authoritative — the point is the
# mechanism, not a live billing feed.
_PRICES: dict[str, ModelPrice] = {
    "gpt-4o": ModelPrice(0.0025, 0.01),
    "gpt-4o-mini": ModelPrice(0.00015, 0.0006),
    "claude-sonnet-4-6": ModelPrice(0.003, 0.015),
    "text-embedding-3-small": ModelPrice(0.00002, 0.0),
}


@dataclass
class CostLine:
    """Tokens and cost for one slice of a run (overall, a model, or an agent)."""

    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0

    def add(self, input_tokens: int, output_tokens: int, cost_usd: float) -> None:
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.cost_usd += cost_usd


@dataclass
class RunCost:
    """A run's cost and performance, broken down by model and by agent."""

    wall_clock_ms: float
    tool_calls: int = 0
    total: CostLine = field(default_factory=CostLine)
    by_model: dict[str, CostLine] = field(default_factory=dict)
    by_agent: dict[str, CostLine] = field(default_factory=dict)


def cost_of(model: str, input_tokens: int, output_tokens: int) -> float:
    """Price a completion. An unknown model costs nothing and warns once per call."""
    price = _PRICES.get(model)
    if price is None:
        _log.warning("no price for model %r; counting it as zero cost", model)
        return 0.0
    return input_tokens / 1000 * price.input_per_1k + output_tokens / 1000 * price.output_per_1k


def summarize(root: SpanNode) -> RunCost:
    """Aggregate a recorded trace into per-run cost and performance totals.

    Each llm span is attributed to its nearest enclosing node (the agent stage it
    ran under) as well as to its model.
    """
    cost = RunCost(wall_clock_ms=root.span.duration_ms)
    _visit(root, root.span.name, cost)
    return cost


def _visit(node: SpanNode, agent: str, cost: RunCost) -> None:
    span = node.span
    if span.kind == "node":
        agent = span.name
    elif span.kind == "tool":
        cost.tool_calls += 1
    elif span.kind == "llm":
        model = str(span.attributes.get(GEN_AI_REQUEST_MODEL, ""))
        input_tokens = int(span.attributes.get(GEN_AI_USAGE_INPUT_TOKENS, 0) or 0)
        output_tokens = int(span.attributes.get(GEN_AI_USAGE_OUTPUT_TOKENS, 0) or 0)
        usd = cost_of(model, input_tokens, output_tokens)
        cost.total.add(input_tokens, output_tokens, usd)
        cost.by_model.setdefault(model, CostLine()).add(input_tokens, output_tokens, usd)
        cost.by_agent.setdefault(agent, CostLine()).add(input_tokens, output_tokens, usd)

    for child in node.children:
        _visit(child, agent, cost)

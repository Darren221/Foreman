"""Observability: real OpenTelemetry tracing exported to an embedded trace store,
plus the cost and replay layers built on it.

O1 provides the spine — a `Tracer` seam (no-op by default), a custom OTel
`SqliteSpanExporter`, and a `TraceStore` that rebuilds the span tree.
"""

from __future__ import annotations

import os

from foreman.observability.cost import CostLine, RunCost, cost_of, summarize
from foreman.observability.exporter import SqliteSpanExporter
from foreman.observability.instrument import TracingProvider
from foreman.observability.store import RunSummary, SpanNode, SpanRecord, TraceStore
from foreman.observability.tracer import NoOpTracer, OTelTracer, Tracer

# Opt into the experimental GenAI semantic conventions. We emit `gen_ai.*` keys as
# literals (see semconv.py), so this is declarative rather than load-order
# critical — it records the honest fact that the convention is still stabilizing.
os.environ.setdefault("OTEL_SEMCONV_STABILITY_OPT_IN", "gen_ai_latest_experimental")

__all__ = [
    "CostLine",
    "NoOpTracer",
    "OTelTracer",
    "RunCost",
    "RunSummary",
    "SpanNode",
    "SpanRecord",
    "SqliteSpanExporter",
    "TraceStore",
    "Tracer",
    "TracingProvider",
    "cost_of",
    "summarize",
]

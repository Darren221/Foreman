"""Observability: real OpenTelemetry tracing exported to an embedded trace store,
plus the cost and replay layers built on it.

O1 provides the spine — a `Tracer` seam (no-op by default), a custom OTel
`SqliteSpanExporter`, and a `TraceStore` that rebuilds the span tree.
"""

from __future__ import annotations

from foreman.observability.exporter import SqliteSpanExporter
from foreman.observability.store import RunSummary, SpanNode, SpanRecord, TraceStore
from foreman.observability.tracer import NoOpTracer, OTelTracer, Tracer

__all__ = [
    "NoOpTracer",
    "OTelTracer",
    "RunSummary",
    "SpanNode",
    "SpanRecord",
    "SqliteSpanExporter",
    "TraceStore",
    "Tracer",
]

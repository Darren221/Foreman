"""A custom OpenTelemetry span exporter that writes into our SQLite trace store.

OpenTelemetry's model is spans -> exporter -> backend. Rather than ship to a
collector/Jaeger, we export to the embedded `TraceStore` our own explorer and
replay read directly — keeping the "no server" theme while using the real OTel
SDK (the spans are genuine OTel spans, not a look-alike). A Jaeger exporter could
be added alongside this one later as a flourish.
"""

from __future__ import annotations

from collections.abc import Sequence

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.trace import StatusCode

from foreman.observability.store import SpanRecord, TraceStore


class SqliteSpanExporter(SpanExporter):
    """Converts finished OTel spans to `SpanRecord`s and stores them."""

    def __init__(self, store: TraceStore) -> None:
        self._store = store

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        for span in spans:
            context = span.get_span_context()
            if context is None:
                continue
            attributes = dict(span.attributes or {})
            self._store.record_span(
                SpanRecord(
                    span_id=f"{context.span_id:016x}",
                    trace_id=f"{context.trace_id:032x}",
                    parent_id=f"{span.parent.span_id:016x}" if span.parent else None,
                    name=span.name,
                    kind=str(attributes.get("foreman.kind", "span")),
                    start_ns=span.start_time or 0,
                    end_ns=span.end_time or 0,
                    status="error" if span.status.status_code is StatusCode.ERROR else "ok",
                    attributes=attributes,
                )
            )
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        return None

"""The tracer seam: where the pipeline emits spans, kept injectable and no-op by
default so tracing is opt-in and the untraced path is unchanged.

`OTelTracer` drives the real OpenTelemetry SDK against a *local* `TracerProvider`
(no global state, so tests stay isolated) and exports through `SqliteSpanExporter`
to the trace store. Spans nest via OTel's own context, so a `span()` opened inside
a `start_run()` becomes its child automatically — that is what builds the tree.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from typing import Any, Protocol

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from foreman.observability.exporter import SqliteSpanExporter
from foreman.observability.store import TraceStore

# Attribute key for our semantic span role (run/node/tool/llm/memory).
KIND = "foreman.kind"


class Tracer(Protocol):
    """What the graph needs from a tracer: a root run span and nested child spans."""

    def start_run(self, run_id: str, task: str = "") -> AbstractContextManager[Any]: ...

    def span(
        self, name: str, kind: str = "node", attributes: dict[str, Any] | None = None
    ) -> AbstractContextManager[Any]: ...


class NoOpTracer:
    """The default: records nothing, so an untraced run behaves exactly as before."""

    def start_run(self, run_id: str, task: str = "") -> AbstractContextManager[Any]:
        return nullcontext()

    def span(
        self, name: str, kind: str = "node", attributes: dict[str, Any] | None = None
    ) -> AbstractContextManager[Any]:
        return nullcontext()


class OTelTracer:
    """Real OpenTelemetry spans, exported to a `TraceStore`."""

    def __init__(self, store: TraceStore) -> None:
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(SqliteSpanExporter(store)))
        self._tracer = provider.get_tracer("foreman")

    @contextmanager
    def start_run(self, run_id: str, task: str = "") -> Iterator[Any]:
        attributes = {KIND: "run", "foreman.run_id": run_id, "foreman.task": task}
        with self._tracer.start_as_current_span("run", attributes=attributes) as span:
            yield span

    @contextmanager
    def span(
        self, name: str, kind: str = "node", attributes: dict[str, Any] | None = None
    ) -> Iterator[Any]:
        attrs: dict[str, Any] = {KIND: kind}
        if attributes:
            attrs.update(attributes)
        with self._tracer.start_as_current_span(name, attributes=attrs) as span:
            yield span

"""O1: the trace store persists spans and rebuilds the parent/child tree, and
lists recorded runs (root spans). No OpenTelemetry here — just storage.

Parameterized over both storage backends via the `open_conn` factory: the same
assertions run against embedded SQLite always, and against a real Postgres when
FOREMAN_TEST_POSTGRES is set (C3's acceptance criterion)."""

from __future__ import annotations

from collections.abc import Callable

from foreman.observability import SpanRecord, TraceStore
from foreman.storage import Conn


def _span(span_id: str, parent_id: str | None, name: str, kind: str, **attrs: object) -> SpanRecord:
    return SpanRecord(
        span_id=span_id,
        trace_id="trace-1",
        parent_id=parent_id,
        name=name,
        kind=kind,
        start_ns=0,
        end_ns=1,
        status="ok",
        attributes=dict(attrs),
    )


def test_store_rebuilds_the_span_tree(open_conn: Callable[[], Conn]) -> None:
    store = TraceStore(open_conn())
    store.record_span(_span("a", None, "run", "run", **{"foreman.task": "research bikes"}))
    store.record_span(_span("b", "a", "node:plan", "node"))
    store.record_span(_span("c", "a", "node:execute", "node"))

    root = store.get_trace("trace-1")
    assert root is not None
    assert root.span.kind == "run"
    assert root.span.attributes["foreman.task"] == "research bikes"
    assert sorted(child.span.name for child in root.children) == ["node:execute", "node:plan"]
    store.close()


def test_list_runs_returns_root_spans(open_conn: Callable[[], Conn]) -> None:
    store = TraceStore(open_conn())
    store.record_span(_span("a", None, "run", "run", **{"foreman.task": "t"}))
    store.record_span(_span("b", "a", "node:plan", "node"))

    runs = store.list_runs()
    assert len(runs) == 1
    assert runs[0].trace_id == "trace-1"
    store.close()


def test_get_trace_survives_reopen(open_conn: Callable[[], Conn]) -> None:
    store = TraceStore(open_conn())
    store.record_span(_span("a", None, "run", "run"))
    store.close()

    reopened = TraceStore(open_conn())
    assert reopened.get_trace("trace-1") is not None
    reopened.close()

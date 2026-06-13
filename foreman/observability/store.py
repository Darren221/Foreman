"""The trace store: where recorded spans live and the tree is rebuilt from them.

A span is the unit of a trace — one timed step (a graph node, a tool call, an LLM
call) with a parent. Stored flat in SQLite (embedded, no server — the same theme
as the memory store and checkpointer); `get_trace` reassembles the parent/child
tree the explorer and replay read. This is the read/write seam the custom
OpenTelemetry exporter writes into.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SpanRecord:
    """One recorded span. `kind` is our semantic role (run/node/tool/llm/memory),
    carried separately from OpenTelemetry's transport-level SpanKind."""

    span_id: str
    trace_id: str
    parent_id: str | None
    name: str
    kind: str
    start_ns: int
    end_ns: int
    status: str
    attributes: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        return (self.end_ns - self.start_ns) / 1e6


@dataclass
class SpanNode:
    """A span plus its children — a node in the reconstructed trace tree."""

    span: SpanRecord
    children: list[SpanNode] = field(default_factory=list)


@dataclass
class RunSummary:
    """A recorded run, identified by its trace, for listing in the explorer."""

    trace_id: str
    name: str
    start_ns: int
    attributes: dict[str, Any]


class TraceStore:
    """SQLite-backed span storage. One connection per instance; reopening the same
    file in another process sees the same traces."""

    def __init__(self, path: str | Path) -> None:
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS spans (
                span_id    TEXT PRIMARY KEY,
                trace_id   TEXT NOT NULL,
                parent_id  TEXT,
                name       TEXT NOT NULL,
                kind       TEXT NOT NULL,
                start_ns   INTEGER NOT NULL,
                end_ns     INTEGER NOT NULL,
                status     TEXT NOT NULL,
                attributes TEXT NOT NULL
            )
            """
        )
        self._conn.commit()

    def record_span(self, span: SpanRecord) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO spans "
            "(span_id, trace_id, parent_id, name, kind, start_ns, end_ns, status, attributes) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                span.span_id,
                span.trace_id,
                span.parent_id,
                span.name,
                span.kind,
                span.start_ns,
                span.end_ns,
                span.status,
                json.dumps(span.attributes),
            ),
        )
        self._conn.commit()

    def get_trace(self, trace_id: str) -> SpanNode | None:
        """Rebuild the span tree for a trace, or None if there are no spans."""
        rows = self._conn.execute(
            "SELECT * FROM spans WHERE trace_id = ? ORDER BY start_ns", (trace_id,)
        ).fetchall()
        if not rows:
            return None

        nodes = {row["span_id"]: SpanNode(self._to_record(row)) for row in rows}
        root: SpanNode | None = None
        for node in nodes.values():
            parent_id = node.span.parent_id
            if parent_id is not None and parent_id in nodes:
                nodes[parent_id].children.append(node)
            elif parent_id is None:
                root = node
        # Fall back to the earliest span if the recorded root isn't a clean parent.
        return root or next(iter(nodes.values()))

    def list_runs(self) -> list[RunSummary]:
        """Recorded runs (root spans), newest first."""
        rows = self._conn.execute(
            "SELECT * FROM spans WHERE parent_id IS NULL ORDER BY start_ns DESC"
        ).fetchall()
        return [
            RunSummary(
                trace_id=row["trace_id"],
                name=row["name"],
                start_ns=row["start_ns"],
                attributes=json.loads(row["attributes"]),
            )
            for row in rows
        ]

    def close(self) -> None:
        self._conn.close()

    @staticmethod
    def _to_record(row: sqlite3.Row) -> SpanRecord:
        return SpanRecord(
            span_id=row["span_id"],
            trace_id=row["trace_id"],
            parent_id=row["parent_id"],
            name=row["name"],
            kind=row["kind"],
            start_ns=row["start_ns"],
            end_ns=row["end_ns"],
            status=row["status"],
            attributes=json.loads(row["attributes"]),
        )

"""The approval queue: durable storage for escalations awaiting a human.

When a run pauses for approval, its `Escalation` is persisted here and the run's
checkpoint is held by the graph's checkpointer. The two together are what let a
paused run outlive the process: the queue says *what* needs deciding, the
checkpointer holds the state to resume into. The queue rides the storage seam, so
it runs on embedded SQLite locally and on shared Postgres once the API and workers
are separate processes.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from foreman.hitl.policy import Escalation
from foreman.schemas import Plan
from foreman.storage import Conn


class DecisionKind(StrEnum):
    """How the human resolved an approval."""

    APPROVE = "approve"  # proceed as planned
    REJECT = "reject"  # redo the work, addressing `feedback`
    MODIFY = "modify"  # proceed, but with the human's replacement `plan`
    TAKE_OVER = "take_over"  # the human supplies `output`; the agents stand down


class Decision(BaseModel):
    """A human's resolution of an escalation. Only the field relevant to the kind
    is populated (feedback for reject, plan for modify, output for take-over)."""

    kind: DecisionKind
    feedback: str = ""
    plan: Plan | None = None
    output: str = ""


class PendingApproval(BaseModel):
    """A row in the queue: the escalation, the run it belongs to, and — once
    resolved — the decision applied to it."""

    id: str
    thread_id: str
    escalation: Escalation
    created_at: datetime
    resolved: bool = False
    decision: Decision | None = None


class ApprovalQueue:
    """A queue of approvals over the storage seam. One connection per instance; safe
    to reopen the same store in another process and see the same pending items.

    Construct with a `Conn` (the factory injects one for the configured backend) or
    a path, which is sugar for an embedded SQLite connection. The DDL below is valid
    on both backends — only TEXT and INTEGER, no backend-specific types."""

    def __init__(self, conn: Conn | str | Path) -> None:
        self._conn = conn if isinstance(conn, Conn) else Conn.sqlite(conn)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS approvals (
                id          TEXT PRIMARY KEY,
                thread_id   TEXT NOT NULL,
                escalation  TEXT NOT NULL,
                created_at  TEXT NOT NULL,
                resolved    INTEGER NOT NULL DEFAULT 0,
                decision    TEXT
            )
            """
        )
        self._conn.commit()

    def enqueue(self, escalation: Escalation, thread_id: str) -> str:
        approval_id = uuid.uuid4().hex[:12]
        self._conn.execute(
            "INSERT INTO approvals (id, thread_id, escalation, created_at, resolved) "
            "VALUES (?, ?, ?, ?, 0)",
            (approval_id, thread_id, escalation.model_dump_json(), datetime.now(UTC).isoformat()),
        )
        self._conn.commit()
        return approval_id

    def pending(self) -> list[PendingApproval]:
        rows = self._conn.execute(
            "SELECT * FROM approvals WHERE resolved = 0 ORDER BY created_at"
        ).fetchall()
        return [self._to_model(row) for row in rows]

    def get(self, approval_id: str) -> PendingApproval | None:
        row = self._conn.execute(
            "SELECT * FROM approvals WHERE id = ?", (approval_id,)
        ).fetchone()
        return self._to_model(row) if row is not None else None

    def resolve(self, approval_id: str, decision: Decision) -> PendingApproval:
        cursor = self._conn.execute(
            "UPDATE approvals SET resolved = 1, decision = ? WHERE id = ? AND resolved = 0",
            (decision.model_dump_json(), approval_id),
        )
        if cursor.rowcount == 0:
            raise ValueError(f"no pending approval with id {approval_id!r}")
        self._conn.commit()
        resolved = self.get(approval_id)
        assert resolved is not None
        return resolved

    def close(self) -> None:
        self._conn.close()

    @staticmethod
    def _to_model(row: Any) -> PendingApproval:
        return PendingApproval(
            id=row["id"],
            thread_id=row["thread_id"],
            escalation=Escalation.model_validate_json(row["escalation"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            resolved=bool(row["resolved"]),
            decision=(
                Decision.model_validate_json(row["decision"])
                if row["decision"] is not None
                else None
            ),
        )

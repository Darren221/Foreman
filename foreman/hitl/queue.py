"""The approval queue: durable storage for escalations awaiting a human.

When a run pauses for approval, its `Escalation` is persisted here and the run's
checkpoint is held by the graph's checkpointer. The two together are what let a
paused run outlive the process: the queue says *what* needs deciding, the
checkpointer holds the state to resume into. SQLite keeps the "embedded, no
server" theme shared with the memory store and the checkpointer.
"""

from __future__ import annotations

import sqlite3
import uuid
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel

from foreman.hitl.policy import Escalation
from foreman.schemas import Plan


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
    """A SQLite-backed queue of approvals. One connection per instance; safe to
    reopen the same file in another process and see the same pending items."""

    def __init__(self, path: str | Path) -> None:
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
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
    def _to_model(row: sqlite3.Row) -> PendingApproval:
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

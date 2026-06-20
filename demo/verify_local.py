#!/usr/bin/env python3
"""Local end-to-end verification of the showcase, with the REAL Anthropic crew.

This proves the showcase works without needing Docker or Postgres, so it can run in
a constrained CI/agent sandbox. Two pieces are swapped for daemon-free equivalents,
clearly and only because the host has no Docker daemon / no installable Chroma; the
*logic under test* (planning, the dependency wave, real db_query against real data,
the real reviewer, the HITL gate, synthesis) is the production code:

  - sales data lives in a local SQLite file via a SQLite DatabaseBackend (same
    `db_query` tool, real SQL, real rows) instead of a Postgres container;
  - the code sandbox runs the analyst's stdlib Python in a subprocess instead of a
    Docker container (the analyst code is stdlib-only, so this is behaviourally
    equivalent for the demo);
  - memory is a no-op store (Chroma can't be installed on this Python).

The real Anthropic provider and real Tavily web search ARE used.

Usage:  python demo/verify_local.py
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from foreman.config import Settings
from foreman.hitl.queue import ApprovalQueue, Decision, DecisionKind
from foreman.hitl.runner import Runner
from foreman.llm import select_provider
from foreman.memory.store import MemoryStore
from foreman.schemas import Task, TaskMemory
from foreman.storage.db import Conn
from foreman.tools import (
    ApiCallTool,
    CodeExecutionTool,
    DatabaseQueryTool,
    FileTool,
    ToolRegistry,
    WebSearchTool,
)
from foreman.tools.api_call import UrllibBackend
from foreman.tools.web_search import TavilyBackend

TASK_TEXT = (
    "Analyze the sales database: find the worst-performing region this quarter "
    "(2026-Q2) by revenue, research likely market causes for that specific region, "
    "and recommend a concrete action plan to turn it around. "
    "The database has one table, sales(region TEXT, quarter TEXT, revenue INTEGER, "
    "units INTEGER, returns INTEGER), where quarter values look like '2026-Q2'."
)

_ROWS = [
    ("North", "2026-Q1", 920_000, 4_100, 95),
    ("South", "2026-Q1", 760_000, 3_500, 80),
    ("East", "2026-Q1", 880_000, 3_900, 88),
    ("West", "2026-Q1", 980_000, 4_300, 90),
    ("North", "2026-Q2", 965_000, 4_250, 92),
    ("South", "2026-Q2", 805_000, 3_650, 78),
    ("East", "2026-Q2", 905_000, 4_000, 85),
    ("West", "2026-Q2", 410_000, 1_850, 240),  # worst this quarter
]


class SQLiteSalesBackend:
    """A DatabaseBackend (the db_query tool's interface) over a SQLite file. Real
    SQL, real rows — only the engine differs from the Postgres demo."""

    def __init__(self, path: str) -> None:
        self._path = path

    def query(self, sql: str) -> list[dict[str, Any]]:
        conn = sqlite3.connect(self._path)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(sql).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def schema(self) -> str:
        conn = sqlite3.connect(self._path)
        try:
            triples = [
                (t, c[1], c[2])
                for (t,) in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
                for c in conn.execute(f"PRAGMA table_info({t})").fetchall()
            ]
        finally:
            conn.close()
        from foreman.tools.database import _format_schema

        return _format_schema(triples)


class SubprocessSandbox:
    """Runs stdlib Python in a subprocess (no Docker daemon needed here). Network
    isn't disabled at the OS level, but the analyst's code is stdlib-only by prompt,
    so this is behaviourally equivalent to the Docker sandbox for the demo."""

    def run(self, code: str) -> dict[str, Any]:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=20,
        )
        return {"stdout": proc.stdout, "stderr": proc.stderr, "exit_code": proc.returncode}


class NoopMemory(MemoryStore):
    def remember(self, memory: TaskMemory) -> None:
        pass

    def recall(self, query: str, k: int = 5) -> list[TaskMemory]:
        return []

    def delete(self, ids: list[str]) -> None:
        pass


def _seed_sqlite(path: str) -> None:
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE sales (id INTEGER PRIMARY KEY, region TEXT, quarter TEXT, "
        "revenue INTEGER, units INTEGER, returns INTEGER)"
    )
    conn.executemany(
        "INSERT INTO sales (region, quarter, revenue, units, returns) VALUES (?,?,?,?,?)",
        _ROWS,
    )
    conn.commit()
    conn.close()


def _registry(settings: Settings, db_path: str) -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(WebSearchTool(TavilyBackend(settings.tavily_api_key or "")))
    reg.register(CodeExecutionTool(SubprocessSandbox()))
    reg.register(FileTool(settings.workspace_path))
    reg.register(DatabaseQueryTool(SQLiteSalesBackend(db_path)))
    reg.register(ApiCallTool(UrllibBackend()))
    return reg


def _enable_eager_celery() -> None:
    # Run specialist subtasks in-process instead of dispatching to a Redis-brokered
    # worker, so the demo needs no broker. JSON crosses the boundary either way, so
    # behaviour is identical to a real worker (see celery_app.py).
    from foreman.workers.celery_app import app

    app.conf.task_always_eager = True
    app.conf.task_eager_propagates = False


def main() -> int:
    _enable_eager_celery()
    tmp = Path(tempfile.mkdtemp(prefix="foreman-verify-"))
    db_path = str(tmp / "sales.sqlite")
    _seed_sqlite(db_path)
    print(f"seeded SQLite sales DB at {db_path} ({len(_ROWS)} rows)")

    settings = Settings()
    provider = select_provider(settings)
    print(f"provider: {provider.name} {provider.model}")
    registry = _registry(settings, db_path)
    checkpointer = _sqlite_checkpointer(str(tmp / "ckpt.sqlite"))
    queue = ApprovalQueue(Conn.sqlite(str(tmp / "approvals.sqlite")))

    runner = Runner(
        provider=provider,
        registry=registry,
        memory_store=NoopMemory(),
        checkpointer=checkpointer,
        queue=queue,
    )

    task = Task(description=TASK_TEXT, sensitive=True)
    print("\n" + "=" * 72)
    print("SUBMIT (sensitive=True):", task.description)
    print("=" * 72)
    result = runner.submit(task)

    fired = {"hitl": False, "wave": False, "retry": False, "db_query": False}

    if result.status == "pending":
        fired["hitl"] = True
        esc = result.escalation
        print(f"\n[HITL] PAUSED: trigger={esc.trigger}  level={esc.level}")
        print(f"       reason: {esc.reason}")
        print("       PLAN:")
        for s in esc.plan.subtasks:
            print(
                f"         - {s.id} [{s.assigned_specialist}] "
                f"deps={s.dependencies}: {s.description}"
            )
        # Evidence of the dependency wave: at least one subtask depends on another,
        # and the researcher depends on the analyst (research keyed off analysis).
        deps = {s.id: s.dependencies for s in esc.plan.subtasks}
        spec = {s.id: s.assigned_specialist for s in esc.plan.subtasks}
        for sid, ds in deps.items():
            if str(spec[sid]) == "researcher" and any(
                str(spec[d]) == "analyst" for d in ds
            ):
                fired["wave"] = True
        print("\n[HITL] operator APPROVES the sensitive action plan")
        result = runner.resume(result.approval_id, Decision(kind=DecisionKind.APPROVE))

    print("\n" + "=" * 72)
    print("FINAL RECOMMENDATION")
    print("=" * 72)
    print(result.result)

    # Read the checkpointed state to report the reviewer's verdicts and how many
    # execute->review attempts ran (>1 means a retry fired).
    snapshot = runner._graph().get_state({"configurable": {"thread_id": task.id}})
    attempts = snapshot.values.get("attempts", 0)
    reviews = snapshot.values.get("subtask_reviews", {})
    fired["retry"] = attempts > 1
    print("\n" + "=" * 72)
    print(f"REVIEW (execute->review attempts: {attempts})")
    print("=" * 72)
    for sid, rv in reviews.items():
        print(f"  {sid}: passed={rv.passed} score={rv.score:.2f}  {rv.feedback[:80]}")

    print("\n" + "=" * 72)
    print("TOOL CALLS")
    print("=" * 72)
    for inv in registry.invocations:
        ok = "ok" if inv.success else f"ERR {inv.error}"
        extra = f"  sql={inv.inputs.get('query')!r}" if inv.tool == "db_query" else ""
        print(f"  {inv.tool:<15} by {inv.caller:<10} {ok}{extra}")
        if inv.tool == "db_query" and inv.success:
            fired["db_query"] = True
            print(f"      -> rows: {json.dumps(inv.output.get('rows'))[:300]}")

    print("\n" + "=" * 72)
    print("SHOWCASE FEATURE CHECK")
    print("=" * 72)
    print(f"  db_query against real data : {'YES' if fired['db_query'] else 'NO'}")
    print(f"  dependency wave (research<-analysis): {'YES' if fired['wave'] else 'NO'}")
    print(f"  HITL approval gate fired   : {'YES' if fired['hitl'] else 'NO'}")
    retry_msg = "YES" if fired["retry"] else "NO (all passed first pass)"
    print(f"  reviewer retry fired       : {retry_msg}")
    return 0


def _sqlite_checkpointer(path: str):  # type: ignore[no-untyped-def]
    from langgraph.checkpoint.sqlite import SqliteSaver

    return SqliteSaver(sqlite3.connect(path, check_same_thread=False))


if __name__ == "__main__":
    raise SystemExit(main())

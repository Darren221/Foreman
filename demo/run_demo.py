#!/usr/bin/env python3
"""Run the data-diagnosis-and-recommend showcase end to end, with the real crew.

This drives the FULL pipeline through the `Runner` (so the human-in-the-loop gates
are live), with the real Anthropic provider, the real tool registry (the analyst's
`db_query` pointed at the seeded sales DB, real Tavily web search, the Docker code
sandbox), a SQLite checkpointer, and an in-memory approval queue.

The task is flagged `sensitive=True` because it ends in an *action plan* — a step a
human should sign off on. That trips the SENSITIVE escalation at the pre-execution
gate, so the run pauses for approval; this script approves it and lets the run
finish, printing the evidence of each showcase feature as it goes.

Prereqs (see DEMO.md):
  - a throwaway Postgres seeded with demo/seed_sales.py
  - ANTHROPIC_API_KEY and TAVILY_API_KEY in .env
  - a working Docker daemon (for the code sandbox)
  - FOREMAN_DEMO_DSN pointing at the seeded sales DB

Usage:
  FOREMAN_DEMO_DSN=postgresql://demo:demo@localhost:55432/salesdemo \
      python demo/run_demo.py
"""

from __future__ import annotations

import os
import sqlite3
import sys
import tempfile

from foreman.config import Settings
from foreman.hitl.queue import ApprovalQueue, Decision, DecisionKind
from foreman.hitl.runner import Runner
from foreman.llm import select_provider
from foreman.memory import build_default_memory_store
from foreman.schemas import Task
from foreman.storage.db import Conn
from foreman.tools import build_default_registry

TASK_TEXT = (
    "Analyze the sales database: find the worst-performing region this quarter "
    "(2026-Q2) by revenue, research likely market causes for that specific region, "
    "and recommend a concrete action plan to turn it around. "
    "The database has one table, sales(region TEXT, quarter TEXT, revenue INTEGER, "
    "units INTEGER, returns INTEGER), where quarter values look like '2026-Q2'."
)


def main() -> int:
    dsn = os.environ.get("FOREMAN_DEMO_DSN")
    if not dsn:
        print("error: set FOREMAN_DEMO_DSN to the seeded sales DB", file=sys.stderr)
        return 2

    # Honour CELERY_TASK_ALWAYS_EAGER=true: run specialists in-process so the demo
    # needs no Redis broker. (The celery app is built at import with the env's value;
    # flip the live app to be safe when the var is set after import.)
    if os.environ.get("CELERY_TASK_ALWAYS_EAGER", "").lower() in {"1", "true", "yes"}:
        from foreman.workers.celery_app import app

        app.conf.task_always_eager = True
        app.conf.task_eager_propagates = False

    # Point the analyst's db_query at the seeded sales DB without touching the
    # operational store config (sqlite, the default).
    settings = Settings(analyst_database_dsn=dsn)  # type: ignore[call-arg]
    provider = select_provider(settings)
    registry = build_default_registry(settings)
    assert registry.has("db_query"), "db_query must be wired for this demo"
    memory_store = build_default_memory_store(settings)

    tmp = tempfile.mkdtemp(prefix="foreman-demo-")
    checkpointer = _sqlite_checkpointer(os.path.join(tmp, "checkpoints.sqlite"))
    queue = ApprovalQueue(Conn.sqlite(os.path.join(tmp, "approvals.sqlite")))

    runner = Runner(
        provider=provider,
        registry=registry,
        memory_store=memory_store,
        checkpointer=checkpointer,
        queue=queue,
    )

    task = Task(description=TASK_TEXT, sensitive=True)
    print("=" * 70)
    print("SUBMIT:", task.description)
    print("=" * 70)
    result = runner.submit(task)

    if result.status == "pending":
        esc = result.escalation
        print(f"\n[HITL] paused for approval: trigger={esc.trigger} level={esc.level}")
        print(f"       reason: {esc.reason}")
        print(f"       plan has {len(esc.plan.subtasks)} subtasks:")
        for s in esc.plan.subtasks:
            print(
                f"         - {s.id} [{s.assigned_specialist}] "
                f"deps={s.dependencies}: {s.description}"
            )
        print("\n[HITL] operator APPROVES the action plan...")
        result = runner.resume(result.approval_id, Decision(kind=DecisionKind.APPROVE))

    print("\n" + "=" * 70)
    print("FINAL RESULT")
    print("=" * 70)
    print(result.result)

    _print_tool_evidence(registry)
    return 0


def _sqlite_checkpointer(path: str):  # type: ignore[no-untyped-def]
    from langgraph.checkpoint.sqlite import SqliteSaver

    return SqliteSaver(sqlite3.connect(path, check_same_thread=False))


def _print_tool_evidence(registry) -> None:  # type: ignore[no-untyped-def]
    print("\n" + "=" * 70)
    print("TOOL CALLS (evidence db_query and the sandbox ran)")
    print("=" * 70)
    for inv in registry.invocations:
        ok = "ok" if inv.success else f"ERR {inv.error}"
        detail = ""
        if inv.tool == "db_query":
            detail = f"  sql={inv.inputs.get('query')!r}"
        print(f"  {inv.tool:<15} by {inv.caller:<10} {ok}{detail}")


if __name__ == "__main__":
    raise SystemExit(main())

# Flagship demo: data-diagnosis-and-recommend

This demo drives Foreman's full crew through one realistic task that exercises the
hardest features at once:

> "Analyze the sales database: find the worst-performing region this quarter
> (2026-Q2) by revenue, research likely market causes for that specific region, and
> recommend a concrete action plan to turn it around."

It is designed to trigger:

1. **A dependency wave.** The analyst must query the DB and name the weak region
   *before* the researcher can web-search causes for *that* region, and the writer
   can only draft the plan once both are in. So the plan is a real DAG:
   `analyst -> researcher -> writer` (the supervisor sometimes splits the analyst
   step into extract + diagnose, deepening the chain).
2. **`db_query` against real data.** The analyst writes SQL, runs it through the
   read-only `db_query` tool against a seeded `sales` table, and grounds its analysis
   in the returned rows. This is the tool that was previously the least demoed.
3. **Real Python compute.** The analyst also runs `code_execution` on the queried
   rows (sandboxed), so the demo proves Foreman is more than a web-research tool.
4. **A human-in-the-loop approval.** The task is submitted `sensitive=True` (an
   action plan is a step a human should sign off on), which trips the SENSITIVE
   escalation at the pre-execution gate. The run pauses; the operator approves; the
   run continues.

## The key plumbing finding (read this first)

Two things were true of `db_query` before this demo, and both are now fixed in the
worktree:

- **The analyst never called `db_query`.** It only wrote stdlib Python for the Docker
  sandbox (`code_execution`), and the sandbox runs with the network disabled, so it
  *cannot* reach a database. `db_query` was registered but unreachable by any agent.
  Fix: the analyst now has a query-then-analyze path. When a DB is wired it (a) reads
  the live schema, (b) writes a read-only `SELECT`/CTE, (c) runs it via `db_query`,
  then (d) feeds the rows into the existing code + write-up steps. With no DB wired it
  behaves exactly as before (so the offline tests are unchanged).
- **The analyst DB and the operational store shared one DSN.** `database_dsn` was used
  both for Foreman's checkpoints/approvals *and* for the analyst's data source. A new
  `analyst_database_dsn` setting decouples them (falling back to `database_dsn` for a
  single-DB deployment), and `db_query` is now registered only when a data source is
  actually configured.

Two smaller fixes the demo surfaced:

- The read-only guard rejected CTEs (`WITH ... SELECT`), which a real analyst leans on.
  It now accepts them; the read-only *connection* remains the actual write guarantee.
- The analyst was guessing column names. It now reads the real schema first
  (`information_schema` on Postgres, `PRAGMA` on SQLite), so its SQL hits real columns.

## How to run it (recommended: throwaway Postgres, real Docker crew)

This is the operator-facing demo on a machine with Docker and a funded LLM key.

```bash
# 1. throwaway Postgres on its own name/port (does NOT touch your live stack)
export DOCKER_HOST=unix:///Users/darren/.docker/run/docker.sock   # this Mac
docker run -d --name foreman-demo-pg \
    -e POSTGRES_USER=demo -e POSTGRES_PASSWORD=demo -e POSTGRES_DB=salesdemo \
    -p 55432:5432 postgres:16-alpine

# 2. seed the sales table (one clearly-worst region in 2026-Q2: West)
export FOREMAN_DEMO_DSN=postgresql://demo:demo@localhost:55432/salesdemo
python demo/seed_sales.py --dsn "$FOREMAN_DEMO_DSN"

# 3. run the full crew (real Anthropic/OpenAI + real Tavily + Docker sandbox)
#    eager Celery (no broker needed); the HITL approval is handled in-script.
CELERY_TASK_ALWAYS_EAGER=true python demo/run_demo.py
```

Teardown is in the last section.

## How to verify it without Docker/Postgres (`verify_local.py`)

`demo/verify_local.py` runs the **same production pipeline** with the real LLM and
real Tavily, but swaps two host-level dependencies for daemon-free equivalents so it
can run in a constrained sandbox:

- sales data in a local **SQLite** file via a SQLite `DatabaseBackend` (same
  `db_query` tool, real SQL, real rows) instead of a Postgres container;
- the code sandbox runs the analyst's stdlib Python in a **subprocess** instead of a
  Docker container (the analyst's code is stdlib-only, so this is equivalent);
- memory is a no-op store.

```bash
python demo/verify_local.py
```

## What actually fired (verified run, gpt-4o)

This was run end-to-end with `verify_local.py` and the real OpenAI provider
(Anthropic was out of API credits at the time; either provider works):

```
[HITL] PAUSED: trigger=sensitive  level=approve_action
       reason: task is flagged as a sensitive operation
       PLAN:
         - subtask1 [analyst]    deps=[]          : identify worst region in 2026-Q2 by revenue
         - subtask2 [researcher] deps=[subtask1]  : research causes for that region
         - subtask3 [writer]     deps=[subtask2]  : draft the action plan
[HITL] operator APPROVES the sensitive action plan

TOOL CALLS
  db_query   by analyst  ok  sql="WITH quarterly_sales AS (SELECT region, SUM(revenue) ...
                              ORDER BY total_revenue ASC LIMIT 1"
      -> rows: [{"region": "West"}]
  code_execution by analyst  ok
  web_search by researcher   ok

SHOWCASE FEATURE CHECK
  db_query against real data          : YES   (analyst CTE correctly flagged West)
  dependency wave (research<-analysis): YES   (researcher depends on analyst output)
  HITL approval gate fired            : YES   (sensitive -> approve_action -> approved)
  reviewer retry fired                : NO    (all sections passed first pass at ~0.90)
```

### On the reviewer retry

The retry path (a failing review bounces a subtask back to `execute`, up to
`MAX_ATTEMPTS=2`) did **not** fire in the live runs: a capable model produced sections
the reviewer scored ~0.90, comfortably above the `0.5` pass threshold. This is the
honest outcome, not a failure of the mechanism. The retry path is real and is proved
deterministically by `tests/e2e/test_showcase.py::test_showcase_runs_end_to_end`,
which fails the writer's review exactly once and asserts only that subtask re-runs. To
see it fire against the live crew you would either raise the reviewer's pass threshold
or hand it a deliberately under-specified subtask; we did not do that here because it
would be tuning the demo to manufacture the event rather than letting it occur.

## Teardown (undo everything)

```bash
# drop the seeded table (Postgres path)
python demo/seed_sales.py --dsn "$FOREMAN_DEMO_DSN" --teardown
# remove the throwaway container
docker rm -f foreman-demo-pg
# the SQLite verify path writes only to a temp dir, which the OS reclaims
```

Nothing in this demo touches the live `foreman` stack (its Postgres is
`foreman-postgres-1`; the throwaway is `foreman-demo-pg` on port 55432). The
worktree's local `.env` is gitignored. All code changes are confined to the worktree.

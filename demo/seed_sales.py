#!/usr/bin/env python3
"""Seed (and tear down) the demo `sales` table for the data-diagnosis showcase.

The flagship demo asks the crew to "find the worst-performing region this quarter".
For that to be a *real* analysis rather than a hallucination, the analyst's
`db_query` tool needs an actual table to read. This script creates a small,
deterministic `sales` table with one clearly-worst region in the current quarter,
so the analyst's SQL lands on an unambiguous answer.

Everything here is reversible: `--teardown` drops the table, and the recommended
way to run the demo is against a *throwaway* Postgres (see DEMO.md), so a full
teardown is just removing that container — nothing of the user's is touched.

Usage:
    python demo/seed_sales.py --dsn postgresql://demo:demo@localhost:55432/salesdemo
    python demo/seed_sales.py --dsn ... --teardown

The DSN can also come from $FOREMAN_DEMO_DSN.
"""

from __future__ import annotations

import argparse
import os
import sys

# (region, quarter, revenue, units, returns) — 4 regions x 2 quarters.
# Q2-2026 is "this quarter"; West is the clear laggard: revenue collapses from
# 980k to 410k, units fall, returns spike. Every other region holds or grows, so
# the worst-performer query has exactly one defensible answer.
_ROWS: list[tuple[str, str, int, int, int]] = [
    ("North", "2026-Q1", 920_000, 4_100, 95),
    ("South", "2026-Q1", 760_000, 3_500, 80),
    ("East", "2026-Q1", 880_000, 3_900, 88),
    ("West", "2026-Q1", 980_000, 4_300, 90),
    ("North", "2026-Q2", 965_000, 4_250, 92),
    ("South", "2026-Q2", 805_000, 3_650, 78),
    ("East", "2026-Q2", 905_000, 4_000, 85),
    ("West", "2026-Q2", 410_000, 1_850, 240),  # <- the worst region this quarter
]

_CREATE = """
CREATE TABLE IF NOT EXISTS sales (
    id       SERIAL PRIMARY KEY,
    region   TEXT    NOT NULL,
    quarter  TEXT    NOT NULL,
    revenue  INTEGER NOT NULL,
    units    INTEGER NOT NULL,
    returns  INTEGER NOT NULL
);
"""


def _connect(dsn: str):  # type: ignore[no-untyped-def]
    import psycopg

    return psycopg.connect(dsn, autocommit=True)


def seed(dsn: str) -> None:
    with _connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(_CREATE)
        cur.execute("DELETE FROM sales")  # idempotent re-seed
        cur.executemany(
            "INSERT INTO sales (region, quarter, revenue, units, returns) "
            "VALUES (%s, %s, %s, %s, %s)",
            _ROWS,
        )
        cur.execute("SELECT count(*) FROM sales")
        n = cur.fetchone()[0]  # type: ignore[index]
    print(f"seeded {n} rows into sales ({dsn})")


def teardown(dsn: str) -> None:
    with _connect(dsn) as conn, conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS sales")
    print(f"dropped sales table ({dsn})")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dsn", default=os.environ.get("FOREMAN_DEMO_DSN"))
    parser.add_argument("--teardown", action="store_true", help="drop the sales table")
    args = parser.parse_args(argv)
    if not args.dsn:
        print("error: pass --dsn or set FOREMAN_DEMO_DSN", file=sys.stderr)
        return 2
    if args.teardown:
        teardown(args.dsn)
    else:
        seed(args.dsn)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

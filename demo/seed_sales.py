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

# (category, quarter, revenue, units, returns) — a consumer-electronics retailer,
# 4 real product categories x 2 quarters. Real categories (not synthetic regions) so
# the "research the causes" step has a genuine market story to find: DVD & Blu-ray
# players are the clear laggard this quarter — revenue collapses 480k -> 180k, units
# crater, returns spike — which maps to the real, researchable secular decline of
# physical media as streaming takes over. Every other category holds or grows, so the
# worst-performer query has exactly one defensible answer.
_ROWS: list[tuple[str, str, int, int, int]] = [
    ("Smartphones", "2026-Q1", 920_000, 4_100, 95),
    ("Laptops", "2026-Q1", 760_000, 3_500, 80),
    ("Streaming Devices", "2026-Q1", 540_000, 6_200, 70),
    ("DVD & Blu-ray Players", "2026-Q1", 480_000, 2_200, 60),
    ("Smartphones", "2026-Q2", 965_000, 4_250, 92),
    ("Laptops", "2026-Q2", 805_000, 3_650, 78),
    ("Streaming Devices", "2026-Q2", 690_000, 7_900, 75),
    ("DVD & Blu-ray Players", "2026-Q2", 180_000, 820, 210),  # <- the worst category
]

# Drop-then-create so a re-seed picks up schema changes (not just new rows) — a plain
# CREATE IF NOT EXISTS would silently keep an older table's columns.
_CREATE = """
DROP TABLE IF EXISTS sales;
CREATE TABLE sales (
    id        SERIAL PRIMARY KEY,
    category  TEXT    NOT NULL,
    quarter   TEXT    NOT NULL,
    revenue   INTEGER NOT NULL,
    units     INTEGER NOT NULL,
    returns   INTEGER NOT NULL
);
"""


def _connect(dsn: str):  # type: ignore[no-untyped-def]
    import psycopg

    return psycopg.connect(dsn, autocommit=True)


def seed(dsn: str) -> None:
    with _connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(_CREATE)  # drops + recreates, so re-seeds are deterministic
        cur.executemany(
            "INSERT INTO sales (category, quarter, revenue, units, returns) "
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

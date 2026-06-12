"""Command-line entrypoint: `foreman run "<task>"`.

Phase 1 wires the stub pipeline so the skeleton is runnable. T5 replaces this
with a real entrypoint once the agents do real work.
"""

from __future__ import annotations

import argparse
import sys

from foreman.config import Settings
from foreman.graph import run_task
from foreman.llm.base import LLMProvider, T
from foreman.llm.router import select_provider
from foreman.schemas import Task


class _NullProvider(LLMProvider):
    """Stand-in used when no API key is configured. Phase-1 nodes never call the
    LLM, so the skeleton stays runnable offline; this is removed once real agents
    land and a provider becomes mandatory."""

    name = "null"

    def structured_complete(self, prompt: str, schema: type[T]) -> T:
        raise RuntimeError("no LLM provider configured (set an API key in .env)")


def _resolve_provider(settings: Settings) -> LLMProvider:
    try:
        return select_provider(settings)
    except ValueError:
        print("note: no provider key set — running the stub pipeline offline\n", file=sys.stderr)
        return _NullProvider()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="foreman")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run a task through the agent pipeline")
    run.add_argument("task", help="the task description")
    args = parser.parse_args(argv)

    if args.command == "run":
        settings = Settings()
        provider = _resolve_provider(settings)
        state = run_task(provider, Task(description=args.task))
        print(state["result"])
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())

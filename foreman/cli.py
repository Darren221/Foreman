"""Command-line entrypoint: `foreman run "<task>"`.

From T2 on, the pipeline does real LLM work, so a provider is required; the CLI
exits cleanly with guidance if none is configured.
"""

from __future__ import annotations

import argparse
import sys

from foreman.config import Settings
from foreman.graph import run_task
from foreman.schemas import Task


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="foreman")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run a task through the agent pipeline")
    run.add_argument("task", help="the task description")
    args = parser.parse_args(argv)

    if args.command == "run":
        settings = Settings()
        try:
            from foreman.llm import select_provider

            provider = select_provider(settings)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            print("set a provider key in .env (see .env.example)", file=sys.stderr)
            return 2

        state = run_task(provider, Task(description=args.task))
        print(state["result"])
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())

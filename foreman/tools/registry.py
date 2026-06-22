"""The tool registry: the single mediated entry point for every tool call.

Routing all invocations through here gives two things for free:
- **Access control** — a specialist can only call tools whose allow-list includes
  it.
- **Observability** — every call is timed and logged (inputs, output, latency,
  success/failure), which Phase 4 builds tracing on top of.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Any

from foreman.observability import NoOpTracer, Tracer
from foreman.schemas import Specialist
from foreman.tools.base import Tool

# The invocation log is observability, not unbounded history: keep only the most
# recent calls so a long-lived registry can't leak memory.
_DEFAULT_MAX_INVOCATIONS = 1000


@dataclass
class ToolInvocation:
    tool: str
    caller: Specialist
    inputs: dict[str, Any]
    output: dict[str, Any] | None
    latency_ms: float
    success: bool
    error: str | None = None


class ToolRegistry:
    def __init__(self, max_invocations: int = _DEFAULT_MAX_INVOCATIONS) -> None:
        self._tools: dict[str, Tool] = {}
        self.invocations: deque[ToolInvocation] = deque(maxlen=max_invocations)
        # Set by build_graph for a traced run; no-op otherwise.
        self.tracer: Tracer = NoOpTracer()

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def has(self, name: str) -> bool:
        """Whether a tool is registered. Lets an agent adapt to an optional
        capability (e.g. the analyst querying a DB only when one is wired)."""
        return name in self._tools

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise ValueError(f"unknown tool {name!r}")
        return self._tools[name]

    def invoke(self, name: str, caller: Specialist, **inputs: Any) -> dict[str, Any]:
        tool = self.get(name)  # raises ValueError for an unknown tool
        if caller not in tool.allowed_specialists:
            raise ValueError(f"{caller.value} is not permitted to call tool '{name}'")

        span_attrs = {"foreman.caller": caller.value}
        with self.tracer.span(f"tool:{name}", kind="tool", attributes=span_attrs):
            started = time.perf_counter()
            try:
                output = tool.run(**inputs)
            except Exception as exc:
                self._log(name, caller, inputs, None, started, success=False, error=str(exc))
                raise
            self._log(name, caller, inputs, output, started, success=True)
            return output

    def _log(
        self,
        name: str,
        caller: Specialist,
        inputs: dict[str, Any],
        output: dict[str, Any] | None,
        started: float,
        *,
        success: bool,
        error: str | None = None,
    ) -> None:
        self.invocations.append(
            ToolInvocation(
                tool=name,
                caller=caller,
                inputs=inputs,
                output=output,
                latency_ms=(time.perf_counter() - started) * 1000,
                success=success,
                error=error,
            )
        )

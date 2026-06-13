"""The tool registry: the single mediated entry point for every tool call.

Routing all invocations through here gives two things for free:
- **Access control** — a specialist can only call tools whose allow-list includes
  it.
- **Observability** — every call is timed and logged (inputs, output, latency,
  success/failure), which Phase 4 builds tracing on top of.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from foreman.observability import NoOpTracer, Tracer
from foreman.schemas import Specialist
from foreman.tools.base import Tool


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
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        self.invocations: list[ToolInvocation] = []
        # Set by build_graph for a traced run; no-op otherwise.
        self.tracer: Tracer = NoOpTracer()

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        return self._tools[name]

    def invoke(self, name: str, caller: Specialist, **inputs: Any) -> dict[str, Any]:
        tool = self._tools[name]  # KeyError for unknown tool — caller's bug
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

"""The specialist worker task: run one subtask on a worker and return its output.

A worker is potentially a separate process, so the task crosses the boundary as
JSON — the subtask, its dependencies' outputs (`upstream`), reviewer feedback, and
the trace context — and returns the `SpecialistOutput` as JSON. JSON in/out means
eager mode and real workers behave identically.

How a worker gets its provider/registry/tracer depends on where it runs:

- **In-process** (a single-process run, or eager-mode tests): the orchestrator sets
  a `WorkerContext` first, and the task reuses the app's already-built services and
  the live trace context. No rebuild, no remote trace propagation — and tests can
  inject fakes simply by being the ones that built them.
- **A real remote worker** (no context): it rebuilds provider/registry from
  `Settings` and re-activates the propagated `traceparent`, so its spans nest under
  the originating run. `_build_provider_and_registry` is the seam that rebuild uses.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from celery import shared_task

from foreman.agents import Analyst, Researcher, Writer
from foreman.agents.base import SpecialistAgent
from foreman.config import Settings
from foreman.llm.base import LLMProvider
from foreman.observability.instrument import TracingProvider
from foreman.observability.tracer import NoOpTracer, Tracer
from foreman.schemas import Specialist, SpecialistOutput, Subtask
from foreman.tools import ToolRegistry

_AGENTS: dict[Specialist, Callable[[ToolRegistry, LLMProvider], SpecialistAgent]] = {
    Specialist.RESEARCHER: Researcher,
    Specialist.ANALYST: Analyst,
    Specialist.WRITER: Writer,
}


@dataclass
class WorkerContext:
    """The in-process services a worker reuses instead of rebuilding from config."""

    provider: LLMProvider
    registry: ToolRegistry
    tracer: Tracer


_CONTEXT: WorkerContext | None = None


@contextmanager
def use_worker_context(
    provider: LLMProvider, registry: ToolRegistry, tracer: Tracer
) -> Iterator[None]:
    """Register in-process services for the duration of a dispatch. Eager tasks run
    synchronously inside this block, so they pick the context up; real workers run in
    other processes where it's never set, so they rebuild from config instead."""
    global _CONTEXT
    previous = _CONTEXT
    _CONTEXT = WorkerContext(provider, registry, tracer)
    try:
        yield
    finally:
        _CONTEXT = previous


def _build_provider_and_registry(settings: Settings) -> tuple[LLMProvider, ToolRegistry]:
    """Rebuild a remote worker's provider + registry from config. A worker otherwise
    constructs real, keyed services; tests don't reach this (they set a context)."""
    from foreman.llm import select_provider
    from foreman.tools import build_default_registry

    return select_provider(settings), build_default_registry(settings)


def _remote_tracer(settings: Settings, traceparent: str | None) -> tuple[Tracer, Any]:
    """A remote worker's tracer + an activation token for the propagated context, or
    NoOp + None. When traced, it builds its own `OTelTracer` over the *same* trace
    store (shared file / DB) and activates the injected W3C context, so its spans
    nest under the run rather than starting a fresh trace."""
    if not traceparent:
        return NoOpTracer(), None
    from opentelemetry.context import attach
    from opentelemetry.propagate import extract

    from foreman.observability import OTelTracer
    from foreman.storage.factory import build_trace_store

    tracer = OTelTracer(build_trace_store(settings))
    token = attach(extract({"traceparent": traceparent}))
    return tracer, token


@shared_task(name="foreman.run_specialist")  # type: ignore[untyped-decorator]  # celery is untyped
def run_specialist(
    subtask_json: str,
    feedback: str | None = None,
    upstream_json: list[str] | None = None,
    traceparent: str | None = None,
) -> str:
    subtask = Subtask.model_validate_json(subtask_json)
    upstream = [SpecialistOutput.model_validate_json(u) for u in (upstream_json or [])]

    ctx = _CONTEXT
    token = None
    if ctx is not None:
        provider, registry, tracer = ctx.provider, ctx.registry, ctx.tracer
    else:
        settings = Settings()
        provider, registry = _build_provider_and_registry(settings)
        tracer, token = _remote_tracer(settings, traceparent)

    registry.tracer = tracer
    agent = _AGENTS[subtask.assigned_specialist](registry, TracingProvider(provider, tracer))
    try:
        with tracer.span(f"specialist:{subtask.assigned_specialist.value}", kind="node"):
            output = agent.execute(subtask, feedback=feedback, upstream=upstream)
    finally:
        if token is not None:
            from opentelemetry.context import detach

            detach(token)
    return output.model_dump_json()

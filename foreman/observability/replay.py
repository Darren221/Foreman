"""Counterfactual forking: re-run a finished run from a chosen node with an edited
input, and compare how it diverges.

This is the brief's "replay" — and it reuses the *same* checkpointer primitive
that powers human-in-the-loop. The checkpointer already stores a snapshot before
each node; forking is just: take the snapshot before node N, write an edit onto
it (`update_state` creates a branch), and resume (`invoke(None, ...)`). The fork
runs under a fresh trace so the two runs sit side by side in the store.

It is non-deterministic by design: with real models the same edit can produce
different output run to run — that's the point of asking "what if", not a bug to
engineer away (temperature=0 is not determinism).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from foreman.observability.store import SpanNode, TraceStore
from foreman.observability.tracer import Tracer


@dataclass
class ForkResult:
    """The outcome of a fork: where it diverged, its trace, and its final result."""

    fork_trace_id: str
    forked_node: str
    result: str | None


@dataclass
class RunComparison:
    """Which nodes the fork inherited from the original (the shared prefix) versus
    which it re-executed (the diverged tail)."""

    forked_node: str
    inherited: list[str] = field(default_factory=list)
    replayed: list[str] = field(default_factory=list)


def fork_run(
    graph: Any,
    thread_id: str,
    at_node: str,
    edits: dict[str, Any],
    tracer: Tracer,
) -> ForkResult:
    """Re-run `thread_id` from `at_node` with `edits` applied to the input state.

    `graph` must be the checkpointed graph the original run used (same tracer).
    """
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = _snapshot_before(graph, config, at_node)
    fork_config = graph.update_state(snapshot.config, edits)

    with tracer.start_run(f"{thread_id}-fork", f"fork@{at_node}") as span:
        final = graph.invoke(None, fork_config)
        trace_id = _trace_id(span)

    result = final.get("result") if isinstance(final, dict) else None
    return ForkResult(fork_trace_id=trace_id, forked_node=at_node, result=result)


def compare_runs(
    store: TraceStore, original_trace_id: str, fork_trace_id: str, forked_node: str
) -> RunComparison:
    """Split the original's nodes at the fork point: those before it were inherited
    by the fork; the fork's own nodes are what got re-executed."""
    original_nodes = _node_names(store.get_trace(original_trace_id))
    fork_nodes = _node_names(store.get_trace(fork_trace_id))

    target = f"node:{forked_node}"
    cut = original_nodes.index(target) if target in original_nodes else len(original_nodes)
    return RunComparison(
        forked_node=forked_node,
        inherited=original_nodes[:cut],
        replayed=fork_nodes,
    )


def _snapshot_before(graph: Any, config: dict[str, Any], at_node: str) -> Any:
    for snapshot in graph.get_state_history(config):
        if at_node in (snapshot.next or ()):
            return snapshot
    raise ValueError(f"node {at_node!r} did not run in this thread's history")


def _trace_id(span: Any) -> str:
    if span is None:
        return ""
    return f"{span.get_span_context().trace_id:032x}"


def _node_names(root: SpanNode | None) -> list[str]:
    if root is None:
        return []
    names = [root.span.name] if root.span.kind == "node" else []
    for child in root.children:
        names.extend(_node_names(child))
    return names

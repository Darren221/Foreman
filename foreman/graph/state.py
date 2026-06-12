"""The shared state object that flows through the graph.

Each node receives the current state and returns a partial update; LangGraph
merges updates by key. Keeping state a typed dict (rather than loose kwargs)
means every node reads and writes a known contract.
"""

from __future__ import annotations

from typing import TypedDict

from foreman.schemas import Plan, ReviewResult, SpecialistOutput, Task


class GraphState(TypedDict, total=False):
    task: Task
    plan: Plan | None
    outputs: list[SpecialistOutput]
    review: ReviewResult | None
    attempts: int
    result: str | None

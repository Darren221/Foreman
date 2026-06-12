"""Core data contracts that flow between agents and through the graph.

These models are the typed boundary of the system: the supervisor emits a `Plan`,
a specialist returns a `SpecialistOutput`, the reviewer returns a `ReviewResult`.
Keeping them strict means a malformed agent decision fails loudly at the edge
rather than corrupting state downstream.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field, model_validator


class Specialist(StrEnum):
    """The specialist agents a subtask can be assigned to.

    Only RESEARCHER is wired in Phase 1; the others are reserved so plans and
    routing don't need reshaping when the full crew lands.
    """

    RESEARCHER = "researcher"
    ANALYST = "analyst"
    WRITER = "writer"


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


class Task(BaseModel):
    """A complex request submitted to the system."""

    id: str = Field(default_factory=_new_id)
    description: str
    require_approval: bool = False
    sensitive: bool = False


class Subtask(BaseModel):
    """One unit of work in a plan, assigned to a single specialist."""

    id: str
    description: str
    assigned_specialist: Specialist
    required_inputs: list[str] = Field(default_factory=list)
    expected_output: str
    complexity: int = Field(ge=1, le=5)
    dependencies: list[str] = Field(default_factory=list)


class Plan(BaseModel):
    """An ordered, dependency-aware decomposition of a task into subtasks."""

    task_id: str
    subtasks: list[Subtask]

    @model_validator(mode="after")
    def _check_dependencies(self) -> Plan:
        ids = {s.id for s in self.subtasks}
        if len(ids) != len(self.subtasks):
            raise ValueError("subtask ids must be unique")
        for s in self.subtasks:
            unknown = [d for d in s.dependencies if d not in ids]
            if unknown:
                raise ValueError(f"subtask {s.id} depends on unknown id(s): {unknown}")
        self._reject_cycles()
        return self

    def _reject_cycles(self) -> None:
        graph = {s.id: s.dependencies for s in self.subtasks}
        # DFS with three colours: detect a back-edge into the current stack.
        WHITE, GREY, BLACK = 0, 1, 2
        colour = dict.fromkeys(graph, WHITE)

        def visit(node: str) -> None:
            colour[node] = GREY
            for dep in graph[node]:
                if colour[dep] == GREY:
                    raise ValueError("plan dependencies contain a cycle")
                if colour[dep] == WHITE:
                    visit(dep)
            colour[node] = BLACK

        for node in graph:
            if colour[node] == WHITE:
                visit(node)


class SpecialistOutput(BaseModel):
    """The result a specialist returns for a subtask."""

    subtask_id: str
    content: str
    tools_used: list[str] = Field(default_factory=list)


class ReviewResult(BaseModel):
    """The reviewer's verdict on a specialist output."""

    passed: bool
    score: float = Field(ge=0.0, le=1.0)
    feedback: str = ""


class ResearchFindings(BaseModel):
    """A specialist's findings, written up from raw tool output."""

    content: str


class Synthesis(BaseModel):
    """The supervisor's final deliverable, composed from specialist outputs."""

    result: str


class TaskMemory(BaseModel):
    """A distilled record of a finished task, stored for future recall.

    `task_description` is the semantic key (what gets embedded); the rest captures
    what happened so a future, similar task can learn from it.
    """

    id: str = Field(default_factory=_new_id)
    task_description: str
    outcome: str
    score: float = Field(ge=0.0, le=1.0)
    tools_used: list[str] = Field(default_factory=list)
    result_snippet: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

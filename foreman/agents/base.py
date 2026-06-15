"""The shape every specialist agent shares.

A `Protocol`, not a base class — the researcher, analyst, and writer satisfy it
structurally without inheriting, which keeps each agent independent while letting
the graph hold them in one typed `{specialist: agent}` dispatch map.
"""

from __future__ import annotations

from typing import Protocol

from foreman.schemas import Specialist, SpecialistOutput, Subtask


class SpecialistAgent(Protocol):
    specialist: Specialist

    def execute(
        self,
        subtask: Subtask,
        feedback: str | None = None,
        upstream: list[SpecialistOutput] | None = None,
    ) -> SpecialistOutput: ...


def render_upstream(upstream: list[SpecialistOutput] | None) -> str:
    """Render a subtask's dependency outputs as prompt context — its inputs.

    C-orchestrated waves feed each subtask the outputs of the subtasks it depends
    on; this turns them into a labelled block for the prompt. Empty -> "none"."""
    if not upstream:
        return "none"
    return "\n\n".join(
        f"[{o.produced_by.value if o.produced_by else 'unknown'}] {o.content}" for o in upstream
    )

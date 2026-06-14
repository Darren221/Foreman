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

    def execute(self, subtask: Subtask, feedback: str | None = None) -> SpecialistOutput: ...

"""The analyst specialist: process and interpret data for a subtask.

In this first cut it reasons over the subtask with the LLM. Its real power —
sandboxed code execution and database queries — is wired in Phase 5 C2; the
registry is held now so those tools slot in without changing the constructor.
"""

from __future__ import annotations

from foreman.llm.base import LLMProvider
from foreman.schemas import ResearchFindings, Specialist, SpecialistOutput, Subtask
from foreman.tools import ToolRegistry

_PROMPT = """\
You are a data analyst. Analyse the subtask and produce clear, specific findings:
extract the figures, relationships, and conclusions the task asks for. State any
assumptions; do not invent data.

Subtask: {description}
Expected output: {expected_output}

Reviewer feedback to address (if any): {feedback}
"""


class Analyst:
    specialist = Specialist.ANALYST

    def __init__(self, registry: ToolRegistry, provider: LLMProvider) -> None:
        self._registry = registry
        self._provider = provider

    def execute(self, subtask: Subtask, feedback: str | None = None) -> SpecialistOutput:
        prompt = _PROMPT.format(
            description=subtask.description,
            expected_output=subtask.expected_output,
            feedback=feedback or "none",
        )
        content = self._provider.structured_complete(prompt, ResearchFindings).content
        return SpecialistOutput(subtask_id=subtask.id, content=content, produced_by=self.specialist)

"""The writer specialist: turn a subtask (and any feedback) into polished prose.

LLM-only for now. The file I/O tool is available in the registry (for persisting or
passing artifacts); the registry is held so wiring it in is a constructor-free change.
"""

from __future__ import annotations

from foreman.llm.base import LLMProvider
from foreman.schemas import ResearchFindings, Specialist, SpecialistOutput, Subtask
from foreman.tools import ToolRegistry

_PROMPT = """\
You are a writer. Produce a clear, well-structured written draft that satisfies the
subtask. Match the expected output; be concise and concrete; do not invent facts
beyond what the subtask provides.

Subtask: {description}
Expected output: {expected_output}

Reviewer feedback to address (if any): {feedback}
"""


class Writer:
    specialist = Specialist.WRITER

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

"""The analyst specialist: compute the answer by running code, then write it up.

It mirrors the researcher's two-step shape — gather, then summarise — but its
"gather" is *running code in the sandbox*: the LLM writes a short program, the
`code_execution` tool runs it in a throwaway container, and the LLM writes up the
findings grounded in the program's output (folding in any reviewer feedback).
"""

from __future__ import annotations

from foreman.llm.base import LLMProvider
from foreman.schemas import AnalysisCode, ResearchFindings, Specialist, SpecialistOutput, Subtask
from foreman.tools import ToolRegistry

_TOOL = "code_execution"

_CODE_PROMPT = """\
You are a data analyst. Write a short Python program that computes what the subtask
asks for and prints the result. Use only the standard library; print the answer.

Subtask: {description}
Expected output: {expected_output}

Reviewer feedback to address (if any): {feedback}
"""

_WRITEUP_PROMPT = """\
Write up the analysis findings for the subtask, grounded in the program output
below. State the result and what it means; be specific; do not invent numbers.

Subtask: {description}

Program output:
{output}

Reviewer feedback to address (if any): {feedback}
"""


class Analyst:
    specialist = Specialist.ANALYST

    def __init__(self, registry: ToolRegistry, provider: LLMProvider) -> None:
        self._registry = registry
        self._provider = provider

    def execute(self, subtask: Subtask, feedback: str | None = None) -> SpecialistOutput:
        code = self._provider.structured_complete(
            _CODE_PROMPT.format(
                description=subtask.description,
                expected_output=subtask.expected_output,
                feedback=feedback or "none",
            ),
            AnalysisCode,
        ).code
        result = self._registry.invoke(_TOOL, self.specialist, code=code)
        output = result.get("stdout") or result.get("stderr") or "(no output)"
        content = self._provider.structured_complete(
            _WRITEUP_PROMPT.format(
                description=subtask.description, output=output, feedback=feedback or "none"
            ),
            ResearchFindings,
        ).content
        return SpecialistOutput(
            subtask_id=subtask.id, content=content, tools_used=[_TOOL], produced_by=self.specialist
        )

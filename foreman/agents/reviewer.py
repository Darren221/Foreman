"""The reviewer agent: validate a specialist's output before it's accepted.

It judges an output against the subtask that produced it and returns a structured
verdict (pass/fail, score, feedback). A failed verdict drives the graph's
retry edge, so the feedback is what the specialist gets on its next attempt.
"""

from __future__ import annotations

from foreman.llm.base import LLMProvider
from foreman.schemas import ReviewResult, SpecialistOutput, Subtask

_PROMPT = """\
You are a strict reviewer. Judge whether the output satisfies the subtask.

Subtask: {description}
Expected output: {expected_output}

Output to review:
{content}

Return: passed (true/false), a score from 0 to 1, and concise feedback. If it
fails, the feedback must say what to fix.
"""


class Reviewer:
    def __init__(self, provider: LLMProvider) -> None:
        self._provider = provider

    def review(self, subtask: Subtask, output: SpecialistOutput) -> ReviewResult:
        prompt = _PROMPT.format(
            description=subtask.description,
            expected_output=subtask.expected_output,
            content=output.content,
        )
        return self._provider.structured_complete(prompt, ReviewResult)

"""The reviewer agent: validate a specialist's output before it's accepted.

The reviewer is split into two responsibilities on purpose:
- the model *assesses* quality (a 0-1 score plus feedback), guided by an explicit
  rubric so its scores are calibrated rather than reflexively harsh;
- our code *decides* pass/fail by comparing the score to `pass_threshold`.

Keeping the decision in code (not in the model's `passed` flag) makes the pass
bar an explicit, testable knob: tune the threshold to move the bar without
re-prompting. A miscalibrated gate that rejects work the specialist can't improve
just burns retries — see scratch/learning/concepts/05.
"""

from __future__ import annotations

from foreman.llm.base import LLMProvider
from foreman.schemas import ReviewResult, SpecialistOutput, Subtask

# Set to accept "usable" output (>=0.5 on the rubric) and reject only genuinely
# bad output (off-topic / inaccurate / empty). The bar is deliberately at the
# bottom of the usable band because the researcher can't yet act on feedback to
# improve on a retry — so a higher bar would only burn retries for no gain. Raise
# this once specialists can meaningfully respond to feedback.
DEFAULT_PASS_THRESHOLD = 0.5

_PROMPT = """\
You are a fair, pragmatic reviewer. Score how well the output satisfies the
subtask. Judge it as the result of a single research pass, not an exhaustive
report — pass good-enough work and only penalise substantive problems.

Subtask: {description}
Expected output: {expected_output}

Output to review:
{content}

Scoring guide (0.0-1.0):
- 0.8-1.0: accurate and addresses the subtask well; minor gaps are acceptable.
- 0.5-0.7: usable, but with a real, fixable shortcoming.
- 0.0-0.4: off-topic, inaccurate, empty, or missing the core ask.

Return a score, and concise feedback. If the score is low, the feedback must name
one specific, fixable gap the researcher can act on next time.
"""


class Reviewer:
    def __init__(
        self, provider: LLMProvider, pass_threshold: float = DEFAULT_PASS_THRESHOLD
    ) -> None:
        self._provider = provider
        self._pass_threshold = pass_threshold

    def review(self, subtask: Subtask, output: SpecialistOutput) -> ReviewResult:
        prompt = _PROMPT.format(
            description=subtask.description,
            expected_output=subtask.expected_output,
            content=output.content,
        )
        assessment = self._provider.structured_complete(prompt, ReviewResult)
        # Our policy, not the model's: the threshold owns the pass/fail decision.
        return ReviewResult(
            passed=assessment.score >= self._pass_threshold,
            score=assessment.score,
            feedback=assessment.feedback,
        )

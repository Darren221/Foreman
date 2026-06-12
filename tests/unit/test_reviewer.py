from foreman.agents import Reviewer
from foreman.llm.base import LLMProvider, T
from foreman.schemas import ReviewResult, Specialist, SpecialistOutput, Subtask


class CannedProvider(LLMProvider):
    name = "canned"

    def __init__(self, response: object) -> None:
        self._response = response

    def structured_complete(self, prompt: str, schema: type[T]) -> T:
        return self._response  # type: ignore[return-value]


def _subtask() -> Subtask:
    return Subtask(
        id="s1",
        description="research bicycles",
        assigned_specialist=Specialist.RESEARCHER,
        expected_output="findings",
        complexity=1,
    )


def _output() -> SpecialistOutput:
    return SpecialistOutput(subtask_id="s1", content="Invented in 1817.")


def test_reviewer_passes_good_output() -> None:
    verdict = ReviewResult(passed=True, score=0.9, feedback="solid")
    result = Reviewer(CannedProvider(verdict)).review(_subtask(), _output())
    assert result.passed is True


def test_reviewer_fails_bad_output() -> None:
    verdict = ReviewResult(passed=False, score=0.2, feedback="missing dates")
    result = Reviewer(CannedProvider(verdict)).review(_subtask(), _output())
    assert result.passed is False
    assert result.feedback == "missing dates"

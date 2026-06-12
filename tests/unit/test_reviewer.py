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


def test_pass_decision_comes_from_score_not_model_flag() -> None:
    # Model says passed=False, but a high score must pass: our threshold owns the
    # decision, the model only assesses quality.
    verdict = ReviewResult(passed=False, score=0.9, feedback="solid")
    result = Reviewer(CannedProvider(verdict)).review(_subtask(), _output())
    assert result.passed is True
    assert result.score == 0.9


def test_low_score_fails_even_if_model_says_passed() -> None:
    verdict = ReviewResult(passed=True, score=0.2, feedback="missing dates")
    result = Reviewer(CannedProvider(verdict)).review(_subtask(), _output())
    assert result.passed is False
    assert result.feedback == "missing dates"


def test_pass_threshold_is_configurable() -> None:
    verdict = ReviewResult(passed=True, score=0.8, feedback="ok")
    strict = Reviewer(CannedProvider(verdict), pass_threshold=0.95)
    lenient = Reviewer(CannedProvider(verdict), pass_threshold=0.5)
    assert strict.review(_subtask(), _output()).passed is False
    assert lenient.review(_subtask(), _output()).passed is True

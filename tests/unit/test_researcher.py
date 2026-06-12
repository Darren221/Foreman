from typing import Any

from foreman.agents import Researcher
from foreman.llm.base import LLMProvider, T
from foreman.schemas import ResearchFindings, Specialist, Subtask
from foreman.tools import ToolRegistry, WebSearchTool


class FakeBackend:
    def search(self, query: str, max_results: int) -> list[dict[str, Any]]:
        return [
            {"title": "Bicycle", "url": "http://x", "content": "Invented in 1817."},
            {"title": "Draisine", "url": "http://y", "content": "Early running machine."},
        ]


class RecordingBackend:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def search(self, query: str, max_results: int) -> list[dict[str, Any]]:
        self.queries.append(query)
        return [{"title": "t", "url": "u", "content": "c"}]


class SummaryProvider(LLMProvider):
    """Returns a fixed findings summary, and records the prompts it received."""

    name = "summary"

    def __init__(self, content: str = "The bicycle was invented in 1817.") -> None:
        self._content = content
        self.prompts: list[str] = []

    def structured_complete(self, prompt: str, schema: type[T]) -> T:
        self.prompts.append(prompt)
        return ResearchFindings(content=self._content)  # type: ignore[return-value]


def _registry(backend: Any) -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(WebSearchTool(backend))
    return reg


def _subtask(description: str = "history of the bicycle") -> Subtask:
    return Subtask(
        id="s1",
        description=description,
        assigned_specialist=Specialist.RESEARCHER,
        expected_output="findings",
        complexity=1,
    )


def test_researcher_summarises_search_results() -> None:
    reg = _registry(FakeBackend())
    output = Researcher(reg, SummaryProvider()).execute(_subtask())

    assert output.subtask_id == "s1"
    assert "web_search" in output.tools_used
    assert "1817" in output.content  # the LLM-written summary
    assert reg.invocations[0].tool == "web_search"


def test_feedback_is_passed_to_the_summariser() -> None:
    provider = SummaryProvider()
    Researcher(_registry(FakeBackend()), provider).execute(
        _subtask(), feedback="include exact dates"
    )
    # the reviewer's feedback must reach the summariser so a retry can act on it
    assert any("include exact dates" in p for p in provider.prompts)


def test_search_query_is_bounded() -> None:
    backend = RecordingBackend()
    # a pathologically long description must not produce an over-long query
    Researcher(_registry(backend), SummaryProvider()).execute(_subtask("bike " * 200))

    assert len(backend.queries[0]) <= 400

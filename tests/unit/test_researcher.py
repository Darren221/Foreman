from typing import Any

from foreman.agents import Researcher
from foreman.schemas import Specialist, Subtask
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


def _registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(WebSearchTool(FakeBackend()))
    return reg


def _subtask() -> Subtask:
    return Subtask(
        id="s1",
        description="history of the bicycle",
        assigned_specialist=Specialist.RESEARCHER,
        expected_output="findings",
        complexity=1,
    )


def test_researcher_uses_web_search_and_returns_output() -> None:
    reg = _registry()
    output = Researcher(reg).execute(_subtask())

    assert output.subtask_id == "s1"
    assert "web_search" in output.tools_used
    assert "1817" in output.content  # content drawn from the search results
    assert reg.invocations[0].tool == "web_search"


def test_query_is_bounded_even_with_long_feedback() -> None:
    # Reviewer feedback can be long prose; it must never push the search query
    # past the provider's length limit (Tavily caps at 400 chars).
    backend = RecordingBackend()
    reg = ToolRegistry()
    reg.register(WebSearchTool(backend))

    Researcher(reg).execute(_subtask(), feedback="x" * 1000)

    assert len(backend.queries[0]) <= 400
    assert backend.queries[0].startswith("history of the bicycle")

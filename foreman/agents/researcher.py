"""The researcher specialist: gather information for a subtask via web search.

Phase 1 keeps it deterministic — it calls the web-search tool through the
registry and assembles the findings. (LLM-driven summarisation of the results is
a natural later enhancement; the contract it returns won't change.)
"""

from __future__ import annotations

from foreman.schemas import Specialist, SpecialistOutput, Subtask
from foreman.tools import ToolRegistry

_TOOL = "web_search"
# Tavily (and most search APIs) reject overly long queries; keep within the limit.
_MAX_QUERY_LEN = 400


class Researcher:
    specialist = Specialist.RESEARCHER

    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry

    def execute(self, subtask: Subtask, feedback: str | None = None) -> SpecialistOutput:
        query = self._build_query(subtask.description, feedback)
        result = self._registry.invoke(_TOOL, self.specialist, query=query)
        content = self._summarise(result.get("results", []))
        return SpecialistOutput(
            subtask_id=subtask.id,
            content=content,
            tools_used=[_TOOL],
        )

    @staticmethod
    def _build_query(description: str, feedback: str | None) -> str:
        # The query is the topic to search for. Reviewer feedback is about output
        # quality, not search terms — fold it in only if it still fits the limit,
        # otherwise search the topic alone. Never exceed the provider's cap.
        query = description
        if feedback:
            candidate = f"{description} {feedback}"
            if len(candidate) <= _MAX_QUERY_LEN:
                query = candidate
        return query[:_MAX_QUERY_LEN]

    @staticmethod
    def _summarise(results: list[dict[str, object]]) -> str:
        if not results:
            return "No results found."
        lines = [f"- {r.get('title', 'untitled')}: {r.get('content', '')}" for r in results]
        return "\n".join(lines)

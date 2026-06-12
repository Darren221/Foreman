"""The researcher specialist: gather information for a subtask via web search.

Phase 1 keeps it deterministic — it calls the web-search tool through the
registry and assembles the findings. (LLM-driven summarisation of the results is
a natural later enhancement; the contract it returns won't change.)
"""

from __future__ import annotations

from foreman.schemas import Specialist, SpecialistOutput, Subtask
from foreman.tools import ToolRegistry

_TOOL = "web_search"


class Researcher:
    specialist = Specialist.RESEARCHER

    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry

    def execute(self, subtask: Subtask, feedback: str | None = None) -> SpecialistOutput:
        query = subtask.description
        if feedback:
            # On a retry, fold the reviewer's feedback into the search.
            query = f"{query} ({feedback})"
        result = self._registry.invoke(_TOOL, self.specialist, query=query)
        content = self._summarise(result.get("results", []))
        return SpecialistOutput(
            subtask_id=subtask.id,
            content=content,
            tools_used=[_TOOL],
        )

    @staticmethod
    def _summarise(results: list[dict[str, object]]) -> str:
        if not results:
            return "No results found."
        lines = [f"- {r.get('title', 'untitled')}: {r.get('content', '')}" for r in results]
        return "\n".join(lines)

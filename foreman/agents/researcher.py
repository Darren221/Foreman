"""The researcher specialist: gather information for a subtask and write it up.

It searches the web for the subtask topic, then uses the LLM to turn the raw
results into a coherent written summary grounded in those sources. On a retry it
receives the reviewer's feedback and is told to address it — so a second attempt
can genuinely improve, not just repeat itself.
"""

from __future__ import annotations

from foreman.llm.base import LLMProvider
from foreman.schemas import ResearchFindings, Specialist, SpecialistOutput, Subtask
from foreman.tools import ToolRegistry

_TOOL = "web_search"
# Tavily (and most search APIs) reject overly long queries; keep within the limit.
_MAX_QUERY_LEN = 400

_SUMMARY_PROMPT = """\
Write a clear, well-organised summary that answers the subtask, using only the
search results provided. Be specific — include names, dates, and figures the
sources support. Do not invent facts beyond the sources.

Subtask: {description}

Reviewer feedback to address (if any): {feedback}

Search results:
{sources}
"""


class Researcher:
    specialist = Specialist.RESEARCHER

    def __init__(self, registry: ToolRegistry, provider: LLMProvider) -> None:
        self._registry = registry
        self._provider = provider

    def execute(self, subtask: Subtask, feedback: str | None = None) -> SpecialistOutput:
        # The query is the topic only; reviewer feedback is handled at write-up
        # time, so it can never bloat the query past the provider's limit.
        query = subtask.description[:_MAX_QUERY_LEN]
        result = self._registry.invoke(_TOOL, self.specialist, query=query)
        content = self._summarise(subtask, result.get("results", []), feedback)
        return SpecialistOutput(subtask_id=subtask.id, content=content, tools_used=[_TOOL])

    def _summarise(
        self, subtask: Subtask, results: list[dict[str, object]], feedback: str | None
    ) -> str:
        if not results:
            return "No results found."
        sources = "\n".join(
            f"- {r.get('title', 'untitled')}: {r.get('content', '')}" for r in results
        )
        prompt = _SUMMARY_PROMPT.format(
            description=subtask.description,
            feedback=feedback or "none",
            sources=sources,
        )
        return self._provider.structured_complete(prompt, ResearchFindings).content

"""Web search tool, backed by Tavily.

The tool depends on a `SearchBackend` interface, not on Tavily directly, so the
live provider is swappable and tests inject a fake backend (no network, no key).
"""

from __future__ import annotations

from typing import Any, Protocol

from foreman.schemas import Specialist
from foreman.tools.base import Tool


class SearchBackend(Protocol):
    def search(self, query: str, max_results: int) -> list[dict[str, Any]]:
        """Return a list of result dicts (each with at least title, url, content)."""
        ...


class TavilyBackend:
    """Lazily-constructed Tavily client. No key or network needed until `search`."""

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key
        self._client: Any = None

    def search(self, query: str, max_results: int) -> list[dict[str, Any]]:
        if self._client is None:
            from tavily import TavilyClient

            self._client = TavilyClient(api_key=self._api_key)
        response = self._client.search(query=query, max_results=max_results)
        results: list[dict[str, Any]] = response.get("results", [])
        return results


class WebSearchTool(Tool):
    name = "web_search"
    description = "Search the web for information on a query."
    allowed_specialists = frozenset({Specialist.RESEARCHER})
    rate_limit_per_min = 60

    def __init__(self, backend: SearchBackend) -> None:
        self._backend = backend

    def run(self, **inputs: Any) -> dict[str, Any]:
        query = inputs["query"]
        max_results = inputs.get("max_results", 5)
        results = self._backend.search(query=query, max_results=max_results)
        return {"results": results}

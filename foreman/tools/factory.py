"""Build the default tool registry from configuration."""

from __future__ import annotations

from foreman.config import Settings
from foreman.tools.registry import ToolRegistry
from foreman.tools.web_search import TavilyBackend, WebSearchTool


def build_default_registry(settings: Settings) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(WebSearchTool(TavilyBackend(settings.tavily_api_key or "")))
    return registry

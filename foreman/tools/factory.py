"""Build the default tool registry from configuration."""

from __future__ import annotations

from foreman.config import Settings
from foreman.tools.code_exec import CodeExecutionTool, DockerSandbox
from foreman.tools.database import DatabaseQueryTool, PostgresBackend
from foreman.tools.files import FileTool
from foreman.tools.registry import ToolRegistry
from foreman.tools.web_search import TavilyBackend, WebSearchTool


def build_default_registry(settings: Settings) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(WebSearchTool(TavilyBackend(settings.tavily_api_key or "")))
    registry.register(CodeExecutionTool(DockerSandbox()))
    registry.register(FileTool(settings.workspace_path))
    registry.register(DatabaseQueryTool(PostgresBackend(settings.database_dsn or "")))
    return registry

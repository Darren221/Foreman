"""Build the default tool registry from configuration."""

from __future__ import annotations

from foreman.config import Settings
from foreman.tools.api_call import ApiCallTool, UrllibBackend
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
    # `db_query` is registered only when a data source is configured. With no DSN
    # the analyst would have nothing to query, so leaving the tool unregistered
    # keeps the analyst on its code-only path (and `registry.has("db_query")` is the
    # honest signal of whether live data is available) rather than registering a
    # tool whose every call fails. The analyst DB is separate from the operational
    # store; it falls back to `database_dsn` for a single-database deployment.
    dsn = settings.analyst_database_dsn or settings.database_dsn
    if dsn:
        registry.register(DatabaseQueryTool(PostgresBackend(dsn)))
    registry.register(ApiCallTool(UrllibBackend()))
    return registry

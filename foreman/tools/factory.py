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
    # `db_query` is registered only when an *analytics* data source is explicitly
    # configured (`analyst_database_dsn`). Without it the analyst stays on its
    # code-only path, and `registry.has("db_query")` is the honest "is live data
    # available" signal. We deliberately do NOT fall back to `database_dsn`: that's
    # Foreman's operational store (checkpoints, approvals, traces), not analytics
    # data — pointing LLM-authored SQL at the control plane would be the wrong data
    # and a needless exposure. The analyst's data source is an intentional choice.
    if settings.analyst_database_dsn:
        registry.register(DatabaseQueryTool(PostgresBackend(settings.analyst_database_dsn)))
    registry.register(ApiCallTool(UrllibBackend()))
    return registry

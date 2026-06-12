"""Tool framework: the registry, the Tool interface, and concrete tools."""

from foreman.tools.base import Tool
from foreman.tools.factory import build_default_registry
from foreman.tools.registry import ToolInvocation, ToolRegistry
from foreman.tools.web_search import WebSearchTool

__all__ = [
    "Tool",
    "ToolRegistry",
    "ToolInvocation",
    "WebSearchTool",
    "build_default_registry",
]

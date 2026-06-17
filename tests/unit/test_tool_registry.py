from typing import Any

import pytest

from foreman.schemas import Specialist
from foreman.tools import Tool, ToolRegistry


class EchoTool(Tool):
    name = "echo"
    description = "echoes its inputs"
    allowed_specialists = frozenset({Specialist.RESEARCHER})

    def run(self, **inputs: Any) -> dict[str, Any]:
        return {"echo": inputs}


class BoomTool(Tool):
    name = "boom"
    description = "always fails"
    allowed_specialists = frozenset({Specialist.RESEARCHER})

    def run(self, **inputs: Any) -> dict[str, Any]:
        raise RuntimeError("kaboom")


def _registry(*tools: Tool) -> ToolRegistry:
    reg = ToolRegistry()
    for t in tools:
        reg.register(t)
    return reg


def test_invoke_returns_output_and_logs() -> None:
    reg = _registry(EchoTool())
    out = reg.invoke("echo", Specialist.RESEARCHER, x=1)
    assert out == {"echo": {"x": 1}}

    assert len(reg.invocations) == 1
    log = reg.invocations[0]
    assert log.tool == "echo"
    assert log.success is True
    assert log.latency_ms >= 0


def test_invoke_rejects_specialist_not_on_allow_list() -> None:
    reg = _registry(EchoTool())
    with pytest.raises(ValueError, match="not permitted"):
        reg.invoke("echo", Specialist.ANALYST, x=1)


def test_invoke_unknown_tool_raises_valueerror() -> None:
    reg = _registry(EchoTool())
    with pytest.raises(ValueError, match="unknown tool"):
        reg.invoke("ghost", Specialist.RESEARCHER)


def test_invocation_log_is_bounded() -> None:
    reg = ToolRegistry(max_invocations=2)
    reg.register(EchoTool())
    for _ in range(5):
        reg.invoke("echo", Specialist.RESEARCHER, x=1)
    assert len(reg.invocations) == 2


def test_failure_is_logged_and_reraised() -> None:
    reg = _registry(BoomTool())
    with pytest.raises(RuntimeError, match="kaboom"):
        reg.invoke("boom", Specialist.RESEARCHER)
    log = reg.invocations[0]
    assert log.success is False
    assert log.error is not None

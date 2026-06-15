"""C2: the analyst generates code, runs it in the sandbox, and writes up the result
that comes back."""

from __future__ import annotations

from foreman.agents import Analyst
from foreman.llm.base import LLMProvider, T
from foreman.schemas import AnalysisCode, ResearchFindings, Specialist, Subtask
from foreman.tools import CodeExecutionTool, ToolRegistry
from foreman.tools.code_exec import FakeSandbox


class AnalystProvider(LLMProvider):
    name = "analyst-test"

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def structured_complete(self, prompt: str, schema: type[T]) -> T:
        self.prompts.append(prompt)
        if schema is AnalysisCode:
            return AnalysisCode(code="print(2 + 4)")  # type: ignore[return-value]
        if schema is ResearchFindings:
            return ResearchFindings(content="the result is 6")  # type: ignore[return-value]
        raise AssertionError(schema)


def _subtask() -> Subtask:
    return Subtask(
        id="s1",
        description="compute the mean of the figures",
        assigned_specialist=Specialist.ANALYST,
        expected_output="a number",
        complexity=1,
    )


def test_analyst_runs_code_and_writes_up_the_output() -> None:
    registry = ToolRegistry()
    registry.register(CodeExecutionTool(FakeSandbox(stdout="6.0\n")))
    provider = AnalystProvider()

    output = Analyst(registry, provider).execute(_subtask())

    assert output.produced_by is Specialist.ANALYST
    assert output.tools_used == ["code_execution"]
    # the sandbox's output was fed into the write-up prompt
    assert any("6.0" in p for p in provider.prompts)

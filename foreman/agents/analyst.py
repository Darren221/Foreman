"""The analyst specialist: compute the answer by running code, then write it up.

It mirrors the researcher's two-step shape — gather, then summarise — but its
"gather" is *running code in the sandbox*: the LLM writes a short program, the
`code_execution` tool runs it in a throwaway container, and the LLM writes up the
findings grounded in the program's output (folding in any reviewer feedback).

When a `db_query` tool is wired (a real database is configured), the analyst gains
a step in front: it first writes a read-only SQL query, runs it to pull the actual
rows, and feeds those rows into the code/write-up steps as the data to analyse.
That is the path that exercises `db_query` against live data — the sandbox itself
has no network, so the database can only be reached through the tool. With no
`db_query` registered (the offline default) the analyst behaves exactly as before,
analysing whatever upstream text it is given.
"""

from __future__ import annotations

import json

from foreman.agents.base import render_upstream
from foreman.llm.base import LLMProvider
from foreman.schemas import (
    AnalysisCode,
    ResearchFindings,
    Specialist,
    SpecialistOutput,
    SQLQuery,
    Subtask,
)
from foreman.tools import ToolRegistry

_CODE_TOOL = "code_execution"
_DB_TOOL = "db_query"
# Cap the rows folded into the prompt so a wide result set can't blow the context
# window; the SQL itself should aggregate, this is a backstop.
_MAX_PROMPT_ROWS = 200

_SQL_PROMPT = """\
You are a data analyst with read-only access to a SQL database. Write a single
read-only SELECT (or WITH ... SELECT) query that pulls the data needed to answer the
subtask. Use ONLY the tables and columns in the schema below — do not invent column
or table names. Aggregate in SQL where you can (GROUP BY, SUM, AVG) so the result is
compact. Return only the SQL.

Database schema (the only tables/columns that exist):
{schema}

Subtask: {description}
Expected output: {expected_output}

Upstream results from prior steps (context for what to query):
{upstream}

Reviewer feedback to address (if any): {feedback}
"""

_CODE_PROMPT = """\
You are a data analyst. Write a short Python program that computes what the subtask
asks for and prints the result. Use only the standard library; print the answer.

Subtask: {description}
Expected output: {expected_output}

{data_block}

Reviewer feedback to address (if any): {feedback}
"""

_WRITEUP_PROMPT = """\
Write up the analysis findings for the subtask, grounded in the program output
below. State the result and what it means; be specific; do not invent numbers.

Subtask: {description}

Program output:
{output}

Reviewer feedback to address (if any): {feedback}
"""


class Analyst:
    specialist = Specialist.ANALYST

    def __init__(self, registry: ToolRegistry, provider: LLMProvider) -> None:
        self._registry = registry
        self._provider = provider

    def execute(
        self,
        subtask: Subtask,
        feedback: str | None = None,
        upstream: list[SpecialistOutput] | None = None,
    ) -> SpecialistOutput:
        tools_used: list[str] = []
        rows = self._fetch_rows(subtask, feedback, upstream, tools_used)

        # The data the analysis is grounded in: queried rows when a DB is wired,
        # otherwise the upstream specialist outputs (the prior behaviour).
        if rows is not None:
            data_block = f"Data queried from the database (JSON rows):\n{rows}"
        else:
            data_block = (
                "Upstream results from prior steps (the data to analyse):\n"
                f"{render_upstream(upstream)}"
            )

        code = self._provider.structured_complete(
            _CODE_PROMPT.format(
                description=subtask.description,
                expected_output=subtask.expected_output,
                data_block=data_block,
                feedback=feedback or "none",
            ),
            AnalysisCode,
        ).code
        result = self._registry.invoke(_CODE_TOOL, self.specialist, code=code)
        tools_used.append(_CODE_TOOL)
        output = result.get("stdout") or result.get("stderr") or "(no output)"
        content = self._provider.structured_complete(
            _WRITEUP_PROMPT.format(
                description=subtask.description, output=output, feedback=feedback or "none"
            ),
            ResearchFindings,
        ).content
        return SpecialistOutput(
            subtask_id=subtask.id,
            content=content,
            tools_used=tools_used,
            produced_by=self.specialist,
        )

    def _fetch_rows(
        self,
        subtask: Subtask,
        feedback: str | None,
        upstream: list[SpecialistOutput] | None,
        tools_used: list[str],
    ) -> str | None:
        """Query the database for the subtask's data, or None when no DB is wired.

        Appends `db_query` to `tools_used` only on a successful query, so the run's
        tool record reflects what actually ran."""
        if not self._registry.has(_DB_TOOL):
            return None
        # Ground the SQL in the live schema so the analyst queries real columns
        # rather than guessing names. Empty when the backend can't introspect.
        db_tool = self._registry.get(_DB_TOOL)
        schema = db_tool.schema() if hasattr(db_tool, "schema") else ""
        sql = self._provider.structured_complete(
            _SQL_PROMPT.format(
                schema=schema or "(schema unavailable; infer from the subtask)",
                description=subtask.description,
                expected_output=subtask.expected_output,
                upstream=render_upstream(upstream),
                feedback=feedback or "none",
            ),
            SQLQuery,
        ).sql
        result = self._registry.invoke(_DB_TOOL, self.specialist, query=sql)
        tools_used.append(_DB_TOOL)
        rows = result.get("rows") or []
        return json.dumps(rows[:_MAX_PROMPT_ROWS], default=str)

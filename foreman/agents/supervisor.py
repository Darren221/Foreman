"""The supervisor agent: decompose a task into a validated plan.

Its single Phase-1 responsibility is *task decomposition* — turn a free-text
request into an ordered, dependency-aware `Plan`. It relies on the provider's
structured output to get a typed `Plan` back, then guards it: the model may only
assign work to specialists that are actually wired.
"""

from __future__ import annotations

from foreman.llm.base import LLMProvider
from foreman.schemas import Plan, Specialist, Task

_DEFAULT_AVAILABLE = frozenset({Specialist.RESEARCHER})

_PROMPT = """\
You are the supervisor of a team of specialist agents. Decompose the task below
into an ordered list of subtasks.

Rules:
- Assign every subtask to one of these available specialists: {specialists}.
- Give each subtask an id, a clear description, the inputs it needs, the expected
  output, and a complexity from 1 (trivial) to 5 (hard).
- Use the `dependencies` field to reference the ids of subtasks that must finish
  first. Dependencies must not form a cycle.

Task id: {task_id}
Task: {description}
"""


class Supervisor:
    def __init__(
        self,
        provider: LLMProvider,
        available_specialists: frozenset[Specialist] | set[Specialist] | None = None,
    ) -> None:
        self._provider = provider
        self._available = frozenset(available_specialists or _DEFAULT_AVAILABLE)

    def plan(self, task: Task) -> Plan:
        prompt = _PROMPT.format(
            specialists=", ".join(sorted(s.value for s in self._available)),
            task_id=task.id,
            description=task.description,
        )
        plan = self._provider.structured_complete(prompt, Plan)
        self._check_specialists(plan)
        return plan

    def _check_specialists(self, plan: Plan) -> None:
        for subtask in plan.subtasks:
            if subtask.assigned_specialist not in self._available:
                raise ValueError(
                    f"subtask {subtask.id} assigned to "
                    f"'{subtask.assigned_specialist.value}', which is not available"
                )

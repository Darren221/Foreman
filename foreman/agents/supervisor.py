"""The supervisor agent: decompose a task into a validated plan.

Its single Phase-1 responsibility is *task decomposition* — turn a free-text
request into an ordered, dependency-aware `Plan`. It relies on the provider's
structured output to get a typed `Plan` back, then guards it: the model may only
assign work to specialists that are actually wired.
"""

from __future__ import annotations

from foreman.llm.base import LLMProvider
from foreman.schemas import Plan, Specialist, SpecialistOutput, Synthesis, Task, TaskMemory

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

Relevant past tasks (use them to inform your plan):
{memories}

Task id: {task_id}
Task: {description}
"""

_SYNTHESIS_PROMPT = """\
You are the supervisor. Combine the specialist findings below into a single,
coherent written answer to the original task.

Task: {description}

Findings:
{findings}
"""


class Supervisor:
    def __init__(
        self,
        provider: LLMProvider,
        available_specialists: frozenset[Specialist] | set[Specialist] | None = None,
    ) -> None:
        self._provider = provider
        self._available = frozenset(available_specialists or _DEFAULT_AVAILABLE)

    def plan(self, task: Task, memories: list[TaskMemory] | None = None) -> Plan:
        prompt = _PROMPT.format(
            specialists=", ".join(sorted(s.value for s in self._available)),
            memories=self._render_memories(memories),
            task_id=task.id,
            description=task.description,
        )
        plan = self._provider.structured_complete(prompt, Plan)
        self._check_specialists(plan)
        return plan

    @staticmethod
    def _render_memories(memories: list[TaskMemory] | None) -> str:
        if not memories:
            return "(none)"
        lines = []
        for m in memories:
            tools = ", ".join(m.tools_used) or "none"
            lines.append(f"- {m.task_description} (outcome: {m.outcome}, tools: {tools})")
        return "\n".join(lines)

    def synthesize(self, task: Task, outputs: list[SpecialistOutput]) -> str:
        findings = "\n\n".join(o.content for o in outputs)
        prompt = _SYNTHESIS_PROMPT.format(description=task.description, findings=findings)
        return self._provider.structured_complete(prompt, Synthesis).result

    def _check_specialists(self, plan: Plan) -> None:
        for subtask in plan.subtasks:
            if subtask.assigned_specialist not in self._available:
                raise ValueError(
                    f"subtask {subtask.id} assigned to "
                    f"'{subtask.assigned_specialist.value}', which is not available"
                )

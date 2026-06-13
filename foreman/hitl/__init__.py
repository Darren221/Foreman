"""Human-in-the-loop: deciding when to pause for a human and carrying their
decision back into the run.

The escalation *policy* (H2) classifies graph state into an `Escalation`. The
`ApprovalQueue` persists pending escalations and decisions; the `Runner` drives a
task through the graph, pausing at an approval and resuming on a `Decision`.
"""

from __future__ import annotations

from foreman.hitl.policy import (
    ApprovalLevel,
    Escalation,
    EscalationPolicy,
    EscalationTrigger,
)
from foreman.hitl.queue import ApprovalQueue, Decision, DecisionKind, PendingApproval
from foreman.hitl.runner import Runner, RunResult

__all__ = [
    "ApprovalLevel",
    "ApprovalQueue",
    "Decision",
    "DecisionKind",
    "Escalation",
    "EscalationPolicy",
    "EscalationTrigger",
    "PendingApproval",
    "RunResult",
    "Runner",
]

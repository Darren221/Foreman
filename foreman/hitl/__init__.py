"""Human-in-the-loop: deciding when to pause for a human and carrying their
decision back into the run.

H2 provides the escalation *policy* — a pure classifier over graph state. The
approval queue, resume semantics, and review UI (H3–H4) build on it.
"""

from __future__ import annotations

from foreman.hitl.policy import (
    ApprovalLevel,
    Escalation,
    EscalationPolicy,
    EscalationTrigger,
)

__all__ = [
    "ApprovalLevel",
    "Escalation",
    "EscalationPolicy",
    "EscalationTrigger",
]

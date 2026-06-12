"""The Tool interface.

A tool is a capability an agent can invoke through the registry. It carries the
metadata the registry needs to mediate access (who may call it) and to make the
call observable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from foreman.schemas import Specialist


class Tool(ABC):
    name: str
    description: str
    allowed_specialists: frozenset[Specialist]
    rate_limit_per_min: int | None = None

    @abstractmethod
    def run(self, **inputs: Any) -> dict[str, Any]:
        """Execute the tool and return its result."""

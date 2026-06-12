"""The LLM provider interface every agent depends on.

Agents never import `openai` or `anthropic` directly — they call an
`LLMProvider`. That indirection (dependency inversion) is what makes multi-model
routing a config choice rather than a code change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMProvider(ABC):
    """A model backend that can return a typed, structured completion."""

    name: str

    @abstractmethod
    def structured_complete(self, prompt: str, schema: type[T]) -> T:
        """Run `prompt` and parse the model's reply into `schema`."""

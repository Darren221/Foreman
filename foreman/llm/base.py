"""The LLM provider interface every agent depends on.

Agents never import `openai` or `anthropic` directly — they call an
`LLMProvider`. That indirection (dependency inversion) is what makes multi-model
routing a config choice rather than a code change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


@dataclass
class Usage:
    """Token counts for a single completion, for cost and trace attribution."""

    input_tokens: int = 0
    output_tokens: int = 0


class LLMProvider(ABC):
    """A model backend that can return a typed, structured completion.

    `model` and `last_usage` are the seam the tracing layer reads: a provider
    sets `last_usage` after each call so a wrapper can attribute tokens without
    changing the completion's return type. They default empty for fakes that
    don't report usage.
    """

    name: str
    model: str = ""
    last_usage: Usage | None = None

    @abstractmethod
    def structured_complete(self, prompt: str, schema: type[T]) -> T:
        """Run `prompt` and parse the model's reply into `schema`."""

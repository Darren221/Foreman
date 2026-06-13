"""Instrumentation wrappers that add spans around the moving parts.

`TracingProvider` wraps any `LLMProvider` so every completion becomes an `llm`
span carrying the `gen_ai.*` attributes — model on the way in, token usage on the
way out (read from the inner provider's `last_usage`). Wrapping (rather than
editing each adapter) means a fake provider in tests is instrumented too, so the
spans are testable offline.
"""

from __future__ import annotations

from foreman.llm.base import LLMProvider, T
from foreman.observability.semconv import (
    GEN_AI_OPERATION_NAME,
    GEN_AI_REQUEST_MODEL,
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
)
from foreman.observability.tracer import Tracer


class TracingProvider(LLMProvider):
    """Decorates an `LLMProvider` with an `llm` span per completion."""

    def __init__(self, inner: LLMProvider, tracer: Tracer) -> None:
        self._inner = inner
        self._tracer = tracer
        self.name = inner.name
        self.model = inner.model

    def structured_complete(self, prompt: str, schema: type[T]) -> T:
        attributes = {
            GEN_AI_OPERATION_NAME: "chat",
            GEN_AI_REQUEST_MODEL: self._inner.model,
        }
        with self._tracer.span(f"llm:{schema.__name__}", kind="llm", attributes=attributes) as span:
            result = self._inner.structured_complete(prompt, schema)
            usage = self._inner.last_usage
            if span is not None and usage is not None:
                span.set_attribute(GEN_AI_USAGE_INPUT_TOKENS, usage.input_tokens)
                span.set_attribute(GEN_AI_USAGE_OUTPUT_TOKENS, usage.output_tokens)
            return result

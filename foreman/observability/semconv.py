"""OpenTelemetry `gen_ai.*` semantic-convention attribute keys.

These live in the *experimental* GenAI semconv, which is still stabilizing — we
opt into it via `OTEL_SEMCONV_STABILITY_OPT_IN` (set in this package's __init__).
Isolating the keys here means a semconv rename is a one-file change.
"""

from __future__ import annotations

GEN_AI_OPERATION_NAME = "gen_ai.operation.name"
GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"

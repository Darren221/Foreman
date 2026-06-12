# Foreman

A multi-agent orchestration platform. A supervisor agent decomposes complex
tasks, delegates subtasks to specialist tool-using agents, validates results
through a reviewer, maintains memory across interactions, and escalates to a
human operator when confidence is low or an action is sensitive — with full
observability into every agent decision.

## Status

Early development. The agent pipeline (supervisor → specialist → reviewer) runs
end-to-end as a LangGraph state machine; memory, human-in-the-loop, and
observability are being layered in.

## Requirements

- Python 3.11–3.13. (ChromaDB depends on `onnxruntime`, which has no wheel for
  3.14 yet — use 3.12 for development.)

## Development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Copy `.env.example` to `.env` and fill in provider API keys before running
against live models.

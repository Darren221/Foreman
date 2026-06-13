"""The `--trace` flag wires a real tracer into a run and records to the trace
store; without it, a run is untraced (the default stays lightweight). The
pipeline itself is stubbed here — we're testing the CLI's wiring, not the graph."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from foreman.cli import main
from foreman.observability import NoOpTracer, OTelTracer


def _stub_run(monkeypatch: pytest.MonkeyPatch, sink: dict[str, Any]) -> None:
    def fake_run_task(provider: Any, task: Any, tracer: Any = None) -> dict[str, Any]:
        sink["tracer"] = tracer
        return {"result": "done"}

    monkeypatch.setattr("foreman.cli.run_task", fake_run_task)
    monkeypatch.setattr("foreman.llm.select_provider", lambda settings: object())


def test_trace_flag_passes_a_real_tracer(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    sink: dict[str, Any] = {}
    _stub_run(monkeypatch, sink)
    monkeypatch.setenv("TRACE_PATH", str(tmp_path / "traces.sqlite"))

    assert main(["run", "--trace", "research bikes"]) == 0
    assert isinstance(sink["tracer"], OTelTracer)
    assert (tmp_path / "traces.sqlite").exists()  # the store was opened


def test_default_run_is_untraced(monkeypatch: pytest.MonkeyPatch) -> None:
    sink: dict[str, Any] = {}
    _stub_run(monkeypatch, sink)

    assert main(["run", "research bikes"]) == 0
    assert sink["tracer"] is None or isinstance(sink["tracer"], NoOpTracer)

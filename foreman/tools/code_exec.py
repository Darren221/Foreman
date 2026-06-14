"""Sandboxed code execution: run agent-authored Python in a throwaway container.

The tool depends on a `SandboxBackend` interface, not on Docker directly, so tests
inject a fake (no daemon, no network) and the real isolation is one swappable
backend. The analyst is the only specialist allowed to run code.
"""

from __future__ import annotations

import subprocess
from typing import Any, Protocol

from foreman.schemas import Specialist
from foreman.tools.base import Tool


class SandboxBackend(Protocol):
    def run(self, code: str) -> dict[str, Any]:
        """Execute `code` in isolation; return at least `stdout` and `exit_code`."""
        ...


class DockerSandbox:
    """Runs code in a throwaway container with the network off and resource limits.

    `docker run --rm --network none --memory ... --cpus ... <image> python -c <code>`.
    The code is passed as an argument (no shell), so there's no shell-injection path;
    isolation and resource bounds come from the container, and a timeout caps runtime.
    """

    def __init__(
        self,
        image: str = "python:3.12-slim",
        timeout_s: int = 10,
        memory: str = "256m",
        cpus: str = "1.0",
    ) -> None:
        self._image = image
        self._timeout_s = timeout_s
        self._memory = memory
        self._cpus = cpus

    def run(self, code: str) -> dict[str, Any]:
        try:
            proc = subprocess.run(
                [
                    "docker", "run", "--rm", "--network", "none",
                    "--memory", self._memory, "--cpus", self._cpus,
                    self._image, "python", "-c", code,
                ],
                capture_output=True,
                text=True,
                timeout=self._timeout_s,
            )
        except subprocess.TimeoutExpired:
            return {"stdout": "", "stderr": "timed out", "exit_code": 124}
        return {"stdout": proc.stdout, "stderr": proc.stderr, "exit_code": proc.returncode}


class FakeSandbox:
    """Deterministic stand-in for tests — runs nothing, returns canned output."""

    def __init__(self, stdout: str = "", exit_code: int = 0) -> None:
        self._stdout = stdout
        self._exit_code = exit_code

    def run(self, code: str) -> dict[str, Any]:
        return {"stdout": self._stdout, "stderr": "", "exit_code": self._exit_code}


class CodeExecutionTool(Tool):
    name = "code_execution"
    description = "Run Python code in a sandboxed container and return its output."
    allowed_specialists = frozenset({Specialist.ANALYST})
    rate_limit_per_min = 20

    def __init__(self, backend: SandboxBackend) -> None:
        self._backend = backend

    def run(self, **inputs: Any) -> dict[str, Any]:
        return self._backend.run(inputs["code"])

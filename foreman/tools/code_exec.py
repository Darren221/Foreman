"""Sandboxed code execution: run agent-authored Python in a throwaway container.

The tool depends on a `SandboxBackend` interface, not on Docker directly, so tests
inject a fake (no daemon, no network) and the real isolation is one swappable
backend. The analyst is the only specialist allowed to run code.
"""

from __future__ import annotations

from typing import Any, Protocol

from foreman.schemas import Specialist
from foreman.tools.base import Tool
from foreman.tools.limits import MAX_OUTPUT_BYTES


class SandboxBackend(Protocol):
    def run(self, code: str) -> dict[str, Any]:
        """Execute `code` in isolation; return at least `stdout` and `exit_code`."""
        ...


class DockerSandbox:
    """Runs code in a throwaway container with the network off and resource limits.

    Uses the Docker SDK: start a detached container that runs `python -c <code>` with
    the network disabled and memory/CPU capped, wait for it (bounded by a timeout),
    collect its output, and remove it. The code is passed as an argument list (no
    shell), so there's no shell-injection path; isolation comes from the container.
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
        self._nano_cpus = int(float(cpus) * 1_000_000_000)
        self._client: Any = None

    def _docker(self) -> Any:
        if self._client is None:
            import docker

            # docker-py ships py.typed but doesn't re-export from_env for mypy.
            self._client = docker.from_env()  # type: ignore[attr-defined]
        return self._client

    def run(self, code: str) -> dict[str, Any]:
        container = self._docker().containers.run(
            self._image,
            command=["python", "-c", code],
            network_disabled=True,
            mem_limit=self._memory,
            nano_cpus=self._nano_cpus,
            detach=True,
        )
        try:
            status = container.wait(timeout=self._timeout_s)
            # Cap captured output so a program printing gigabytes can't blow up the
            # worker's memory. (Truncates the bytes Docker buffered; the container's
            # own mem_limit doesn't bound what we read back.)
            stdout = container.logs(stdout=True, stderr=False)[:MAX_OUTPUT_BYTES]
            stderr = container.logs(stdout=False, stderr=True)[:MAX_OUTPUT_BYTES]
            return {
                "stdout": stdout.decode("utf-8", "replace"),
                "stderr": stderr.decode("utf-8", "replace"),
                "exit_code": status.get("StatusCode", -1),
            }
        except Exception:
            container.kill()
            return {"stdout": "", "stderr": "timed out", "exit_code": 124}
        finally:
            container.remove(force=True)


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

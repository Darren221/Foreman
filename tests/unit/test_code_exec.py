"""C2: the sandboxed code-execution tool. The unit tests use a fake backend (no
Docker); a real-Docker test is skipped when the daemon isn't available."""

from __future__ import annotations

import pytest

from foreman.schemas import Specialist
from foreman.tools.code_exec import CodeExecutionTool, DockerSandbox, FakeSandbox


def test_tool_runs_code_through_its_backend() -> None:
    tool = CodeExecutionTool(FakeSandbox(stdout="42\n"))
    result = tool.run(code="print(42)")
    assert result["stdout"] == "42\n"
    assert result["exit_code"] == 0


def test_tool_is_restricted_to_the_analyst() -> None:
    assert CodeExecutionTool(FakeSandbox()).allowed_specialists == frozenset({Specialist.ANALYST})


@pytest.mark.requires_docker
def test_docker_sandbox_runs_real_code() -> None:
    result = DockerSandbox(timeout_s=120).run("print(6 * 7)")
    assert "42" in result["stdout"]
    assert result["exit_code"] == 0


@pytest.mark.requires_docker
def test_docker_sandbox_is_hardened() -> None:
    sandbox = DockerSandbox(timeout_s=120)
    # runs as nobody (uid 65534), not root
    assert sandbox.run("import os; print(os.getuid())")["stdout"].strip() == "65534"
    # the writable in-memory /tmp keeps legit scratch files working
    wrote = sandbox.run("open('/tmp/x', 'w').write('ok'); print('wrote')")
    assert wrote["stdout"].strip() == "wrote"
    # but the rest of the filesystem is read-only: a write outside /tmp fails
    assert sandbox.run("open('/srv/x', 'w').write('no')")["exit_code"] != 0


@pytest.mark.requires_docker
def test_docker_sandbox_times_out() -> None:
    result = DockerSandbox(timeout_s=2).run("import time; time.sleep(30)")
    assert result["exit_code"] == 124
    assert "timed out" in result["stderr"]

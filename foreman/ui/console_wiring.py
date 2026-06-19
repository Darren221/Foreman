"""The logic beneath the submit-and-watch console, kept free of Streamlit so it can
be tested. The console is a thin *client* of the orchestration API (not an in-process
runner like the review/explorer UIs): it POSTs a task and polls its status, so the work
runs on the api+worker services, exactly as a real caller would drive it.

Two concerns live here: talking to the API over HTTP, and turning a task's JSON
response into the few display values the page renders. The Streamlit page owns neither.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol
from urllib import request as _request


@dataclass(frozen=True)
class TaskView:
    """The slice of a task's API response the console renders. A flattened, client-side
    view so the page never touches the server's pydantic models."""

    id: str
    status: str  # "completed" | "pending"
    result: str | None = None
    approval_id: str | None = None
    escalation_summary: str | None = None


def task_view_from_payload(payload: dict[str, Any]) -> TaskView:
    """Parse a `TaskResponse` JSON body into a `TaskView`. A pending run carries an
    escalation; we distil it to a one-line summary (`trigger → level: reason`) rather
    than surfacing the whole object, since the review UI is where it's acted on."""
    escalation = payload.get("escalation")
    summary = None
    if escalation:
        trigger = escalation.get("trigger", "?")
        level = escalation.get("level", "?")
        reason = escalation.get("reason", "")
        summary = f"{trigger} → {level}: {reason}".rstrip(": ")
    return TaskView(
        id=payload["id"],
        status=payload["status"],
        result=payload.get("result"),
        approval_id=payload.get("approval_id"),
        escalation_summary=summary,
    )


def is_pending(view: TaskView) -> bool:
    """True when the run paused for a human approval (so the page points at the review UI)."""
    return view.status == "pending"


def is_running(view: TaskView) -> bool:
    """True when the run is still executing (e.g. resumed after an approval and not yet
    finished), so the page invites a re-check instead of showing an empty result."""
    return view.status == "running"


def body_text(view: TaskView) -> str:
    """What to show under the status line: the synthesised result when finished, a
    pointer to the approval when paused, or a re-check hint while it's still running."""
    if is_pending(view):
        detail = view.escalation_summary or "Paused for approval."
        return f"Paused for approval — resolve it in the review UI.\n\n{detail}"
    if is_running(view):
        return "Still running — give it a moment, then re-check."
    return view.result or "(the run finished but produced no result text)"


class ForemanClient(Protocol):
    def submit(
        self, description: str, *, require_approval: bool = False, sensitive: bool = False
    ) -> TaskView: ...

    def status(self, run_id: str) -> TaskView: ...


class HttpForemanClient:
    """Real client over the orchestration API, using the standard library (the same
    no-third-party-HTTP posture as the fetch tool). `base_url` is the api service, e.g.
    `http://localhost:8000` locally or `http://api:8000` in the compose network."""

    def __init__(self, base_url: str, *, timeout_s: float = 120.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s

    def submit(
        self, description: str, *, require_approval: bool = False, sensitive: bool = False
    ) -> TaskView:
        payload = self._call(
            "POST",
            "/tasks",
            {
                "description": description,
                "require_approval": require_approval,
                "sensitive": sensitive,
            },
        )
        return task_view_from_payload(payload)

    def status(self, run_id: str) -> TaskView:
        return task_view_from_payload(self._call("GET", f"/tasks/{run_id}"))

    def _call(
        self, method: str, path: str, body: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        data = json.dumps(body).encode() if body is not None else None
        req = _request.Request(
            f"{self._base_url}{path}",
            data=data,
            method=method,
            headers={"content-type": "application/json"},
        )
        with _request.urlopen(req, timeout=self._timeout_s) as response:  # noqa: S310 (fixed api base)
            parsed: dict[str, Any] = json.loads(response.read().decode())
            return parsed

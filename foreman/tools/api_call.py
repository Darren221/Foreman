"""Read-only HTTP fetch.

The tool depends on an `HttpBackend` interface (real stdlib client + a fake for
tests). It's GET-only — a specialist can read from an API, never POST/PUT/DELETE —
and rejects non-HTTP schemes so it can't be turned into a file/local read.
"""

from __future__ import annotations

from typing import Any, Protocol

from foreman.schemas import Specialist
from foreman.tools.base import Tool


class HttpBackend(Protocol):
    def get(self, url: str) -> dict[str, Any]:
        """Fetch `url`; return at least `status` and `body`."""
        ...


class UrllibBackend:
    """Real client over the standard library — no third-party HTTP dependency."""

    def __init__(self, timeout_s: int = 10) -> None:
        self._timeout_s = timeout_s

    def get(self, url: str) -> dict[str, Any]:
        from urllib.request import urlopen

        with urlopen(url, timeout=self._timeout_s) as response:  # noqa: S310 (scheme checked in the tool)
            return {
                "status": response.status,
                "body": response.read().decode("utf-8", "replace"),
            }


class FakeHttp:
    """Deterministic stand-in for tests — returns a canned response for any URL."""

    def __init__(self, status: int = 200, body: str = "") -> None:
        self._status = status
        self._body = body

    def get(self, url: str) -> dict[str, Any]:
        return {"status": self._status, "body": self._body}


class ApiCallTool(Tool):
    name = "api_call"
    description = "Fetch data from an HTTP(S) URL (read-only GET)."
    allowed_specialists = frozenset({Specialist.RESEARCHER, Specialist.ANALYST})
    rate_limit_per_min = 60

    def __init__(self, backend: HttpBackend) -> None:
        self._backend = backend

    def run(self, **inputs: Any) -> dict[str, Any]:
        url = inputs["url"]
        if not url.startswith(("http://", "https://")):
            raise ValueError("url must be an http(s) URL")
        return self._backend.get(url)

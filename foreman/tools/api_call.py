"""Read-only HTTP fetch.

The tool depends on an `HttpBackend` interface (real stdlib client + a fake for
tests). It's GET-only — a specialist can read from an API, never POST/PUT/DELETE —
rejects non-HTTP schemes, and (in the real client) blocks SSRF to internal/metadata
addresses, on the initial URL and on every redirect hop.
"""

from __future__ import annotations

from typing import Any, Protocol
from urllib.request import HTTPRedirectHandler

from foreman.schemas import Specialist
from foreman.tools.base import Tool
from foreman.tools.limits import read_capped
from foreman.tools.net import assert_public_url


class HttpBackend(Protocol):
    def get(self, url: str) -> dict[str, Any]:
        """Fetch `url`; return at least `status` and `body`."""
        ...


class _ValidatingRedirectHandler(HTTPRedirectHandler):
    """Re-checks each redirect target, so an allowed host can't 302 us inward."""

    def redirect_request(
        self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str
    ) -> Any:
        assert_public_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class UrllibBackend:
    """Real client over the standard library — no third-party HTTP dependency."""

    def __init__(self, timeout_s: int = 10) -> None:
        self._timeout_s = timeout_s

    def get(self, url: str) -> dict[str, Any]:
        from urllib.request import build_opener

        assert_public_url(url)
        opener = build_opener(_ValidatingRedirectHandler())
        with opener.open(url, timeout=self._timeout_s) as response:  # noqa: S310 (validated above)
            raw, truncated = read_capped(response)
            return {
                "status": response.status,
                "body": raw.decode("utf-8", "replace"),
                "truncated": truncated,
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

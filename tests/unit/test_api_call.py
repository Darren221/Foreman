"""C2: the API-call tool fetches an HTTP(S) URL (read-only GET) through a backend,
and refuses non-HTTP schemes."""

from __future__ import annotations

import pytest

from foreman.schemas import Specialist
from foreman.tools.api_call import ApiCallTool, FakeHttp


def test_fetches_through_the_backend() -> None:
    tool = ApiCallTool(FakeHttp(status=200, body='{"ok": true}'))
    result = tool.run(url="https://example.org/data")
    assert result["status"] == 200
    assert result["body"] == '{"ok": true}'


def test_rejects_non_http_urls() -> None:
    tool = ApiCallTool(FakeHttp())
    with pytest.raises(ValueError, match="http"):
        tool.run(url="file:///etc/passwd")


def test_allowed_for_researcher_and_analyst() -> None:
    assert ApiCallTool(FakeHttp()).allowed_specialists == frozenset(
        {Specialist.RESEARCHER, Specialist.ANALYST}
    )

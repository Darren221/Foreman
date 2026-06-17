"""Provider calls retry transient errors with bounded backoff."""

from __future__ import annotations

import pytest

from foreman.llm.retry import with_retries


def test_retries_then_succeeds() -> None:
    calls = {"n": 0}

    def flaky() -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise ValueError("transient")
        return "ok"

    result = with_retries(
        flaky, attempts=3, base_delay=0, retry_on=(ValueError,), sleep=lambda _s: None
    )
    assert result == "ok"
    assert calls["n"] == 3


def test_gives_up_after_max_attempts() -> None:
    calls = {"n": 0}

    def always_fails() -> str:
        calls["n"] += 1
        raise ValueError("transient")

    with pytest.raises(ValueError):
        with_retries(
            always_fails, attempts=2, base_delay=0, retry_on=(ValueError,), sleep=lambda _s: None
        )
    assert calls["n"] == 2


def test_does_not_retry_unlisted_exceptions() -> None:
    def boom() -> str:
        raise KeyError("not transient")

    with pytest.raises(KeyError):
        with_retries(boom, attempts=3, retry_on=(ValueError,), sleep=lambda _s: None)

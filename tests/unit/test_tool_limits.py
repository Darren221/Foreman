"""Tool output is capped so a runaway program/response/file can't OOM the worker."""

from __future__ import annotations

from io import BytesIO

from foreman.tools.limits import read_capped


def test_read_capped_truncates_and_flags() -> None:
    data, truncated = read_capped(BytesIO(b"x" * 100), limit=10)
    assert data == b"x" * 10
    assert truncated is True


def test_read_capped_passes_small_input_through() -> None:
    data, truncated = read_capped(BytesIO(b"hi"), limit=10)
    assert data == b"hi"
    assert truncated is False

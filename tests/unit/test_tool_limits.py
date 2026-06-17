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


def test_read_capped_exact_boundary_is_not_truncated() -> None:
    data, truncated = read_capped(BytesIO(b"x" * 10), limit=10)
    assert data == b"x" * 10
    assert truncated is False


def test_read_capped_handles_short_reads() -> None:
    # A reader that returns one byte per read() must still be capped correctly.
    class _Drip:
        def __init__(self, data: bytes) -> None:
            self._data = data
            self._i = 0

        def read(self, n: int) -> bytes:
            chunk = self._data[self._i : self._i + 1]  # one byte, ignoring n
            self._i += 1
            return chunk

    data, truncated = read_capped(_Drip(b"x" * 100), limit=10)
    assert data == b"x" * 10
    assert truncated is True

"""Bounds on tool output, so a runaway program/response/file can't OOM the worker.

A sandboxed program, an HTTP response, or a file can be arbitrarily large; reading
all of it into the worker's memory is the risk (the container's own memory cap
doesn't bound what the *worker* reads from it). Every tool that reads external bytes
routes through `read_capped`.
"""

from __future__ import annotations

from typing import Protocol

# 1 MB is plenty for an LLM to summarise; anything larger is truncated.
MAX_OUTPUT_BYTES = 1_000_000


class _Reader(Protocol):
    def read(self, n: int = ..., /) -> bytes: ...


def read_capped(reader: _Reader, limit: int = MAX_OUTPUT_BYTES) -> tuple[bytes, bool]:
    """Read at most `limit` bytes from `reader`. Returns (data, truncated). Reads one
    extra byte to detect (but not keep) overflow, so memory stays bounded by `limit`."""
    data = reader.read(limit + 1)
    if len(data) > limit:
        return data[:limit], True
    return data, False

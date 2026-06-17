"""Bounded retry with exponential backoff for transient provider errors.

LLM APIs return transient failures (rate limits, 5xx, dropped connections) that a
short retry usually clears. `with_retries` wraps a call, retrying only the exception
types it's told are transient, with a capped number of attempts and exponential
backoff. `sleep` is injectable so tests don't actually wait.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TypeVar

R = TypeVar("R")


def with_retries(
    fn: Callable[[], R],
    *,
    attempts: int = 3,
    base_delay: float = 0.5,
    retry_on: tuple[type[BaseException], ...] = (Exception,),
    sleep: Callable[[float], None] = time.sleep,
) -> R:
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except retry_on:
            if attempt >= attempts:
                raise
            sleep(base_delay * 2 ** (attempt - 1))
    raise AssertionError("unreachable")  # the loop always returns or raises

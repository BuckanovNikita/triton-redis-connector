"""Composable middleware that wraps processors for cross-cutting concerns.

Each middleware satisfies the Processor[TIn, TOut] protocol so they
compose naturally via nesting.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from collections.abc import Callable


class RetryMiddleware[TIn, TOut]:
    """Wraps a processor with retry logic using exponential backoff.

    Uses time.sleep() for delays between attempts. Safe because each
    parallel node runs in its own thread. Stateless — no timers, no locks.
    """

    def __init__(
        self,
        inner: Callable[[TIn], TOut | None],
        max_attempts: int = 3,
        base_delay_s: float = 0.1,
        backoff_multiplier: float = 4.0,
        on_failure: Callable[[TIn, str], TOut | None] | None = None,
    ) -> None:
        """Initialize retry middleware with inner callable and backoff parameters."""
        self._inner = inner
        self._max_attempts = max_attempts
        self._base_delay_s = base_delay_s
        self._backoff_multiplier = backoff_multiplier
        self._on_failure = on_failure

    def __call__(self, event: TIn) -> TOut | None:
        """Execute inner processor with retries on exception."""
        for attempt in range(self._max_attempts):
            try:
                return self._inner(event)
            except Exception as exc:
                if attempt + 1 >= self._max_attempts:
                    if self._on_failure is not None:
                        return self._on_failure(event, str(exc))
                    raise
                delay = self._base_delay_s * (self._backoff_multiplier**attempt)
                logger.warning(
                    "Attempt {}/{} failed ({}), retrying in {:.3f}s",
                    attempt + 1,
                    self._max_attempts,
                    exc,
                    delay,
                )
                time.sleep(delay)
        return None

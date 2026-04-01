"""Composable middleware classes that wrap processors for cross-cutting concerns.

Each middleware satisfies the Processor[TIn, TOut] protocol so they
compose naturally via nesting: RetryMiddleware(ConcurrencyLimitMiddleware(inner)).
"""

from __future__ import annotations

import threading
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

    def __init__(  # noqa: D107
        self,
        inner: Callable[[TIn], TOut | None],
        max_attempts: int = 3,
        base_delay_s: float = 0.1,
        backoff_multiplier: float = 4.0,
        on_failure: Callable[[TIn, str], TOut | None] | None = None,
    ) -> None:
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


class ConcurrencyLimitMiddleware[TIn, TOut]:
    """Wraps a processor with semaphore-based concurrency limiting.

    When the semaphore cannot be acquired within the timeout, the event
    is dropped and None is returned.
    """

    def __init__(  # noqa: D107
        self,
        inner: Callable[[TIn], TOut | None],
        max_concurrent: int,
        acquire_timeout: float = 0.1,
    ) -> None:
        self._inner = inner
        self._semaphore = threading.Semaphore(max_concurrent)
        self._acquire_timeout = acquire_timeout

    def __call__(self, event: TIn) -> TOut | None:
        """Execute inner processor if concurrency limit not reached."""
        acquired = self._semaphore.acquire(timeout=self._acquire_timeout)
        if not acquired:
            logger.error("Concurrency limit reached, dropping event")
            return None
        try:
            return self._inner(event)
        finally:
            self._semaphore.release()

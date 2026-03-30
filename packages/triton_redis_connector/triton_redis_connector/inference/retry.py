"""Retry handler for async Triton inference with exponential backoff.

Uses threading.Timer to schedule retry attempts. The semaphore is
acquired by the caller before the first attempt, and released here
only on final success or final failure (all retries exhausted).
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from collections.abc import Callable


class RetryHandler:
    """Schedules retries with exponential backoff via threading.Timer.

    Configured with max_attempts, base_delay, and backoff_multiplier.
    Delay for attempt N is: base_delay * multiplier^attempt
    (e.g., 0.1s, 0.4s, 1.6s for base=0.1, multiplier=4.0).
    """

    def __init__(  # noqa: D107
        self,
        max_attempts: int = 3,
        base_delay_s: float = 0.1,
        backoff_multiplier: float = 4.0,
    ) -> None:
        self._max_attempts = max_attempts
        self._base_delay_s = base_delay_s
        self._backoff_multiplier = backoff_multiplier
        self._active_timers: list[threading.Timer] = []
        self._lock = threading.Lock()

    @property
    def max_attempts(self) -> int:
        """Return the maximum number of retry attempts."""
        return self._max_attempts

    def execute_with_retry(
        self,
        execute_fn: Callable[[], None],
        on_success: Callable[[Any], None],
        on_failure: Callable[[str], None],
        semaphore: threading.Semaphore,
    ) -> None:
        """Begin the first attempt. Retries are scheduled via Timer on failure.

        The semaphore must already be acquired by the caller. It will be
        released here on final success or final failure.
        """
        self._attempt(
            execute_fn=execute_fn,
            on_success=on_success,
            on_failure=on_failure,
            semaphore=semaphore,
            attempt=0,
        )

    def _attempt(
        self,
        execute_fn: Callable[[], None],
        on_success: Callable[[Any], None],
        on_failure: Callable[[str], None],
        semaphore: threading.Semaphore,
        attempt: int,
    ) -> None:
        try:
            execute_fn()
            on_success(None)
            semaphore.release()
        except Exception as exc:  # noqa: BLE001
            if attempt + 1 >= self._max_attempts:
                logger.error(
                    "All {} retry attempts exhausted: {}",
                    self._max_attempts,
                    exc,
                )
                on_failure(str(exc))
                semaphore.release()
                return

            delay = self._base_delay_s * (self._backoff_multiplier**attempt)
            next_attempt = attempt + 1
            logger.warning(
                "Attempt {}/{} failed ({}), retrying in {:.3f}s",
                next_attempt,
                self._max_attempts,
                exc,
                delay,
            )
            timer = threading.Timer(
                delay,
                self._attempt,
                kwargs={
                    "execute_fn": execute_fn,
                    "on_success": on_success,
                    "on_failure": on_failure,
                    "semaphore": semaphore,
                    "attempt": next_attempt,
                },
            )
            timer.daemon = True
            with self._lock:
                self._active_timers.append(timer)
            timer.start()

    def cancel_pending(self) -> None:
        """Cancel all pending retry timers."""
        with self._lock:
            for timer in self._active_timers:
                timer.cancel()
            self._active_timers.clear()

    def delay_for_attempt(self, attempt: int) -> float:
        """Calculate the delay in seconds for a given attempt number."""
        return self._base_delay_s * (self._backoff_multiplier**attempt)

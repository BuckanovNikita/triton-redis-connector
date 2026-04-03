"""Aggregates typed error events with thread-safe counting and flush.

ErrorCollector tracks error counts by type and stores error instances
for periodic flush by the ErrorHandler.
"""

from __future__ import annotations

import threading
from collections import Counter
from typing import Protocol, runtime_checkable


@runtime_checkable
class _HasErrorType(Protocol):
    @property
    def error_type(self) -> str: ...


class ErrorCollector:
    """Thread-safe aggregator for pipeline error events.

    Multiple producers record errors concurrently. ErrorHandler reads
    counts and flushes periodically for summary logging and metrics.
    """

    def __init__(self) -> None:
        """Initialize empty error buffer with thread lock."""
        self._lock = threading.Lock()
        self._errors: list[object] = []
        self._counts: Counter[str] = Counter()

    def record(self, error: object) -> None:
        """Record an error event, incrementing its type counter."""
        with self._lock:
            self._errors.append(error)
            if isinstance(error, _HasErrorType):
                self._counts[error.error_type] += 1
            else:
                self._counts[type(error).__name__] += 1

    def get_summary(self) -> dict[str, int]:
        """Return error counts by error_type (snapshot under lock)."""
        with self._lock:
            return dict(self._counts)

    def flush(self) -> list[object]:
        """Return all recorded errors and reset internal state."""
        with self._lock:
            errors = self._errors
            self._errors = []
            self._counts = Counter()
        return errors

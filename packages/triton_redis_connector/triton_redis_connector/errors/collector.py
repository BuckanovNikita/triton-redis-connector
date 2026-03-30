"""Aggregates typed error events with thread-safe counting and flush.

ErrorCollector tracks error counts by type and stores error instances
for periodic flush by the ErrorHandler.
"""

from __future__ import annotations

import threading
from collections import Counter

from triton_redis_connector.errors.models import (
    ConnectorError,
    InferenceError,
    PreprocessingError,
)

type PipelineError = ConnectorError | InferenceError | PreprocessingError


class ErrorCollector:
    """Thread-safe aggregator for pipeline error events.

    Multiple producers record errors concurrently. ErrorHandler reads
    counts and flushes periodically for summary logging and metrics.
    """

    def __init__(self) -> None:  # noqa: D107
        self._lock = threading.Lock()
        self._errors: list[PipelineError] = []
        self._counts: Counter[str] = Counter()

    def record(self, error: PipelineError) -> None:
        """Record an error event, incrementing its type counter."""
        with self._lock:
            self._errors.append(error)
            self._counts[error.error_type] += 1

    def get_summary(self) -> dict[str, int]:
        """Return error counts by error_type (snapshot under lock)."""
        with self._lock:
            return dict(self._counts)

    def flush(self) -> list[PipelineError]:
        """Return all recorded errors and reset internal state."""
        with self._lock:
            errors = self._errors
            self._errors = []
            self._counts = Counter()
        return errors

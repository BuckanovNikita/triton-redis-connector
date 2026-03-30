"""Processes error events from the pipeline error queue.

Logs each error at the appropriate severity level and records it
into the ErrorCollector. Periodically logs an error summary.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

from loguru import logger

from triton_redis_connector.errors.models import InferenceError

if TYPE_CHECKING:
    from triton_redis_connector.errors.collector import ErrorCollector, PipelineError


class ErrorHandler:
    """Consumes error events, logs them, and records to ErrorCollector.

    Runs periodic summary logging in a background daemon thread at
    the configured flush interval.
    """

    def __init__(  # noqa: D107
        self,
        collector: ErrorCollector,
        flush_interval_s: float = 30.0,
    ) -> None:
        self._collector = collector
        self._flush_interval_s = flush_interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def handle(self, error: PipelineError) -> None:
        """Log an error event and record it to the collector."""
        if isinstance(error, InferenceError):
            logger.error(
                "Inference error on model {}: {} (retries={})",
                error.model_name,
                error.error_message,
                error.retry_count,
            )
        else:
            logger.warning(
                "{}: {} - {}",
                type(error).__name__,
                error.error_type,
                error.error_message,
            )
        self._collector.record(error)

    def start(self) -> None:
        """Start the periodic summary logging thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run_summary_loop,
            name="error-handler-summary",
            daemon=True,
        )
        self._thread.start()
        logger.info("ErrorHandler started (flush_interval={}s)", self._flush_interval_s)

    def stop(self) -> None:
        """Stop the periodic summary logging thread."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self._flush_interval_s * 2)
            self._thread = None
        logger.info("ErrorHandler stopped")

    def _run_summary_loop(self) -> None:
        while not self._stop.is_set():
            self._stop.wait(timeout=self._flush_interval_s)
            self._log_summary()

    def _log_summary(self) -> None:
        summary = self._collector.get_summary()
        if not summary:
            return
        total = sum(summary.values())
        logger.info(
            "Error summary (last {}s): {} total errors — {}",
            self._flush_interval_s,
            total,
            summary,
        )

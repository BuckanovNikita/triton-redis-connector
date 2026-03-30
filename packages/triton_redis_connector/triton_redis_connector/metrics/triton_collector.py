"""Polls Triton server inference statistics via gRPC.

Runs in a background daemon thread, converting Triton stats into
MetricEvent instances for downstream consumption.
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING, Any

from loguru import logger

from triton_redis_connector.metrics.collector import MetricEvent

if TYPE_CHECKING:
    from triton_redis_connector.metrics.buffer import MetricBuffer

NANOS_PER_MS = 1_000_000


class TritonMetricsCollector:
    """Background daemon that polls Triton inference statistics.

    Periodically queries Triton via get_inference_statistics(), converts
    the results to MetricEvent instances, and records them into a MetricBuffer.
    Also stores the latest stats for synchronous querying.
    """

    def __init__(  # noqa: D107
        self,
        client: Any,  # noqa: ANN401
        buffer: MetricBuffer,
        poll_interval_s: float,
    ) -> None:
        self._client = client
        self._buffer = buffer
        self._poll_interval_s = poll_interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._latest_stats: dict[str, dict[str, float]] = {}

    def start(self) -> None:
        """Start the background polling thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="triton-metrics-collector",
            daemon=True,
        )
        self._thread.start()
        logger.info("TritonMetricsCollector started (interval={}s)", self._poll_interval_s)

    def stop(self) -> None:
        """Signal the polling thread to stop and wait for it to finish."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self._poll_interval_s * 2)
            self._thread = None
        logger.info("TritonMetricsCollector stopped")

    def get_model_stats(self, model_name: str) -> dict[str, float] | None:
        """Return the latest polled stats for a given model, or None if not available."""
        with self._lock:
            return self._latest_stats.get(model_name)

    def _run(self) -> None:
        while not self._stop.is_set():
            self._stop.wait(timeout=self._poll_interval_s)
            if self._stop.is_set():
                break
            self._poll_stats()

    def _poll_stats(self) -> None:
        try:
            response = self._client.get_inference_statistics()
        except Exception:  # noqa: BLE001
            logger.warning("Failed to poll Triton inference statistics")
            return

        timestamp_ms = int(time.time() * 1000)
        model_stats = self._parse_statistics(response)

        with self._lock:
            self._latest_stats = model_stats

        for model_name, stats in model_stats.items():
            for measure, value in stats.items():
                self._buffer.record(
                    MetricEvent(
                        component_name="triton",
                        measure=measure,
                        value=value,
                        timestamp_ms=timestamp_ms,
                        tags={"model": model_name},
                    ),
                )

    def _parse_statistics(
        self,
        response: Any,  # noqa: ANN401
    ) -> dict[str, dict[str, float]]:
        """Parse Triton protobuf statistics response into plain dicts.

        Triton's gRPC statistics response is an opaque protobuf message
        with nested attributes (model_stats, inference_stats, etc.).
        We use direct attribute access since protobuf fields default
        to zero values when not set.
        """
        result: dict[str, dict[str, float]] = {}

        for model_stat in response.model_stats:
            model_name: str = model_stat.name
            inference_stats = model_stat.inference_stats
            stats: dict[str, float] = {
                "inference_count": float(model_stat.inference_count),
                "execution_count": float(model_stat.execution_count),
                "success_count": float(inference_stats.success.count),
                "success_time_ms": float(inference_stats.success.ns) / NANOS_PER_MS,
                "fail_count": float(inference_stats.fail.count),
                "queue_time_ms": float(inference_stats.queue.ns) / NANOS_PER_MS,
                "compute_infer_time_ms": float(inference_stats.compute_infer.ns) / NANOS_PER_MS,
            }
            result[model_name] = stats

        return result

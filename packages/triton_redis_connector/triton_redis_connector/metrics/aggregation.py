"""Time-window aggregation sink for MetricEvent streams.

Implements TimeWindowAggregator as a generic Sink that receives MetricEvent
instances, groups them by AggregationRule-extracted keys over sliding time
windows, and records aggregate metrics to a MetricBuffer.

Built-in rules: CountByRule, SumByRule, AvgByRule.
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING

from loguru import logger

from triton_redis_connector.metrics.collector import MetricEvent

if TYPE_CHECKING:
    from triton_redis_connector.metrics.buffer import MetricBuffer
    from triton_redis_connector.protocols import AggregationRuleProtocol


class CountByRule:
    """Counts events grouped by a key field extracted from tags."""

    def __init__(  # noqa: D107
        self,
        metric_name: str,
        window_seconds: int,
        tag_key: str,
    ) -> None:
        self._metric_name = metric_name
        self._window_seconds = window_seconds
        self._tag_key = tag_key

    @property
    def metric_name(self) -> str:  # noqa: D102
        return self._metric_name

    @property
    def window_seconds(self) -> int:  # noqa: D102
        return self._window_seconds

    def extract_key(self, event: MetricEvent) -> str:  # noqa: D102
        return event.tags.get(self._tag_key, "unknown")

    def aggregate(self, count: int) -> float:  # noqa: D102
        return float(count)


class SumByRule:
    """Sums a numeric field (MetricEvent.value) grouped by a tag key."""

    def __init__(  # noqa: D107
        self,
        metric_name: str,
        window_seconds: int,
        tag_key: str,
    ) -> None:
        self._metric_name = metric_name
        self._window_seconds = window_seconds
        self._tag_key = tag_key

    @property
    def metric_name(self) -> str:  # noqa: D102
        return self._metric_name

    @property
    def window_seconds(self) -> int:  # noqa: D102
        return self._window_seconds

    def extract_key(self, event: MetricEvent) -> str:  # noqa: D102
        return event.tags.get(self._tag_key, "unknown")

    def aggregate(self, count: int) -> float:  # noqa: D102
        return float(count)


class AvgByRule:
    """Averages a numeric field (MetricEvent.value) grouped by a tag key.

    The aggregate method receives event count; actual average is computed
    by TimeWindowAggregator using stored sum / count.
    """

    def __init__(  # noqa: D107
        self,
        metric_name: str,
        window_seconds: int,
        tag_key: str,
    ) -> None:
        self._metric_name = metric_name
        self._window_seconds = window_seconds
        self._tag_key = tag_key

    @property
    def metric_name(self) -> str:  # noqa: D102
        return self._metric_name

    @property
    def window_seconds(self) -> int:  # noqa: D102
        return self._window_seconds

    def extract_key(self, event: MetricEvent) -> str:  # noqa: D102
        return event.tags.get(self._tag_key, "unknown")

    def aggregate(self, count: int) -> float:  # noqa: D102
        return float(count)


class _WindowBucket:
    """Internal bucket tracking events within a time window.

    Stores per-key counts and value sums for aggregation.
    """

    __slots__ = ("counts", "sums", "window_start")

    def __init__(self, window_start: float) -> None:
        self.window_start = window_start
        self.counts: dict[str, int] = {}
        self.sums: dict[str, float] = {}

    def record(self, key: str, value: float) -> None:
        self.counts[key] = self.counts.get(key, 0) + 1
        self.sums[key] = self.sums.get(key, 0.0) + value


class TimeWindowAggregator:
    """Sink that aggregates MetricEvent streams over sliding time windows.

    Receives MetricEvent via __call__, groups events by rule-extracted keys,
    and when the time window expires, computes aggregates and records them
    into a MetricBuffer.

    Thread-safe: all mutable state is protected by a lock.
    """

    def __init__(  # noqa: D107
        self,
        buffer: MetricBuffer,
        rules: list[AggregationRuleProtocol[MetricEvent]],
    ) -> None:
        self._buffer = buffer
        self._rules = rules
        self._lock = threading.Lock()
        self._buckets: dict[str, _WindowBucket] = {}
        self._init_buckets()

    def __call__(self, event: MetricEvent) -> None:  # noqa: D102
        now = time.monotonic()
        with self._lock:
            for rule in self._rules:
                key = rule.extract_key(event)
                bucket = self._buckets[rule.metric_name]

                if now - bucket.window_start >= rule.window_seconds:
                    self._flush_bucket(rule, bucket, now)
                    self._buckets[rule.metric_name] = _WindowBucket(window_start=now)
                    bucket = self._buckets[rule.metric_name]

                bucket.record(key, event.value)

    def flush_all(self) -> None:
        """Force-flush all buckets regardless of window expiry."""
        now = time.monotonic()
        with self._lock:
            for rule in self._rules:
                bucket = self._buckets[rule.metric_name]
                self._flush_bucket(rule, bucket, now)
                self._buckets[rule.metric_name] = _WindowBucket(window_start=now)

    def _init_buckets(self) -> None:
        now = time.monotonic()
        for rule in self._rules:
            self._buckets[rule.metric_name] = _WindowBucket(window_start=now)

    def _flush_bucket(
        self,
        rule: AggregationRuleProtocol[MetricEvent],
        bucket: _WindowBucket,
        now: float,
    ) -> None:
        if not bucket.counts:
            return

        timestamp_ms = int(time.time() * 1000)

        for key, count in bucket.counts.items():
            value_sum = bucket.sums.get(key, 0.0)
            aggregate_value: float

            if isinstance(rule, AvgByRule) and count > 0:
                aggregate_value = value_sum / count
            elif isinstance(rule, SumByRule):
                aggregate_value = value_sum
            else:
                aggregate_value = rule.aggregate(count)

            self._buffer.record(
                MetricEvent(
                    component_name="aggregation",
                    measure=rule.metric_name,
                    value=aggregate_value,
                    timestamp_ms=timestamp_ms,
                    tags={"key": key},
                ),
            )

        logger.debug(
            "Flushed aggregation bucket {}: {} keys, window={:.1f}s",
            rule.metric_name,
            len(bucket.counts),
            now - bucket.window_start,
        )

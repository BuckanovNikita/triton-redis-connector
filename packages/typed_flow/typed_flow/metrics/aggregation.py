"""Time-window aggregation sink for MetricEvent streams.

Implements TimeWindowAggregator as a generic Sink that receives MetricEvent
instances, groups them by AggregationRule-extracted keys over sliding time
windows, and records aggregate metrics to a MetricBuffer.

Built-in rules: CountByRule, SumByRule, AvgByRule.
"""

import threading
import time

from loguru import logger

from typed_flow.metrics.buffer import MetricBuffer
from typed_flow.metrics.collector import MetricEvent
from typed_flow.protocols import AggregationRuleProtocol
from typed_flow.protocols.framework import BucketStats


class CountByRule:
    """Counts events grouped by measure name."""

    def __init__(
        self,
        metric_name: str,
        window_seconds: int,
    ) -> None:
        """Initialize with metric name and aggregation window size."""
        self._metric_name = metric_name
        self._window_seconds = window_seconds

    @property
    def metric_name(self) -> str:
        """Return the metric name for this rule."""
        return self._metric_name

    @property
    def window_seconds(self) -> int:
        """Return the time window in seconds."""
        return self._window_seconds

    def extract_key(self, event: MetricEvent) -> str:
        """Extract the grouping key from a metric event."""
        return event.measure

    def aggregate(self, stats: BucketStats) -> float:
        """Compute aggregate value from bucket statistics."""
        return float(stats.count)


class SumByRule:
    """Sums a numeric field (MetricEvent.value) grouped by measure name."""

    def __init__(
        self,
        metric_name: str,
        window_seconds: int,
    ) -> None:
        """Initialize with metric name and aggregation window size."""
        self._metric_name = metric_name
        self._window_seconds = window_seconds

    @property
    def metric_name(self) -> str:
        """Return the metric name for this rule."""
        return self._metric_name

    @property
    def window_seconds(self) -> int:
        """Return the time window in seconds."""
        return self._window_seconds

    def extract_key(self, event: MetricEvent) -> str:
        """Extract the grouping key from a metric event."""
        return event.measure

    def aggregate(self, stats: BucketStats) -> float:
        """Compute aggregate value from bucket statistics."""
        return stats.value_sum


class AvgByRule:
    """Averages a numeric field (MetricEvent.value) grouped by measure name."""

    def __init__(
        self,
        metric_name: str,
        window_seconds: int,
    ) -> None:
        """Initialize with metric name and aggregation window size."""
        self._metric_name = metric_name
        self._window_seconds = window_seconds

    @property
    def metric_name(self) -> str:
        """Return the metric name for this rule."""
        return self._metric_name

    @property
    def window_seconds(self) -> int:
        """Return the time window in seconds."""
        return self._window_seconds

    def extract_key(self, event: MetricEvent) -> str:
        """Extract the grouping key from a metric event."""
        return event.measure

    def aggregate(self, stats: BucketStats) -> float:
        """Compute aggregate value from bucket statistics."""
        return stats.value_sum / stats.count if stats.count > 0 else 0.0


class MinByRule:
    """Tracks the minimum value grouped by measure name."""

    def __init__(
        self,
        metric_name: str,
        window_seconds: int,
    ) -> None:
        """Initialize with metric name and aggregation window size."""
        self._metric_name = metric_name
        self._window_seconds = window_seconds

    @property
    def metric_name(self) -> str:
        """Return the metric name for this rule."""
        return self._metric_name

    @property
    def window_seconds(self) -> int:
        """Return the time window in seconds."""
        return self._window_seconds

    def extract_key(self, event: MetricEvent) -> str:
        """Extract the grouping key from a metric event."""
        return event.measure

    def aggregate(self, stats: BucketStats) -> float:
        """Compute aggregate value from bucket statistics."""
        return stats.value_min


class MaxByRule:
    """Tracks the maximum value grouped by measure name."""

    def __init__(
        self,
        metric_name: str,
        window_seconds: int,
    ) -> None:
        """Initialize with metric name and aggregation window size."""
        self._metric_name = metric_name
        self._window_seconds = window_seconds

    @property
    def metric_name(self) -> str:
        """Return the metric name for this rule."""
        return self._metric_name

    @property
    def window_seconds(self) -> int:
        """Return the time window in seconds."""
        return self._window_seconds

    def extract_key(self, event: MetricEvent) -> str:
        """Extract the grouping key from a metric event."""
        return event.measure

    def aggregate(self, stats: BucketStats) -> float:
        """Compute aggregate value from bucket statistics."""
        return stats.value_max


class _WindowBucket:
    """Internal bucket tracking events within a time window.

    Stores per-key counts, value sums, minimums, and maximums for aggregation.
    """

    __slots__ = ("counts", "maxs", "mins", "sums", "window_start")

    def __init__(self, window_start: float) -> None:
        self.window_start = window_start
        self.counts: dict[str, int] = {}
        self.sums: dict[str, float] = {}
        self.mins: dict[str, float] = {}
        self.maxs: dict[str, float] = {}

    def record(self, key: str, value: float) -> None:
        self.counts[key] = self.counts.get(key, 0) + 1
        self.sums[key] = self.sums.get(key, 0.0) + value
        if key not in self.mins or value < self.mins[key]:
            self.mins[key] = value
        if key not in self.maxs or value > self.maxs[key]:
            self.maxs[key] = value


class TimeWindowAggregator:
    """Sink that aggregates MetricEvent streams over sliding time windows.

    Receives MetricEvent via __call__, groups events by rule-extracted keys,
    and when the time window expires, computes aggregates and records them
    into a MetricBuffer.

    Thread-safe: all mutable state is protected by a lock.
    """

    def __init__(
        self,
        buffer: MetricBuffer,
        rules: list[AggregationRuleProtocol[MetricEvent]],
    ) -> None:
        """Initialize aggregator with metric buffer and aggregation rules."""
        self._buffer = buffer
        self._rules = rules
        self._lock = threading.Lock()
        self._buckets: dict[str, _WindowBucket] = {}
        self._init_buckets()

    def __call__(self, event: MetricEvent) -> None:
        """Process a metric event through all aggregation rules."""
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

        for key, count in bucket.counts.items():
            stats = BucketStats(
                count=count,
                value_sum=bucket.sums.get(key, 0.0),
                value_min=bucket.mins.get(key, 0.0),
                value_max=bucket.maxs.get(key, 0.0),
            )
            aggregate_value = rule.aggregate(stats)

            self._buffer.record(
                MetricEvent(
                    component_name="aggregation",
                    measure=rule.metric_name,
                    value=aggregate_value,
                ),
            )

        logger.debug(
            "Flushed aggregation bucket {}: {} keys, window={:.1f}s",
            rule.metric_name,
            len(bucket.counts),
            now - bucket.window_start,
        )

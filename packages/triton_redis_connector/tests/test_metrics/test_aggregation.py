"""Tests for time-window aggregation metrics."""

from __future__ import annotations

import threading
import time
from unittest.mock import patch

import pytest
from triton_redis_connector.metrics.aggregation import (
    AvgByRule,
    CountByRule,
    SumByRule,
    TimeWindowAggregator,
)
from triton_redis_connector.metrics.buffer import MetricBuffer
from triton_redis_connector.metrics.collector import MetricEvent


def _make_event(
    measure: str = "test",
    value: float = 1.0,
    tags: dict[str, str] | None = None,
) -> MetricEvent:
    return MetricEvent(
        component_name="test",
        measure=measure,
        value=value,
        timestamp_ms=int(time.time() * 1000),
        tags=tags or {},
    )


class TestCountByRule:
    def test_extract_key_from_tags(self) -> None:
        rule = CountByRule(metric_name="request_count", window_seconds=60, tag_key="model")
        event = _make_event(tags={"model": "yolo11s"})
        assert rule.extract_key(event) == "yolo11s"

    def test_extract_key_missing_returns_unknown(self) -> None:
        rule = CountByRule(metric_name="request_count", window_seconds=60, tag_key="model")
        event = _make_event(tags={})
        assert rule.extract_key(event) == "unknown"


class TestSumByRule:
    def test_extract_key_from_tags(self) -> None:
        rule = SumByRule(metric_name="bytes_total", window_seconds=120, tag_key="stream")
        event = _make_event(tags={"stream": "camera_1"})
        assert rule.extract_key(event) == "camera_1"

    def test_extract_key_missing_returns_unknown(self) -> None:
        rule = SumByRule(metric_name="bytes_total", window_seconds=120, tag_key="stream")
        event = _make_event(tags={"other": "value"})
        assert rule.extract_key(event) == "unknown"


class TestAvgByRule:
    def test_extract_key_from_tags(self) -> None:
        rule = AvgByRule(metric_name="avg_latency", window_seconds=60, tag_key="model")
        event = _make_event(tags={"model": "yolo26s"})
        assert rule.extract_key(event) == "yolo26s"

    def test_aggregate_returns_count_as_float(self) -> None:
        rule = AvgByRule(metric_name="avg_latency", window_seconds=60, tag_key="model")
        assert rule.aggregate(5) == 5.0


class TestTimeWindowAggregator:
    def test_records_events(self) -> None:
        buffer = MetricBuffer()
        rule = CountByRule(metric_name="req_count", window_seconds=3600, tag_key="model")
        aggregator = TimeWindowAggregator(buffer=buffer, rules=[rule])

        event = _make_event(tags={"model": "yolo11s"})
        aggregator(event)

        aggregator.flush_all()

        flushed = buffer.flush()
        assert len(flushed) == 1
        assert flushed[0].measure == "req_count"
        assert flushed[0].value == 1.0

    def test_groups_by_key(self) -> None:
        buffer = MetricBuffer()
        rule = CountByRule(metric_name="req_count", window_seconds=3600, tag_key="model")
        aggregator = TimeWindowAggregator(buffer=buffer, rules=[rule])

        aggregator(_make_event(tags={"model": "yolo11s"}))
        aggregator(_make_event(tags={"model": "yolo11s"}))
        aggregator(_make_event(tags={"model": "yolo26s"}))

        aggregator.flush_all()
        flushed = buffer.flush()

        assert len(flushed) == 2
        values_by_key = {e.tags["key"]: e.value for e in flushed}
        assert values_by_key["yolo11s"] == 2.0
        assert values_by_key["yolo26s"] == 1.0

    def test_window_expiry_triggers_flush(self) -> None:
        buffer = MetricBuffer()
        rule = CountByRule(metric_name="req_count", window_seconds=1, tag_key="model")
        aggregator = TimeWindowAggregator(buffer=buffer, rules=[rule])

        aggregator(_make_event(tags={"model": "yolo11s"}))

        with patch("triton_redis_connector.metrics.aggregation.time") as mock_time:
            mock_time.monotonic.return_value = time.monotonic() + 2.0
            mock_time.time.return_value = time.time()
            aggregator(_make_event(tags={"model": "yolo11s"}))

        flushed = buffer.flush()
        assert len(flushed) >= 1
        count_metrics = [e for e in flushed if e.measure == "req_count"]
        assert len(count_metrics) >= 1

    def test_sum_by_rule_aggregation(self) -> None:
        buffer = MetricBuffer()
        rule = SumByRule(metric_name="total_bytes", window_seconds=3600, tag_key="stream")
        aggregator = TimeWindowAggregator(buffer=buffer, rules=[rule])

        aggregator(_make_event(value=100.0, tags={"stream": "cam1"}))
        aggregator(_make_event(value=200.0, tags={"stream": "cam1"}))
        aggregator(_make_event(value=50.0, tags={"stream": "cam2"}))

        aggregator.flush_all()
        flushed = buffer.flush()

        assert len(flushed) == 2
        values_by_key = {e.tags["key"]: e.value for e in flushed}
        assert values_by_key["cam1"] == pytest.approx(300.0)
        assert values_by_key["cam2"] == pytest.approx(50.0)

    def test_avg_by_rule_aggregation(self) -> None:
        buffer = MetricBuffer()
        rule = AvgByRule(metric_name="avg_latency", window_seconds=3600, tag_key="model")
        aggregator = TimeWindowAggregator(buffer=buffer, rules=[rule])

        aggregator(_make_event(value=10.0, tags={"model": "yolo11s"}))
        aggregator(_make_event(value=20.0, tags={"model": "yolo11s"}))
        aggregator(_make_event(value=30.0, tags={"model": "yolo11s"}))

        aggregator.flush_all()
        flushed = buffer.flush()

        assert len(flushed) == 1
        assert flushed[0].value == pytest.approx(20.0)

    def test_multiple_rules(self) -> None:
        buffer = MetricBuffer()
        count_rule = CountByRule(metric_name="count", window_seconds=3600, tag_key="model")
        sum_rule = SumByRule(metric_name="sum", window_seconds=3600, tag_key="model")
        aggregator = TimeWindowAggregator(buffer=buffer, rules=[count_rule, sum_rule])

        aggregator(_make_event(value=10.0, tags={"model": "yolo11s"}))
        aggregator(_make_event(value=20.0, tags={"model": "yolo11s"}))

        aggregator.flush_all()
        flushed = buffer.flush()

        measures = {e.measure: e.value for e in flushed}
        assert measures["count"] == 2.0
        assert measures["sum"] == pytest.approx(30.0)

    def test_flush_all_empties_buckets(self) -> None:
        buffer = MetricBuffer()
        rule = CountByRule(metric_name="count", window_seconds=3600, tag_key="model")
        aggregator = TimeWindowAggregator(buffer=buffer, rules=[rule])

        aggregator(_make_event(tags={"model": "yolo11s"}))
        aggregator.flush_all()
        buffer.flush()

        aggregator.flush_all()
        flushed = buffer.flush()
        assert len(flushed) == 0

    def test_thread_safety(self) -> None:
        buffer = MetricBuffer()
        rule = CountByRule(metric_name="count", window_seconds=3600, tag_key="model")
        aggregator = TimeWindowAggregator(buffer=buffer, rules=[rule])

        errors: list[Exception] = []
        events_per_thread = 100
        num_threads = 4

        def writer(model_name: str) -> None:
            try:
                for _ in range(events_per_thread):
                    aggregator(_make_event(tags={"model": model_name}))
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=writer, args=(f"model_{i}",)) for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

        aggregator.flush_all()
        flushed = buffer.flush()

        total_count = sum(e.value for e in flushed)
        assert total_count == num_threads * events_per_thread

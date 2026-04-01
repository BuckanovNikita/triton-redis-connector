"""Tests for MetricAggregator, Tracked/Counter/Gauge markers."""

from __future__ import annotations

import threading
from typing import Annotated

from triton_redis_connector.engine.metric_aggregator import (
    Counter,
    Gauge,
    MetricAggregator,
    NodeMetricSnapshot,
    Tracked,
    extract_metric_markers,
    extract_tracked,
)


class TestMetricAggregator:
    def test_snapshot_zero_initially(self) -> None:
        agg = MetricAggregator()
        snap = agg.snapshot()
        assert snap == NodeMetricSnapshot(
            call_count=0,
            error_count=0,
            empty_count=0,
            total_processing_ms=0.0,
        )

    def test_record_call_increments_count(self) -> None:
        agg = MetricAggregator()
        agg.record_call(10.0)
        agg.record_call(20.0)
        snap = agg.snapshot()
        assert snap.call_count == 2
        assert snap.total_processing_ms == 30.0

    def test_record_call_tracks_errors(self) -> None:
        agg = MetricAggregator()
        agg.record_call(5.0, is_error=True)
        agg.record_call(5.0)
        snap = agg.snapshot()
        assert snap.error_count == 1
        assert snap.call_count == 2

    def test_record_call_tracks_empties(self) -> None:
        agg = MetricAggregator()
        agg.record_call(1.0, is_empty=True)
        agg.record_call(1.0, is_empty=True)
        agg.record_call(1.0)
        snap = agg.snapshot()
        assert snap.empty_count == 2

    def test_interval_calls_returns_delta(self) -> None:
        agg = MetricAggregator()
        agg.record_call(1.0)
        agg.record_call(1.0)
        assert agg.interval_calls() == 2
        agg.record_call(1.0)
        assert agg.interval_calls() == 1
        assert agg.interval_calls() == 0

    def test_thread_safety(self) -> None:
        agg = MetricAggregator()
        iterations = 1000

        def worker() -> None:
            for _ in range(iterations):
                agg.record_call(0.1)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        snap = agg.snapshot()
        assert snap.call_count == 4 * iterations


class TestTracked:
    def test_extract_tracked_from_annotated(self) -> None:
        from typing import Annotated

        annotation = Annotated[int, Tracked("my-node")]
        result = extract_tracked(annotation)
        assert result is not None
        assert result.name == "my-node"

    def test_extract_tracked_returns_none_for_plain_type(self) -> None:
        assert extract_tracked(int) is None

    def test_extract_tracked_returns_none_without_marker(self) -> None:
        from typing import Annotated

        annotation = Annotated[int, "some-other-metadata"]
        assert extract_tracked(annotation) is None

    def test_tracked_default_name(self) -> None:
        t = Tracked()
        assert t.name == ""


class TestCounterGauge:
    def test_extract_counter_from_annotated(self) -> None:
        annotation = Annotated[int, Counter("my_count")]
        markers = extract_metric_markers(annotation)
        assert markers == [Counter("my_count")]

    def test_extract_gauge_from_annotated(self) -> None:
        annotation = Annotated[float, Gauge("my_gauge")]
        markers = extract_metric_markers(annotation)
        assert markers == [Gauge("my_gauge")]

    def test_extract_mixed_markers(self) -> None:
        annotation = Annotated[int, Counter("c"), Gauge("g")]
        markers = extract_metric_markers(annotation)
        assert len(markers) == 2
        assert Counter("c") in markers
        assert Gauge("g") in markers

    def test_extract_no_markers_from_plain_type(self) -> None:
        assert extract_metric_markers(int) == []

    def test_extract_no_markers_from_annotated_without_metric(self) -> None:
        annotation = Annotated[int, "just-a-string"]
        assert extract_metric_markers(annotation) == []

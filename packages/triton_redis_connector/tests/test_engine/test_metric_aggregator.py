"""Tests for MetricAggregator, Tracked marker, and InstrumentedNode protocol."""

from __future__ import annotations

import threading

from triton_redis_connector.engine.metric_aggregator import (
    InstrumentedNode,
    MetricAggregator,
    NodeMetricSnapshot,
    Tracked,
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


class TestInstrumentedNode:
    def test_wrapper_satisfies_protocol(self) -> None:
        from triton_redis_connector.engine.wrappers import DetachedThreadProcessor

        wrapper = DetachedThreadProcessor(lambda x: x, aggregator=MetricAggregator())
        assert isinstance(wrapper, InstrumentedNode)

    def test_wrapper_without_aggregator_still_satisfies(self) -> None:
        from triton_redis_connector.engine.wrappers import DetachedThreadProcessor

        wrapper = DetachedThreadProcessor(lambda x: x)
        assert isinstance(wrapper, InstrumentedNode)

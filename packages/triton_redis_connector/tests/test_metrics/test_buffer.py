"""Tests for MetricBuffer thread safety and flush semantics."""

from __future__ import annotations

import threading
import time

from triton_redis_connector.metrics.buffer import MetricBuffer
from triton_redis_connector.metrics.collector import MetricEvent


def _make_event(measure: str = "test", value: float = 1.0) -> MetricEvent:
    return MetricEvent(
        component_name="test-component",
        measure=measure,
        value=value,
    )


class TestMetricBuffer:
    def test_record_and_flush_returns_events(self) -> None:
        buffer = MetricBuffer()
        event = _make_event()
        buffer.record(event)
        flushed = buffer.flush()
        assert len(flushed) == 1
        assert flushed[0] is event

    def test_flush_clears_buffer(self) -> None:
        buffer = MetricBuffer()
        buffer.record(_make_event())
        buffer.record(_make_event())
        first = buffer.flush()
        assert len(first) == 2
        second = buffer.flush()
        assert len(second) == 0

    def test_empty_flush_returns_empty_list(self) -> None:
        buffer = MetricBuffer()
        assert buffer.flush() == []

    def test_multiple_records_accumulate(self) -> None:
        buffer = MetricBuffer()
        for i in range(10):
            buffer.record(_make_event(value=float(i)))
        flushed = buffer.flush()
        assert len(flushed) == 10
        assert [e.value for e in flushed] == [float(i) for i in range(10)]

    def test_concurrent_record_and_flush(self) -> None:
        buffer = MetricBuffer()
        events_per_thread = 100
        num_threads = 8
        all_flushed: list[MetricEvent] = []
        flush_lock = threading.Lock()

        def producer(thread_id: int) -> None:
            for i in range(events_per_thread):
                buffer.record(
                    _make_event(
                        measure=f"thread-{thread_id}",
                        value=float(i),
                    ),
                )

        def consumer() -> None:
            for _ in range(50):
                batch = buffer.flush()
                if batch:
                    with flush_lock:
                        all_flushed.extend(batch)
                time.sleep(0.001)

        threads = [threading.Thread(target=producer, args=(i,)) for i in range(num_threads)]
        consumer_thread = threading.Thread(target=consumer)

        consumer_thread.start()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        consumer_thread.join()

        remaining = buffer.flush()
        all_flushed.extend(remaining)
        assert len(all_flushed) == num_threads * events_per_thread

    def test_flush_returns_events_in_insertion_order(self) -> None:
        buffer = MetricBuffer()
        buffer.record(_make_event(measure="first"))
        buffer.record(_make_event(measure="second"))
        buffer.record(_make_event(measure="third"))
        flushed = buffer.flush()
        assert [e.measure for e in flushed] == ["first", "second", "third"]

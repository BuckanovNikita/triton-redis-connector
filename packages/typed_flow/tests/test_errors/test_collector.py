"""Tests for ErrorCollector thread-safe counting and flush."""

import threading
import time

from typed_flow.errors.collector import ErrorCollector
from typed_flow.errors.models import ConnectorError, ProcessingError


def _connector_error(error_type: str = "ConnectionError") -> ConnectorError:
    return ConnectorError(
        node_name="test_source",
        connector_name="redis-source",
        stream_name="stream:input",
        message_id=None,
        error_type=error_type,
        error_message="test error",
        timestamp_ms=int(time.time() * 1000),
    )


def _processing_error(error_type: str = "ProcessingFailed") -> ProcessingError:
    return ProcessingError(
        node_name="processor:test",
        error_type=error_type,
        error_message="test error",
        timestamp_ms=int(time.time() * 1000),
        event_type="TestEvent",
    )


class TestErrorCollector:
    def test_record_and_get_summary(self) -> None:
        collector = ErrorCollector()
        collector.record(_connector_error())
        collector.record(_connector_error())
        collector.record(_processing_error())
        summary = collector.get_summary()
        assert summary == {"ConnectionError": 2, "ProcessingFailed": 1}

    def test_empty_summary(self) -> None:
        collector = ErrorCollector()
        assert collector.get_summary() == {}

    def test_flush_returns_all_errors(self) -> None:
        collector = ErrorCollector()
        collector.record(_connector_error())
        collector.record(_processing_error())
        collector.record(_processing_error("DecodeError"))
        flushed = collector.flush()
        assert len(flushed) == 3

    def test_flush_clears_state(self) -> None:
        collector = ErrorCollector()
        collector.record(_connector_error())
        collector.flush()
        assert collector.get_summary() == {}
        assert collector.flush() == []

    def test_counts_by_error_type(self) -> None:
        collector = ErrorCollector()
        collector.record(_connector_error("ConnectionError"))
        collector.record(_connector_error("TimeoutError"))
        collector.record(_connector_error("ConnectionError"))
        summary = collector.get_summary()
        assert summary["ConnectionError"] == 2
        assert summary["TimeoutError"] == 1

    def test_mixed_error_types(self) -> None:
        collector = ErrorCollector()
        collector.record(_connector_error("ConnectionError"))
        collector.record(_processing_error("ProcessingFailed"))
        collector.record(_processing_error("DecodeError"))
        summary = collector.get_summary()
        assert len(summary) == 3
        assert summary["ConnectionError"] == 1
        assert summary["ProcessingFailed"] == 1
        assert summary["DecodeError"] == 1

    def test_concurrent_record(self) -> None:
        collector = ErrorCollector()
        errors_per_thread = 50
        num_threads = 8

        def producer(thread_id: int) -> None:
            for _ in range(errors_per_thread):
                collector.record(_connector_error(f"Error-{thread_id}"))

        threads = [threading.Thread(target=producer, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        flushed = collector.flush()
        assert len(flushed) == num_threads * errors_per_thread

    def test_flush_preserves_insertion_order(self) -> None:
        collector = ErrorCollector()
        collector.record(_connector_error("First"))
        collector.record(_processing_error("Second"))
        collector.record(_connector_error("Third"))
        flushed = collector.flush()
        types = [getattr(e, "error_type", None) for e in flushed]
        assert types == ["First", "Second", "Third"]

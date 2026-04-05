"""Tests for ErrorHandler logging and ErrorCollector recording."""

import time

from typed_flow.errors.collector import ErrorCollector
from typed_flow.errors.handler import ErrorHandler
from typed_flow.errors.models import ConnectorError, ProcessingError


def _connector_error() -> ConnectorError:
    return ConnectorError(
        node_name="test_source",
        connector_name="redis-source",
        stream_name="stream:input",
        message_id=None,
        error_type="ConnectionError",
        error_message="Connection refused",
        timestamp_ms=int(time.time() * 1000),
    )


def _processing_error() -> ProcessingError:
    return ProcessingError(
        node_name="processor:test",
        error_type="ProcessingFailed",
        error_message="Unexpected failure",
        timestamp_ms=int(time.time() * 1000),
        event_type="TestEvent",
    )


class TestErrorHandler:
    def test_handle_records_to_collector(self) -> None:
        collector = ErrorCollector()
        handler = ErrorHandler(collector=collector)
        handler.handle(_connector_error())
        summary = collector.get_summary()
        assert summary["ConnectionError"] == 1

    def test_handle_processing_error_records(self) -> None:
        collector = ErrorCollector()
        handler = ErrorHandler(collector=collector)
        handler.handle(_processing_error())
        summary = collector.get_summary()
        assert summary["ProcessingFailed"] == 1

    def test_handle_multiple_errors(self) -> None:
        collector = ErrorCollector()
        handler = ErrorHandler(collector=collector)
        handler.handle(_connector_error())
        handler.handle(_processing_error())
        handler.handle(_connector_error())
        flushed = collector.flush()
        assert len(flushed) == 3

    def test_start_and_stop(self) -> None:
        collector = ErrorCollector()
        handler = ErrorHandler(collector=collector, flush_interval_s=0.05)
        handler.start()
        time.sleep(0.02)
        handler.stop()

    def test_start_is_idempotent(self) -> None:
        collector = ErrorCollector()
        handler = ErrorHandler(collector=collector, flush_interval_s=0.05)
        handler.start()
        handler.start()
        handler.stop()

    def test_periodic_summary_runs(self) -> None:
        collector = ErrorCollector()
        handler = ErrorHandler(collector=collector, flush_interval_s=0.05)
        handler.handle(_connector_error())
        handler.start()
        time.sleep(0.15)
        handler.stop()

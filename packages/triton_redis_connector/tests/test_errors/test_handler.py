"""Tests for ErrorHandler logging and ErrorCollector recording."""

from __future__ import annotations

import time

from triton_redis_connector.errors.models import (
    InferenceError,
    PreprocessingError,
)
from typed_flow.errors.collector import ErrorCollector
from typed_flow.errors.handler import ErrorHandler
from typed_flow.errors.models import ConnectorError


def _connector_error() -> ConnectorError:
    return ConnectorError(
        connector_name="redis-source",
        stream_name="stream:input",
        message_id=None,
        error_type="ConnectionError",
        error_message="Connection refused",
        timestamp_ms=int(time.time() * 1000),
    )


def _inference_error() -> InferenceError:
    return InferenceError(
        model_name="yolo11s",
        model_version="1",
        error_type="InferenceServerException",
        error_message="Model not found",
        retry_count=3,
        timestamp_ms=int(time.time() * 1000),
    )


def _preprocessing_error() -> PreprocessingError:
    return PreprocessingError(
        message_id="123-0",
        decoder_type="opencv",
        error_type="DecodeError",
        error_message="Invalid JPEG header",
        timestamp_ms=int(time.time() * 1000),
    )


class TestErrorHandler:
    def test_handle_records_to_collector(self) -> None:
        collector = ErrorCollector()
        handler = ErrorHandler(collector=collector)
        handler.handle(_connector_error())
        summary = collector.get_summary()
        assert summary["ConnectionError"] == 1

    def test_handle_inference_error_records(self) -> None:
        collector = ErrorCollector()
        handler = ErrorHandler(collector=collector)
        handler.handle(_inference_error())
        summary = collector.get_summary()
        assert summary["InferenceServerException"] == 1

    def test_handle_preprocessing_error_records(self) -> None:
        collector = ErrorCollector()
        handler = ErrorHandler(collector=collector)
        handler.handle(_preprocessing_error())
        summary = collector.get_summary()
        assert summary["DecodeError"] == 1

    def test_handle_multiple_errors(self) -> None:
        collector = ErrorCollector()
        handler = ErrorHandler(collector=collector)
        handler.handle(_connector_error())
        handler.handle(_inference_error())
        handler.handle(_preprocessing_error())
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

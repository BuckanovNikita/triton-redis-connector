"""Tests for TritonMetricsCollector stats polling and MetricEvent conversion."""

from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import MagicMock

from triton_redis_connector.metrics.buffer import MetricBuffer
from triton_redis_connector.metrics.triton_collector import TritonMetricsCollector


def _build_triton_stats_response(  # noqa: PLR0913
    model_name: str = "yolo11s",
    inference_count: int = 100,
    execution_count: int = 50,
    success_count: int = 95,
    success_ns: int = 5_000_000_000,
    fail_count: int = 5,
    queue_ns: int = 1_000_000_000,
    compute_infer_ns: int = 3_000_000_000,
) -> SimpleNamespace:
    return SimpleNamespace(
        model_stats=[
            SimpleNamespace(
                name=model_name,
                inference_count=inference_count,
                execution_count=execution_count,
                inference_stats=SimpleNamespace(
                    success=SimpleNamespace(count=success_count, ns=success_ns),
                    fail=SimpleNamespace(count=fail_count, ns=0),
                    queue=SimpleNamespace(count=0, ns=queue_ns),
                    compute_infer=SimpleNamespace(count=0, ns=compute_infer_ns),
                ),
            ),
        ],
    )


class TestTritonMetricsCollector:
    def test_poll_records_metrics_to_buffer(self) -> None:
        buffer = MetricBuffer()
        client = MagicMock()
        client.get_inference_statistics.return_value = _build_triton_stats_response()
        collector = TritonMetricsCollector(client=client, buffer=buffer, poll_interval_s=10.0)

        collector._poll_stats()

        events = buffer.flush()
        measures = {e.measure for e in events}
        assert "inference_count" in measures
        assert "execution_count" in measures
        assert "success_count" in measures
        assert "fail_count" in measures
        assert "queue_time_ms" in measures
        assert "compute_infer_time_ms" in measures

    def test_poll_converts_nanoseconds_to_milliseconds(self) -> None:
        buffer = MetricBuffer()
        client = MagicMock()
        client.get_inference_statistics.return_value = _build_triton_stats_response(
            success_ns=2_000_000_000,
        )
        collector = TritonMetricsCollector(client=client, buffer=buffer, poll_interval_s=10.0)

        collector._poll_stats()

        events = buffer.flush()
        success_time_event = next(e for e in events if e.measure == "success_time_ms")
        assert success_time_event.value == 2000.0

    def test_poll_tags_events_with_model_name(self) -> None:
        buffer = MetricBuffer()
        client = MagicMock()
        client.get_inference_statistics.return_value = _build_triton_stats_response(
            model_name="custom-model",
        )
        collector = TritonMetricsCollector(client=client, buffer=buffer, poll_interval_s=10.0)

        collector._poll_stats()

        events = buffer.flush()
        assert all(e.tags.get("model") == "custom-model" for e in events)

    def test_get_model_stats_returns_none_before_poll(self) -> None:
        buffer = MetricBuffer()
        client = MagicMock()
        collector = TritonMetricsCollector(client=client, buffer=buffer, poll_interval_s=10.0)

        assert collector.get_model_stats("yolo11s") is None

    def test_get_model_stats_returns_latest_after_poll(self) -> None:
        buffer = MetricBuffer()
        client = MagicMock()
        client.get_inference_statistics.return_value = _build_triton_stats_response(
            model_name="yolo11s",
            inference_count=42,
        )
        collector = TritonMetricsCollector(client=client, buffer=buffer, poll_interval_s=10.0)

        collector._poll_stats()

        stats = collector.get_model_stats("yolo11s")
        assert stats is not None
        assert stats["inference_count"] == 42.0

    def test_poll_handles_client_exception(self) -> None:
        buffer = MetricBuffer()
        client = MagicMock()
        client.get_inference_statistics.side_effect = RuntimeError("connection failed")
        collector = TritonMetricsCollector(client=client, buffer=buffer, poll_interval_s=10.0)

        collector._poll_stats()

        assert buffer.flush() == []

    def test_start_and_stop(self) -> None:
        buffer = MetricBuffer()
        client = MagicMock()
        client.get_inference_statistics.return_value = _build_triton_stats_response()
        collector = TritonMetricsCollector(client=client, buffer=buffer, poll_interval_s=0.05)

        collector.start()
        time.sleep(0.15)
        collector.stop()

    def test_start_is_idempotent(self) -> None:
        buffer = MetricBuffer()
        client = MagicMock()
        client.get_inference_statistics.return_value = _build_triton_stats_response()
        collector = TritonMetricsCollector(client=client, buffer=buffer, poll_interval_s=0.05)

        collector.start()
        collector.start()
        collector.stop()

    def test_multiple_models_in_response(self) -> None:
        buffer = MetricBuffer()
        client = MagicMock()
        response = SimpleNamespace(
            model_stats=[
                SimpleNamespace(
                    name="model_a",
                    inference_count=10,
                    execution_count=5,
                    inference_stats=SimpleNamespace(
                        success=SimpleNamespace(count=10, ns=1_000_000),
                        fail=SimpleNamespace(count=0, ns=0),
                        queue=SimpleNamespace(count=0, ns=500_000),
                        compute_infer=SimpleNamespace(count=0, ns=800_000),
                    ),
                ),
                SimpleNamespace(
                    name="model_b",
                    inference_count=20,
                    execution_count=10,
                    inference_stats=SimpleNamespace(
                        success=SimpleNamespace(count=20, ns=2_000_000),
                        fail=SimpleNamespace(count=0, ns=0),
                        queue=SimpleNamespace(count=0, ns=1_000_000),
                        compute_infer=SimpleNamespace(count=0, ns=1_500_000),
                    ),
                ),
            ],
        )
        client.get_inference_statistics.return_value = response
        collector = TritonMetricsCollector(client=client, buffer=buffer, poll_interval_s=10.0)

        collector._poll_stats()

        events = buffer.flush()
        model_a_events = [e for e in events if e.tags.get("model") == "model_a"]
        model_b_events = [e for e in events if e.tags.get("model") == "model_b"]
        assert len(model_a_events) == 7
        assert len(model_b_events) == 7

"""Tests for MetricEmitter resend behavior: prefix, accumulation, periodic flush."""

import time
from unittest.mock import MagicMock

from typed_flow.metrics.buffer import MetricBuffer
from typed_flow.metrics.collector import MetricEvent
from typed_flow.metrics.emitter import MetricEmitter, _sanitize_prefix


def _make_event(
    measure: str = "test_measure",
    value: float = 1.0,
    component: str = "test-component",
) -> MetricEvent:
    return MetricEvent(component_name=component, measure=measure, value=value)


class TestSanitizePrefix:
    def test_replaces_hyphens_with_underscores(self) -> None:
        assert _sanitize_prefix("my-app-name") == "my_app_name"

    def test_replaces_dots_with_underscores(self) -> None:
        assert _sanitize_prefix("host.name.com") == "host_name_com"

    def test_strips_leading_trailing_underscores(self) -> None:
        assert _sanitize_prefix("--app--") == "app"

    def test_preserves_alphanumeric(self) -> None:
        assert _sanitize_prefix("myApp123") == "myApp123"

    def test_complex_hostname(self) -> None:
        assert _sanitize_prefix("worker-01.prod.us-east") == "worker_01_prod_us_east"


class TestMetricEmitterComponentName:
    def test_sets_component_name_from_app_name(self) -> None:
        buffer = MetricBuffer()
        client: MagicMock = MagicMock()
        emitter = MetricEmitter(
            buffer=buffer,
            client=client,
            stream="test:metrics",
            interval_s=0.05,
            app_name="detector",
        )

        buffer.record(_make_event(measure="images_processed", value=5.0))
        emitter.start()
        time.sleep(0.15)
        emitter.stop()

        assert client.xadd.called
        fields = client.xadd.call_args.kwargs["fields"]
        assert fields["measure"] == "images_processed"
        assert fields["component_name"] == "detector"

    def test_hostname_fallback_when_app_name_empty(self) -> None:
        import socket

        buffer = MetricBuffer()
        client: MagicMock = MagicMock()
        emitter = MetricEmitter(
            buffer=buffer,
            client=client,
            stream="test:metrics",
            interval_s=0.05,
            app_name="",
        )

        expected_name = _sanitize_prefix(socket.gethostname())
        assert emitter.component_name == expected_name

        buffer.record(_make_event(measure="counter"))
        emitter.start()
        time.sleep(0.15)
        emitter.stop()

        assert client.xadd.called
        fields = client.xadd.call_args.kwargs["fields"]
        assert fields["measure"] == "counter"
        assert fields["component_name"] == expected_name

    def test_hostname_fallback_when_app_name_not_provided(self) -> None:
        import socket

        buffer = MetricBuffer()
        client: MagicMock = MagicMock()
        emitter = MetricEmitter(
            buffer=buffer,
            client=client,
            stream="metrics",
            interval_s=0.05,
        )

        expected_name = _sanitize_prefix(socket.gethostname())
        assert emitter.component_name == expected_name


class TestMetricEmitterAccumulation:
    def test_counters_accumulate_across_flushes(self) -> None:
        buffer = MetricBuffer()
        client: MagicMock = MagicMock()
        emitter = MetricEmitter(
            buffer=buffer,
            client=client,
            stream="test:metrics",
            interval_s=0.05,
            app_name="app",
        )

        buffer.record(_make_event(measure="requests", value=3.0))
        emitter.start()
        time.sleep(0.12)

        buffer.record(_make_event(measure="requests", value=7.0))
        time.sleep(0.12)
        emitter.stop()

        xadd_calls = client.xadd.call_args_list
        requests_values = [
            float(c.kwargs["fields"]["value"]) for c in xadd_calls if c.kwargs["fields"]["measure"] == "requests"
        ]

        assert len(requests_values) >= 2
        assert requests_values[0] == 3.0
        assert requests_values[1] == 7.0

    def test_multiple_distinct_measures_emitted_separately(self) -> None:
        buffer = MetricBuffer()
        client: MagicMock = MagicMock()
        emitter = MetricEmitter(
            buffer=buffer,
            client=client,
            stream="test:metrics",
            interval_s=0.05,
            app_name="svc",
        )

        buffer.record(_make_event(measure="latency_ms", value=12.5))
        buffer.record(_make_event(measure="throughput", value=100.0))

        emitter.start()
        time.sleep(0.15)
        emitter.stop()

        xadd_calls = client.xadd.call_args_list
        measures_emitted = {c.kwargs["fields"]["measure"] for c in xadd_calls}
        assert "latency_ms" in measures_emitted
        assert "throughput" in measures_emitted


class TestMetricEmitterPeriodicResend:
    def test_flushes_periodically_not_one_shot(self) -> None:
        buffer = MetricBuffer()
        client: MagicMock = MagicMock()
        emitter = MetricEmitter(
            buffer=buffer,
            client=client,
            stream="test:metrics",
            interval_s=0.05,
            app_name="app",
        )

        buffer.record(_make_event(measure="a", value=1.0))
        emitter.start()
        time.sleep(0.08)

        first_call_count = client.xadd.call_count
        assert first_call_count >= 1

        buffer.record(_make_event(measure="b", value=2.0))
        time.sleep(0.12)
        emitter.stop()

        assert client.xadd.call_count > first_call_count

    def test_zero_value_guarantee_emits_every_cycle(self) -> None:
        buffer = MetricBuffer()
        client: MagicMock = MagicMock()
        emitter = MetricEmitter(
            buffer=buffer,
            client=client,
            stream="test:metrics",
            interval_s=0.05,
            known_metrics={"idle_counter"},
            app_name="app",
        )

        emitter.start()
        time.sleep(0.2)
        emitter.stop()

        zero_calls = [
            c
            for c in client.xadd.call_args_list
            if c.kwargs["fields"]["measure"] == "idle_counter" and c.kwargs["fields"]["value"] == "0.0"
        ]
        assert len(zero_calls) >= 2

    def test_zero_value_skipped_when_real_event_present(self) -> None:
        buffer = MetricBuffer()
        client: MagicMock = MagicMock()
        emitter = MetricEmitter(
            buffer=buffer,
            client=client,
            stream="test:metrics",
            interval_s=0.05,
            known_metrics={"active_metric"},
            app_name="app",
        )

        buffer.record(_make_event(measure="active_metric", value=42.0))
        emitter.start()
        time.sleep(0.12)
        emitter.stop()

        all_calls = client.xadd.call_args_list
        active_calls = [c for c in all_calls if c.kwargs["fields"]["measure"] == "active_metric"]
        assert len(active_calls) >= 1

        first_flush_value = float(active_calls[0].kwargs["fields"]["value"])
        assert first_flush_value == 42.0


class TestMetricEmitterRegisterMetric:
    def test_register_metric_adds_to_known(self) -> None:
        buffer = MetricBuffer()
        client: MagicMock = MagicMock()
        emitter = MetricEmitter(
            buffer=buffer,
            client=client,
            stream="metrics",
            interval_s=0.05,
            app_name="app",
        )

        emitter.register_metric("new_metric")

        emitter.start()
        time.sleep(0.12)
        emitter.stop()

        zero_calls = [c for c in client.xadd.call_args_list if c.kwargs["fields"]["measure"] == "new_metric"]
        assert len(zero_calls) >= 1
        assert zero_calls[0].kwargs["fields"]["value"] == "0.0"


class TestFlushToRedisDirectly:
    def test_flush_produces_correct_xadd_fields(self) -> None:
        buffer = MetricBuffer()
        client: MagicMock = MagicMock()
        emitter = MetricEmitter(
            buffer=buffer,
            client=client,
            stream="s:metrics",
            interval_s=999.0,
            app_name="myapp",
        )

        buffer.record(
            MetricEvent(
                component_name="worker-1",
                measure="queue_depth",
                value=15.0,
            ),
        )

        emitter._flush_to_redis()

        assert client.xadd.call_count == 1
        kwargs = client.xadd.call_args.kwargs
        assert kwargs["name"] == "s:metrics"
        assert kwargs["fields"]["component_name"] == "myapp"
        assert kwargs["fields"]["measure"] == "queue_depth"
        assert kwargs["fields"]["value"] == "15.0"
        assert kwargs["maxlen"] == 10_000
        assert kwargs["approximate"] is True

    def test_flush_emits_zero_for_absent_known_metrics(self) -> None:
        buffer = MetricBuffer()
        client: MagicMock = MagicMock()
        emitter = MetricEmitter(
            buffer=buffer,
            client=client,
            stream="metrics",
            interval_s=999.0,
            known_metrics={"metric_a", "metric_b"},
            app_name="app",
        )

        buffer.record(_make_event(measure="metric_a", value=5.0))

        emitter._flush_to_redis()

        assert client.xadd.call_count == 2

        emitted_measures = {c.kwargs["fields"]["measure"] for c in client.xadd.call_args_list}
        assert "metric_a" in emitted_measures
        assert "metric_b" in emitted_measures

        metric_b_call = next(c for c in client.xadd.call_args_list if c.kwargs["fields"]["measure"] == "metric_b")
        assert metric_b_call.kwargs["fields"]["value"] == "0.0"
        assert metric_b_call.kwargs["fields"]["component_name"] == "app"

    def test_flush_no_duplicates_when_known_metric_present(self) -> None:
        buffer = MetricBuffer()
        client: MagicMock = MagicMock()
        emitter = MetricEmitter(
            buffer=buffer,
            client=client,
            stream="metrics",
            interval_s=999.0,
            known_metrics={"shared_metric"},
            app_name="app",
        )

        buffer.record(_make_event(measure="shared_metric", value=10.0))

        emitter._flush_to_redis()

        assert client.xadd.call_count == 1
        fields = client.xadd.call_args.kwargs["fields"]
        assert fields["measure"] == "shared_metric"
        assert fields["value"] == "10.0"

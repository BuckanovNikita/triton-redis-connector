"""Tests for MetricEmitter periodic flush to Redis."""

import time
from unittest.mock import MagicMock

from typed_flow.metrics.buffer import MetricBuffer
from typed_flow.metrics.collector import MetricEvent
from typed_flow.metrics.emitter import MetricEmitter


def _make_event(measure: str = "test", value: float = 1.0) -> MetricEvent:
    return MetricEvent(
        component_name="test-component",
        measure=measure,
        value=value,
    )


class TestMetricEmitter:
    def test_start_and_stop(self) -> None:
        buffer = MetricBuffer()
        client = MagicMock()
        emitter = MetricEmitter(buffer=buffer, client=client, stream="metrics", interval_s=0.05)
        emitter.start()
        time.sleep(0.02)
        emitter.stop()

    def test_flushes_events_to_redis(self) -> None:
        buffer = MetricBuffer()
        client = MagicMock()
        emitter = MetricEmitter(buffer=buffer, client=client, stream="test:metrics", interval_s=0.05)

        event = _make_event(measure="images_processed", value=42.0)
        buffer.record(event)

        emitter.start()
        time.sleep(0.15)
        emitter.stop()

        assert client.xadd.called
        xadd_call = client.xadd.call_args
        assert xadd_call.kwargs["name"] == "test:metrics"

    def test_xadd_called_for_each_event(self) -> None:
        buffer = MetricBuffer()
        client = MagicMock()
        emitter = MetricEmitter(buffer=buffer, client=client, stream="metrics", interval_s=0.05)

        buffer.record(_make_event(measure="m1"))
        buffer.record(_make_event(measure="m2"))
        buffer.record(_make_event(measure="m3"))

        emitter.start()
        time.sleep(0.15)
        emitter.stop()

        assert client.xadd.call_count == 3

    def test_no_xadd_when_buffer_empty(self) -> None:
        buffer = MetricBuffer()
        client = MagicMock()
        emitter = MetricEmitter(buffer=buffer, client=client, stream="metrics", interval_s=0.05)

        emitter.start()
        time.sleep(0.15)
        emitter.stop()

        client.xadd.assert_not_called()

    def test_handles_redis_connection_error(self) -> None:
        buffer = MetricBuffer()
        client = MagicMock()
        client.xadd.side_effect = ConnectionError("Connection lost")
        emitter = MetricEmitter(buffer=buffer, client=client, stream="metrics", interval_s=0.05)

        buffer.record(_make_event())

        emitter.start()
        time.sleep(0.15)
        emitter.stop()

        assert client.xadd.called

    def test_start_is_idempotent(self) -> None:
        buffer = MetricBuffer()
        client = MagicMock()
        emitter = MetricEmitter(buffer=buffer, client=client, stream="metrics", interval_s=0.05)

        emitter.start()
        emitter.start()
        emitter.stop()

    def test_xadd_uses_approximate_maxlen(self) -> None:
        buffer = MetricBuffer()
        client = MagicMock()
        emitter = MetricEmitter(buffer=buffer, client=client, stream="metrics", interval_s=0.05)

        buffer.record(_make_event())

        emitter.start()
        time.sleep(0.15)
        emitter.stop()

        xadd_call = client.xadd.call_args
        assert xadd_call.kwargs["maxlen"] == 10_000
        assert xadd_call.kwargs["approximate"] is True

    def test_fields_are_string_values(self) -> None:
        buffer = MetricBuffer()
        client = MagicMock()
        emitter = MetricEmitter(buffer=buffer, client=client, stream="metrics", interval_s=0.05, app_name="testapp")

        buffer.record(_make_event(measure="test_field", value=3.14))

        emitter.start()
        time.sleep(0.15)
        emitter.stop()

        xadd_call = client.xadd.call_args
        fields = xadd_call.kwargs["fields"]
        assert all(isinstance(v, str) for v in fields.values())
        assert fields["measure"] == "test_field"
        assert fields["component_name"] == "testapp"
        assert fields["value"] == "3.14"

    def test_component_name_from_app_name(self) -> None:
        buffer = MetricBuffer()
        client = MagicMock()
        emitter = MetricEmitter(buffer=buffer, client=client, stream="metrics", interval_s=0.05, app_name="my-app")
        assert emitter.component_name == "my_app"

        buffer.record(_make_event(measure="counter"))

        emitter.start()
        time.sleep(0.15)
        emitter.stop()

        xadd_call = client.xadd.call_args
        fields = xadd_call.kwargs["fields"]
        assert fields["measure"] == "counter"
        assert fields["component_name"] == "my_app"

    def test_component_name_falls_back_to_hostname(self) -> None:
        import socket

        buffer = MetricBuffer()
        client = MagicMock()
        emitter = MetricEmitter(buffer=buffer, client=client, stream="metrics", interval_s=0.05)
        expected_name = "".join(c if c.isalnum() else "_" for c in socket.gethostname()).strip("_")
        assert emitter.component_name == expected_name

    def test_zero_value_events_use_component_name(self) -> None:
        buffer = MetricBuffer()
        client = MagicMock()
        emitter = MetricEmitter(
            buffer=buffer,
            client=client,
            stream="metrics",
            interval_s=0.05,
            known_metrics={"idle_metric"},
            app_name="testapp",
        )

        emitter.start()
        time.sleep(0.15)
        emitter.stop()

        xadd_call = client.xadd.call_args
        fields = xadd_call.kwargs["fields"]
        assert fields["measure"] == "idle_metric"
        assert fields["component_name"] == "testapp"
        assert fields["value"] == "0.0"

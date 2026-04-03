"""Tests for LiveRedisStreamSource and TypedRedisStreamSink."""

from __future__ import annotations

from unittest.mock import MagicMock

from pydantic import BaseModel, ConfigDict
from typed_flow.config.schema import SinkConfig, SourceConfig
from typed_flow.connectors.redis_stream import (
    LiveRedisStreamSource,
    RedisStreamInputConnector,
    RedisStreamOutputConnector,
    TypedRedisStreamSink,
)
from typed_flow.errors.models import ConnectorError
from typed_flow.protocols import Envelope
from typed_flow.protocols.messaging import BaseEvent, FrameworkMeta


class SampleEvent(BaseEvent):
    model_config = ConfigDict(frozen=True, strict=True)

    camera_id: str
    image_path: str = ""


class SampleResult(BaseEvent):
    score: float
    label: str


def _make_source_config(**overrides: object) -> SourceConfig:
    defaults: dict[str, object] = {
        "stream_name": "stream:test",
        "consumer_group": "workers",
        "consumer_name": "c1",
        "event_model": "SampleEvent",
        "batch_size": 10,
        "block_ms": 100,
    }
    defaults.update(overrides)
    return SourceConfig(**defaults)  # type: ignore[arg-type]


def _make_envelope(
    msg_id: str,
    fields: dict[str, str] | None = None,
    payload: bytes = b"",
    stream: str = "stream:test",
) -> Envelope[bytes]:
    return Envelope(
        payload=payload,
        source_stream=stream,
        message_id=msg_id,
        original_fields=fields or {},
    )


class TestLiveRedisStreamSource:
    def test_returns_none_when_no_messages(self) -> None:
        connector = MagicMock(spec=RedisStreamInputConnector)
        connector.read_new.return_value = []
        config = _make_source_config()

        source = LiveRedisStreamSource(connector, config, SampleEvent)
        result = source()
        assert result is None

    def test_deserializes_event_with_flow_meta(self) -> None:
        connector = MagicMock(spec=RedisStreamInputConnector)
        envelope = _make_envelope(
            msg_id="1-0",
            fields={"camera_id": "cam01", "image_path": "/img/1.jpg"},
        )
        connector.read_new.return_value = [envelope]
        config = _make_source_config()

        source = LiveRedisStreamSource(connector, config, SampleEvent)
        result = source()

        assert isinstance(result, SampleEvent)
        assert result.camera_id == "cam01"
        assert result.image_path == "/img/1.jpg"
        assert result.flow_meta.source_stream == "stream:test"
        assert result.flow_meta.message_id == "1-0"
        assert result.flow_meta.priority == 0

    def test_returns_connector_error_on_validation_failure(self) -> None:
        connector = MagicMock(spec=RedisStreamInputConnector)
        envelope = _make_envelope(msg_id="2-0", fields={})
        connector.read_new.return_value = [envelope]
        config = _make_source_config()

        source = LiveRedisStreamSource(connector, config, SampleEvent)
        result = source()

        assert isinstance(result, ConnectorError)
        assert result.stream_name == "stream:test"
        assert result.message_id == "2-0"
        assert result.error_type == "ValidationError"

    def test_default_name_includes_stream(self) -> None:
        connector = MagicMock(spec=RedisStreamInputConnector)
        config = _make_source_config(stream_name="my:stream")
        source = LiveRedisStreamSource(connector, config, SampleEvent)
        assert source.name == "live_source_my:stream"

    def test_custom_name(self) -> None:
        connector = MagicMock(spec=RedisStreamInputConnector)
        config = _make_source_config()
        source = LiveRedisStreamSource(connector, config, SampleEvent, name="my_live")
        assert source.name == "my_live"


class TestTypedRedisStreamSink:
    def test_serializes_event_fields(self) -> None:
        mock_redis = MagicMock()
        mock_redis.xadd.return_value = b"1-0"

        sink_config = SinkConfig(stream_name="stream:output")
        output_connector = RedisStreamOutputConnector(mock_redis, sink_config)

        sink: TypedRedisStreamSink[SampleResult] = TypedRedisStreamSink(
            output_connector=output_connector,
            name="test_sink",
        )

        event = SampleResult(
            flow_meta=FrameworkMeta(source_stream="stream:input", message_id="5-0"),
            score=0.95,
            label="cat",
        )

        sink(event)

        mock_redis.xadd.assert_called_once()
        call_fields = mock_redis.xadd.call_args.kwargs.get("fields") or mock_redis.xadd.call_args[1].get("fields")
        if call_fields is None:
            call_fields = mock_redis.xadd.call_args[0][1]

        assert "score" in call_fields
        assert "label" in call_fields

    def test_flow_meta_not_in_output(self) -> None:
        mock_redis = MagicMock()
        mock_redis.xadd.return_value = b"1-0"

        sink_config = SinkConfig(stream_name="stream:output")
        output_connector = RedisStreamOutputConnector(mock_redis, sink_config)

        sink: TypedRedisStreamSink[SampleResult] = TypedRedisStreamSink(
            output_connector=output_connector,
        )

        event = SampleResult(
            flow_meta=FrameworkMeta(source_stream="s", message_id="1-0"),
            score=0.5,
            label="dog",
        )

        sink(event)

        written_data = mock_redis.xadd.call_args
        fields = written_data.kwargs.get("fields") or written_data[1].get("fields")
        if fields is None:
            fields = written_data[0][1]

        assert "flow_meta" not in fields

    def test_model_without_flow_meta_passes_through(self) -> None:
        """Models without flow_meta are serialized as-is."""
        mock_redis = MagicMock()
        mock_redis.xadd.return_value = b"1-0"

        class PlainModel(BaseModel):
            value: int

        sink_config = SinkConfig(stream_name="stream:output")
        output_connector = RedisStreamOutputConnector(mock_redis, sink_config)

        sink: TypedRedisStreamSink[PlainModel] = TypedRedisStreamSink(
            output_connector=output_connector,
        )

        event = PlainModel(value=42)
        sink(event)

        written = mock_redis.xadd.call_args
        fields = written.kwargs.get("fields") or written[1].get("fields")
        if fields is None:
            fields = written[0][1]

        assert "value" in fields

"""Tests for LiveRedisStreamSource and TypedRedisStreamSink."""

from __future__ import annotations

from unittest.mock import MagicMock

from pydantic import BaseModel, ConfigDict
from triton_redis_connector.config.schema import SourceConfig
from triton_redis_connector.connectors.redis_stream import (
    LiveRedisStreamSource,
    RedisStreamInputConnector,
    RedisStreamOutputConnector,
    TypedRedisStreamSink,
)
from triton_redis_connector.engine.events import (
    Event,
    FrameworkMeta,
    PipelineEvent,
)
from triton_redis_connector.errors.models import ConnectorError
from triton_redis_connector.protocols import Envelope


class SampleEvent(Event):
    camera_id: str
    image_path: str = ""


class SampleResult(BaseModel):
    model_config = ConfigDict(frozen=True)

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

    def test_deserializes_event_with_fw_metadata(self) -> None:
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
        assert result.fw.source_stream == "stream:test"
        assert result.fw.message_id == "1-0"
        assert result.fw.priority == 0

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
    def test_serializes_pipeline_event(self) -> None:
        mock_redis = MagicMock()
        mock_redis.xadd.return_value = b"1-0"

        from triton_redis_connector.config.schema import SinkConfig

        sink_config = SinkConfig(stream_name="stream:output")
        output_connector = RedisStreamOutputConnector(mock_redis, sink_config)

        sink: TypedRedisStreamSink[SampleEvent, SampleResult] = TypedRedisStreamSink(
            output_connector=output_connector,
            name="test_sink",
        )

        ctx = SampleEvent(
            camera_id="cam01",
            image_path="/img/1.jpg",
            fw=FrameworkMeta(source_stream="stream:input", message_id="5-0"),
        )
        payload = SampleResult(score=0.95, label="cat")
        event = PipelineEvent(context=ctx, payload=payload)

        sink(event)

        mock_redis.xadd.assert_called_once()
        call_fields = mock_redis.xadd.call_args.kwargs.get("fields") or mock_redis.xadd.call_args[1].get("fields")
        if call_fields is None:
            call_fields = mock_redis.xadd.call_args[0][1]

        assert "camera_id" in call_fields
        assert "result" in call_fields

    def test_fw_stripped_from_output(self) -> None:
        mock_redis = MagicMock()
        mock_redis.xadd.return_value = b"1-0"

        from triton_redis_connector.config.schema import SinkConfig

        sink_config = SinkConfig(stream_name="stream:output")
        output_connector = RedisStreamOutputConnector(mock_redis, sink_config)

        sink: TypedRedisStreamSink[SampleEvent, SampleResult] = TypedRedisStreamSink(
            output_connector=output_connector,
        )

        ctx = SampleEvent(
            camera_id="cam01",
            fw=FrameworkMeta(source_stream="s", message_id="1-0"),
        )
        payload = SampleResult(score=0.5, label="dog")
        event = PipelineEvent(context=ctx, payload=payload)

        sink(event)

        written_data = mock_redis.xadd.call_args
        fields = written_data.kwargs.get("fields") or written_data[1].get("fields")
        if fields is None:
            fields = written_data[0][1]

        assert "fw" not in fields

    def test_context_fields_as_top_level_entries(self) -> None:
        mock_redis = MagicMock()
        mock_redis.xadd.return_value = b"1-0"

        from triton_redis_connector.config.schema import SinkConfig

        sink_config = SinkConfig(stream_name="stream:output")
        output_connector = RedisStreamOutputConnector(mock_redis, sink_config)

        sink: TypedRedisStreamSink[SampleEvent, SampleResult] = TypedRedisStreamSink(
            output_connector=output_connector,
        )

        ctx = SampleEvent(camera_id="cam99", image_path="/data/test.jpg")
        payload = SampleResult(score=0.99, label="person")
        event = PipelineEvent(context=ctx, payload=payload)

        sink(event)

        written = mock_redis.xadd.call_args
        fields = written.kwargs.get("fields") or written[1].get("fields")
        if fields is None:
            fields = written[0][1]

        assert fields["camera_id"] == b"cam99"
        assert fields["image_path"] == b"/data/test.jpg"

    def test_result_field_is_json(self) -> None:
        mock_redis = MagicMock()
        mock_redis.xadd.return_value = b"1-0"

        import json

        from triton_redis_connector.config.schema import SinkConfig

        sink_config = SinkConfig(stream_name="stream:output")
        output_connector = RedisStreamOutputConnector(mock_redis, sink_config)

        sink: TypedRedisStreamSink[SampleEvent, SampleResult] = TypedRedisStreamSink(
            output_connector=output_connector,
        )

        ctx = SampleEvent(camera_id="cam01")
        payload = SampleResult(score=0.75, label="bicycle")
        event = PipelineEvent(context=ctx, payload=payload)

        sink(event)

        written = mock_redis.xadd.call_args
        fields = written.kwargs.get("fields") or written[1].get("fields")
        if fields is None:
            fields = written[0][1]

        result_bytes = fields.get(b"result") or fields.get("result")
        assert result_bytes is not None
        parsed = json.loads(result_bytes)
        assert parsed["score"] == 0.75
        assert parsed["label"] == "bicycle"

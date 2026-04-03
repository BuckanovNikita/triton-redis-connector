"""Tests for Redis Stream connectors and execution variant wrappers."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from typed_flow.config.schema import SinkConfig, SourceConfig
from typed_flow.connectors.consumer_group import ConsumerGroupManager
from typed_flow.connectors.redis_stream import (
    RedisStreamInputConnector,
    RedisStreamOutputConnector,
    RedisStreamSink,
)
from typed_flow.protocols import Envelope


@pytest.fixture
def mock_redis() -> MagicMock:
    return MagicMock()


@pytest.fixture
def source_config() -> SourceConfig:
    return SourceConfig(
        stream_name="stream:input",
        consumer_group="workers",
        consumer_name="c1",
        event_model="ImageEvent",
        batch_size=10,
        block_ms=1000,
    )


@pytest.fixture
def sink_config() -> SinkConfig:
    return SinkConfig(
        stream_name="stream:output",
        maxlen=10000,
        approximate_maxlen=True,
    )


@pytest.fixture
def group_manager() -> ConsumerGroupManager:
    return ConsumerGroupManager()


class TestRedisStreamInputConnector:
    def test_read_new_returns_envelopes(
        self, mock_redis: MagicMock, source_config: SourceConfig, group_manager: ConsumerGroupManager
    ) -> None:
        mock_redis.xreadgroup.return_value = [
            (
                b"stream:input",
                [
                    (b"1-0", {b"data": b"payload1"}),
                    (b"2-0", {b"data": b"payload2"}),
                ],
            )
        ]
        connector = RedisStreamInputConnector(mock_redis, source_config, group_manager)
        results = connector.read_new(count=10, block_ms=1000)

        assert len(results) == 2
        assert results[0].payload == b"payload1"
        assert results[0].source_stream == "stream:input"
        assert results[0].message_id == "1-0"
        assert results[1].payload == b"payload2"
        assert results[1].message_id == "2-0"

    def test_read_new_returns_empty_on_no_messages(
        self, mock_redis: MagicMock, source_config: SourceConfig, group_manager: ConsumerGroupManager
    ) -> None:
        mock_redis.xreadgroup.return_value = None
        connector = RedisStreamInputConnector(mock_redis, source_config, group_manager)
        results = connector.read_new(count=10, block_ms=1000)
        assert results == []

    def test_missing_data_field_returns_empty_bytes(
        self, mock_redis: MagicMock, source_config: SourceConfig, group_manager: ConsumerGroupManager
    ) -> None:
        mock_redis.xreadgroup.return_value = [
            (
                b"stream:input",
                [
                    (b"1-0", {b"other_field": b"value"}),
                ],
            )
        ]
        connector = RedisStreamInputConnector(mock_redis, source_config, group_manager)
        results = connector.read_new(count=1, block_ms=100)
        assert results[0].payload == b""


class TestRedisStreamOutputConnector:
    def test_write_calls_xadd(self, mock_redis: MagicMock, sink_config: SinkConfig) -> None:
        mock_redis.xadd.return_value = b"1-0"
        connector = RedisStreamOutputConnector(mock_redis, sink_config)
        result = connector.write({"key": b"value"})
        assert result == "1-0"
        mock_redis.xadd.assert_called_once_with(
            name="stream:output",
            fields={"key": b"value"},
            maxlen=10000,
            approximate=True,
        )

    def test_write_with_no_maxlen(self, mock_redis: MagicMock) -> None:
        config = SinkConfig(stream_name="stream:out", maxlen=None)
        mock_redis.xadd.return_value = b"2-0"
        connector = RedisStreamOutputConnector(mock_redis, config)
        result = connector.write({"data": b"test"})
        assert result == "2-0"


class TestRedisStreamSink:
    def test_writes_envelope_data(self, mock_redis: MagicMock, sink_config: SinkConfig) -> None:
        mock_redis.xadd.return_value = b"1-0"
        connector = RedisStreamOutputConnector(mock_redis, sink_config)
        sink = RedisStreamSink(connector)
        envelope: Envelope[bytes] = Envelope(
            payload=b"result_data",
            source_stream="stream:input",
            message_id="1-0",
        )
        sink(envelope)
        mock_redis.xadd.assert_called_once()

"""Tests for Redis Stream connectors and execution variant wrappers."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from triton_redis_connector.config.schema import SinkConfig, SourceConfig
from triton_redis_connector.connectors.consumer_group import ConsumerGroupManager
from triton_redis_connector.connectors.redis_stream import (
    RedisStreamInputConnector,
    RedisStreamOutputConnector,
    RedisStreamSink,
    RedisStreamSource,
)
from triton_redis_connector.protocols import Envelope


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
        recovery_interleave_ratio=4,
        pending_ttl_s=60.0,
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

    def test_read_pending_returns_envelopes_newest_first(
        self, mock_redis: MagicMock, source_config: SourceConfig, group_manager: ConsumerGroupManager
    ) -> None:
        connector = RedisStreamInputConnector(mock_redis, source_config, group_manager)
        mock_redis.xreadgroup.return_value = [
            (
                b"stream:input",
                [
                    (b"1-0", {b"data": b"old"}),
                    (b"2-0", {b"data": b"new"}),
                ],
            )
        ]
        results = connector.read_pending(count=10)
        assert len(results) == 2
        assert results[0].message_id == "2-0"
        assert results[1].message_id == "1-0"

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


class TestRedisStreamSource:
    def test_first_call_loads_pending(
        self, mock_redis: MagicMock, source_config: SourceConfig, group_manager: ConsumerGroupManager
    ) -> None:
        mock_redis.xreadgroup.return_value = None
        connector = RedisStreamInputConnector(mock_redis, source_config, group_manager)
        source = RedisStreamSource(connector, source_config)
        result = source()
        assert result is None

    def test_reads_live_messages(
        self, mock_redis: MagicMock, source_config: SourceConfig, group_manager: ConsumerGroupManager
    ) -> None:
        connector = RedisStreamInputConnector(mock_redis, source_config, group_manager)
        mock_redis.xreadgroup.side_effect = [
            None,
            [(b"stream:input", [(b"1-0", {b"data": b"live_msg"})])],
        ]
        source = RedisStreamSource(connector, source_config)
        result = source()
        assert result is not None
        assert result.payload == b"live_msg"

    def test_interleave_reads_pending_every_n_calls(
        self, mock_redis: MagicMock, group_manager: ConsumerGroupManager
    ) -> None:
        config = SourceConfig(
            stream_name="stream:input",
            consumer_group="workers",
            consumer_name="c1",
            event_model="ImageEvent",
            recovery_interleave_ratio=2,
            pending_ttl_s=9999.0,
        )
        pending_envelopes = [
            Envelope(payload=b"pending1", source_stream="stream:input", message_id="p1"),
            Envelope(payload=b"pending2", source_stream="stream:input", message_id="p2"),
        ]
        mock_connector = MagicMock(spec=RedisStreamInputConnector)
        mock_connector.read_pending.return_value = pending_envelopes
        mock_connector.read_new.return_value = [
            Envelope(payload=b"live", source_stream="stream:input", message_id="l1"),
        ]

        source = RedisStreamSource(mock_connector, config)
        results = []
        for _ in range(5):
            r = source()
            if r is not None:
                results.append(r.message_id)
        assert "p1" in results


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

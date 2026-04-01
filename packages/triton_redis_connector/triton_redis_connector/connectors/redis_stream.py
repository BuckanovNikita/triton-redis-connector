"""Redis Stream connectors and pure Source/Sink implementations.

Implements InputConnectorProtocol and OutputConnectorProtocol for Redis Streams,
plus pure Source and Sink classes for the execution variant system.
Concurrency wrapping is handled by framework wrappers at assembly time.
"""

from __future__ import annotations

import time
from collections import deque
from typing import TYPE_CHECKING, cast

from loguru import logger

from triton_redis_connector.config.schema import CollectorMode
from triton_redis_connector.protocols import Envelope

if TYPE_CHECKING:
    from redis import Redis

    from triton_redis_connector.config.schema import SinkConfig, SourceConfig
    from triton_redis_connector.connectors.consumer_group import ConsumerGroupManager


class RedisStreamInputConnector:
    """Reads messages from a Redis Stream via consumer groups.

    Implements InputConnectorProtocol. Handles XREADGROUP for both new
    and pending messages, wrapping raw Redis data into Envelopes.
    """

    def __init__(  # noqa: D107
        self,
        client: Redis[bytes],
        config: SourceConfig,
        group_manager: ConsumerGroupManager,
    ) -> None:
        self._client = client
        self._config = config
        self._group_manager = group_manager
        self._group_manager.ensure_group_exists(
            client=self._client,
            stream=self._config.stream_name,
            group=self._config.consumer_group,
        )

    def read_new(self, count: int, block_ms: int) -> list[Envelope[bytes]]:
        """Read new messages from the stream, wrapping each in an Envelope."""
        entries = self._group_manager.read_new(
            client=self._client,
            stream=self._config.stream_name,
            group=self._config.consumer_group,
            consumer=self._config.consumer_name,
            count=count,
            block_ms=block_ms,
        )
        return [self._entry_to_envelope(msg_id, fields) for msg_id, fields in entries]

    def read_pending(self, count: int) -> list[Envelope[bytes]]:
        """Read pending messages sorted newest-first, wrapping each in an Envelope."""
        entries = self._group_manager.read_pending_newest_first(
            client=self._client,
            stream=self._config.stream_name,
            group=self._config.consumer_group,
            consumer=self._config.consumer_name,
            count=count,
        )
        return [self._entry_to_envelope(msg_id, fields) for msg_id, fields in entries]

    def acknowledge(self, message_id: str) -> None:
        """Acknowledge a processed message via XACK."""
        self._group_manager.acknowledge(
            client=self._client,
            stream=self._config.stream_name,
            group=self._config.consumer_group,
            message_id=message_id,
        )

    def _entry_to_envelope(self, msg_id: str, fields: dict[bytes, bytes]) -> Envelope[bytes]:
        payload = fields.get(b"data", b"")
        original_fields = {k.decode(): v.decode() for k, v in fields.items() if k != b"data"}
        return Envelope(
            payload=payload,
            source_stream=self._config.stream_name,
            message_id=msg_id,
            original_fields=original_fields,
        )


class RedisStreamOutputConnector:
    """Writes messages to a Redis Stream via XADD.

    Implements OutputConnectorProtocol. Applies maxlen trimming
    as configured.
    """

    def __init__(self, client: Redis[bytes], config: SinkConfig) -> None:  # noqa: D107
        self._client = client
        self._config = config

    def write(self, data: dict[str, bytes]) -> str:
        """Write a message to the output stream, returning the assigned message ID."""
        result = cast(
            "bytes",
            self._client.xadd(
                name=self._config.stream_name,
                fields=data,
                maxlen=self._config.maxlen,
                approximate=self._config.approximate_maxlen,
            ),
        )
        return result.decode()


class RedisStreamSource:
    """Pure Source wrapping RedisStreamInputConnector.

    Reads from Redis with newest-first recovery and configurable
    interleave ratio for pending messages. Concurrency wrapping
    is handled by DetachedThreadSource at assembly time.
    """

    def __init__(  # noqa: D107
        self,
        connector: RedisStreamInputConnector,
        config: SourceConfig,
    ) -> None:
        self._connector = connector
        self._config = config
        self._pending_buffer: deque[Envelope[bytes]] = deque()
        self._live_count_since_pending: int = 0
        self._recovery_done = config.collector_mode != CollectorMode.RECOVERY
        self._pending_ttl_s = config.pending_ttl_s
        self._interleave_ratio = config.recovery_interleave_ratio
        if config.collector_mode != CollectorMode.RECOVERY:
            logger.info(
                "collector_mode={} for stream {}, skipping pending recovery", config.collector_mode, config.stream_name
            )

    def __call__(self) -> Envelope[bytes] | None:
        """Produce the next event, interleaving pending and live messages.

        On startup, loads pending messages sorted newest-first. During
        normal operation, uses the interleave ratio to mix pending
        recovery messages with live messages. Pending messages older
        than TTL are dropped with XACK.
        """
        if not self._recovery_done:
            self._load_pending()
            self._recovery_done = True

        if self._pending_buffer and self._should_read_pending():
            return self._next_pending()

        return self._read_live()

    def _load_pending(self) -> None:
        pending = self._connector.read_pending(count=self._config.batch_size * 10)
        now = time.monotonic()
        for envelope in pending:
            age_s = now - envelope.created_at
            if age_s > self._pending_ttl_s:
                self._connector.acknowledge(envelope.message_id)
                logger.warning(
                    "Dropped stale pending message {} (age={:.1f}s > ttl={:.1f}s)",
                    envelope.message_id,
                    age_s,
                    self._pending_ttl_s,
                )
            else:
                self._pending_buffer.append(envelope)
        if self._pending_buffer:
            logger.info(
                "Loaded {} pending messages for recovery on stream {}",
                len(self._pending_buffer),
                self._config.stream_name,
            )

    def _should_read_pending(self) -> bool:
        self._live_count_since_pending += 1
        if self._live_count_since_pending >= self._interleave_ratio:
            self._live_count_since_pending = 0
            return True
        return False

    def _next_pending(self) -> Envelope[bytes] | None:
        if not self._pending_buffer:
            return None
        return self._pending_buffer.popleft()

    def _read_live(self) -> Envelope[bytes] | None:
        results = self._connector.read_new(
            count=1,
            block_ms=self._config.block_ms,
        )
        if results:
            return results[0]
        return None


class RedisStreamSink:
    """Pure Sink wrapping RedisStreamOutputConnector.

    Writes processed results to Redis. Concurrency wrapping
    is handled by DetachedThreadSink at assembly time.
    """

    def __init__(self, connector: RedisStreamOutputConnector) -> None:  # noqa: D107
        self._connector = connector

    def __call__(self, event: Envelope[bytes]) -> None:
        """Write the envelope payload to the output Redis stream."""
        data: dict[str, bytes] = {k: v.encode() for k, v in event.original_fields.items()}
        data["data"] = event.payload
        data["source_stream"] = event.source_stream.encode()
        data["message_id"] = event.message_id.encode()
        self._connector.write(data)

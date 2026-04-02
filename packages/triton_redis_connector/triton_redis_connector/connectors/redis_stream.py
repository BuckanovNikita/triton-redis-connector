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
from pydantic import BaseModel, ValidationError

from triton_redis_connector.config.schema import CollectorMode
from triton_redis_connector.engine.events import Event, FrameworkMeta, PipelineEvent
from triton_redis_connector.errors.models import ConnectorError
from triton_redis_connector.protocols import BaseThreadSink, BaseThreadSource, Envelope

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
            mode=self._config.collector_mode,
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
    """Pure Source wrapping RedisStreamInputConnector (deprecated).

    Reads from Redis with newest-first recovery and configurable
    interleave ratio for pending messages. Concurrency wrapping
    is handled by DetachedThreadSource at assembly time.

    Deprecated: use LiveRedisStreamSource and RecoveryRedisStreamSource instead.
    """

    def __init__(  # noqa: D107
        self,
        connector: RedisStreamInputConnector,
        config: SourceConfig,
    ) -> None:
        import warnings  # noqa: PLC0415

        warnings.warn(
            "RedisStreamSource is deprecated. Use LiveRedisStreamSource or RecoveryRedisStreamSource.",
            DeprecationWarning,
            stacklevel=2,
        )
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


def _deserialize_event(
    envelope: Envelope[bytes],
    event_model: type[Event],
    connector_name: str,
    priority: int,
) -> Event | ConnectorError:
    """Deserialize an Envelope into a typed Event, setting FrameworkMeta fields.

    On ValidationError, returns a ConnectorError instead of raising.
    """
    try:
        raw_fields = dict(envelope.original_fields)
        if envelope.payload:
            raw_fields["data"] = envelope.payload.decode()
        event = event_model.model_validate(raw_fields)
    except ValidationError as exc:
        return ConnectorError(
            connector_name=connector_name,
            stream_name=envelope.source_stream,
            message_id=envelope.message_id,
            error_type=type(exc).__name__,
            error_message=str(exc),
            timestamp_ms=int(time.time() * 1000),
        )
    fw = FrameworkMeta(
        source_stream=envelope.source_stream,
        message_id=envelope.message_id,
        priority=priority,
    )
    return event.model_copy(update={"fw": fw})


class LiveRedisStreamSource[T: Event](BaseThreadSource[T | ConnectorError]):
    """Typed thread source that reads live (new) messages from a Redis Stream.

    Always reads from the latest cursor (``>``). Deserializes raw bytes into
    event model ``T`` internally. On deserialization failure returns
    ConnectorError.

    Sets ``fw.priority = 0`` (highest) on all produced events.
    """

    def __init__(  # noqa: D107
        self,
        connector: RedisStreamInputConnector,
        config: SourceConfig,
        event_model: type[T],
        *,
        name: str = "",
    ) -> None:
        super().__init__()
        self._connector = connector
        self._config = config
        self._event_model = event_model
        self.name = name or f"live_source_{config.stream_name}"

    def __call__(self) -> T | ConnectorError | None:
        """Read the next live message, deserialize, and return."""
        results = self._connector.read_new(count=1, block_ms=self._config.block_ms)
        if not results:
            return None
        envelope = results[0]
        return _deserialize_event(envelope, self._event_model, self.name, priority=0)  # type: ignore[return-value]


class RecoveryRedisStreamSource[T: Event](BaseThreadSource[T | ConnectorError]):
    """Typed thread source that recovers pending (unACKed) messages from a Redis Stream.

    On first call, bulk-reads ALL pending messages into memory, sorts them
    newest-first, then emits one by one respecting ``recovery_max_rps``
    throttle. Drops stale messages older than ``pending_ttl_s`` (ACKs them).
    Self-terminates (returns None) when the pending buffer is empty.

    Sets ``fw.priority = 1`` (lower than live) on all produced events.
    """

    def __init__(  # noqa: D107
        self,
        connector: RedisStreamInputConnector,
        config: SourceConfig,
        event_model: type[T],
        *,
        name: str = "",
    ) -> None:
        super().__init__()
        self._connector = connector
        self._config = config
        self._event_model = event_model
        self.name = name or f"recovery_source_{config.stream_name}"
        self._pending_buffer: deque[Envelope[bytes]] = deque()
        self._loaded = False
        self._sleep_interval_s = 1.0 / config.recovery_max_rps

    def __call__(self) -> T | ConnectorError | None:
        """Emit the next pending message, throttled by recovery_max_rps."""
        if not self._loaded:
            self._load_pending()
            self._loaded = True

        if not self._pending_buffer:
            return None

        envelope = self._pending_buffer.popleft()
        result = _deserialize_event(envelope, self._event_model, self.name, priority=1)
        time.sleep(self._sleep_interval_s)
        return result  # type: ignore[return-value]

    def _load_pending(self) -> None:
        all_pending = self._connector.read_pending(count=self._config.batch_size * 100)
        now = time.monotonic()
        kept = 0
        dropped = 0
        for envelope in all_pending:
            age_s = now - envelope.created_at
            if age_s > self._config.pending_ttl_s:
                self._connector.acknowledge(envelope.message_id)
                dropped += 1
            else:
                self._pending_buffer.append(envelope)
                kept += 1
        if dropped > 0:
            logger.warning(
                "Recovery dropped {} stale pending messages (ttl={:.1f}s) on stream {}",
                dropped,
                self._config.pending_ttl_s,
                self._config.stream_name,
            )
        if kept > 0:
            logger.info(
                "Recovery loaded {} pending messages for stream {}",
                kept,
                self._config.stream_name,
            )


class TypedRedisStreamSink[TCtx: Event, TResult: BaseModel](
    BaseThreadSink[PipelineEvent[TCtx, TResult]],
):
    """Typed sink that serializes PipelineEvent to Redis Stream output.

    Serialization is built into the sink:
    - Context fields written as top-level Redis hash entries (matching TCtx fields)
    - Payload written as a single ``result`` JSON field
    - ``fw`` field stripped from output (framework-internal only)
    - After writing, ACKs the source message via connector if ``fw.message_id`` is set
    """

    def __init__(  # noqa: D107
        self,
        output_connector: RedisStreamOutputConnector,
        input_connector: RedisStreamInputConnector | None = None,
        *,
        name: str = "",
    ) -> None:
        super().__init__()
        self._output = output_connector
        self._input = input_connector
        self.name = name or "typed_sink"

    def __call__(self, event: PipelineEvent[TCtx, TResult]) -> None:
        """Serialize PipelineEvent and write to Redis, then ACK source message."""
        ctx = event.context
        result = event.payload

        data: dict[str, bytes] = {}
        ctx_dict = ctx.model_dump(exclude={"fw"})
        for key, value in ctx_dict.items():
            data[key] = str(value).encode()

        data["result"] = result.model_dump_json().encode()

        self._output.write(data)

        fw = ctx.fw
        if fw.message_id and self._input is not None:
            self._input.acknowledge(fw.message_id)

"""Redis Stream connectors and pure Source/Sink implementations.

Implements InputConnectorProtocol and OutputConnectorProtocol for Redis Streams,
plus pure Source and Sink classes for the execution variant system.
Concurrency wrapping is handled by framework wrappers at assembly time.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, cast

from pydantic import BaseModel, ValidationError

from triton_redis_connector.engine.events import Event, FrameworkMeta, PipelineEvent
from triton_redis_connector.errors.models import ConnectorError
from triton_redis_connector.protocols import BaseThreadSink, BaseThreadSource, Envelope

if TYPE_CHECKING:
    from redis import Redis

    from triton_redis_connector.config.schema import SinkConfig, SourceConfig
    from triton_redis_connector.connectors.consumer_group import ConsumerGroupManager


class RedisStreamInputConnector:
    """Reads messages from a Redis Stream via consumer groups.

    Implements InputConnectorProtocol. Handles XREADGROUP for new
    messages, wrapping raw Redis data into Envelopes.
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


class TypedRedisStreamSink[TCtx: Event, TResult: BaseModel](
    BaseThreadSink[PipelineEvent[TCtx, TResult]],
):
    """Typed sink that serializes PipelineEvent to Redis Stream output.

    Serialization is built into the sink:
    - Context fields written as top-level Redis hash entries (matching TCtx fields)
    - Payload written as a single ``result`` JSON field
    - ``fw`` field stripped from output (framework-internal only)
    """

    def __init__(  # noqa: D107
        self,
        output_connector: RedisStreamOutputConnector,
        *,
        name: str = "",
    ) -> None:
        super().__init__()
        self._output = output_connector
        self.name = name or "typed_sink"

    def __call__(self, event: PipelineEvent[TCtx, TResult]) -> None:
        """Serialize PipelineEvent and write to Redis."""
        ctx = event.context
        result = event.payload

        data: dict[str, bytes] = {}
        ctx_dict = ctx.model_dump(exclude={"fw"})
        for key, value in ctx_dict.items():
            data[key] = str(value).encode()

        data["result"] = result.model_dump_json().encode()

        self._output.write(data)

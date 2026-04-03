"""Redis Stream connectors and pure Source/Sink implementations.

Implements InputConnectorProtocol and OutputConnectorProtocol for Redis Streams,
plus pure Source and Sink classes for the execution variant system.
Concurrency wrapping is handled by framework wrappers at assembly time.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, cast

from pydantic import BaseModel, ValidationError

from typed_flow.errors.models import ConnectorError
from typed_flow.protocols import BaseThreadSink, BaseThreadSource, Envelope
from typed_flow.protocols.messaging import FrameworkMeta, HasFlowMeta

if TYPE_CHECKING:
    from redis import Redis

    from typed_flow.config.schema import SinkConfig, SourceConfig
    from typed_flow.connectors.consumer_group import ConsumerGroupManager


class RedisStreamInputConnector:
    """Reads messages from a Redis Stream via consumer groups.

    Implements InputConnectorProtocol. Handles XREADGROUP for new
    messages, wrapping raw Redis data into Envelopes.
    """

    def __init__(
        self,
        client: Redis[bytes],
        config: SourceConfig,
        group_manager: ConsumerGroupManager,
    ) -> None:
        """Initialize input connector with Redis client, config, and consumer group."""
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

    def __init__(self, client: Redis[bytes], config: SinkConfig) -> None:
        """Initialize output connector with Redis client and sink config."""
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

    def __init__(self, connector: RedisStreamOutputConnector) -> None:
        """Initialize sink with an output connector."""
        self._connector = connector

    def __call__(self, event: Envelope[bytes]) -> None:
        """Write the envelope payload to the output Redis stream."""
        data: dict[str, bytes] = {k: v.encode() for k, v in event.original_fields.items()}
        data["data"] = event.payload
        data["source_stream"] = event.source_stream.encode()
        data["message_id"] = event.message_id.encode()
        self._connector.write(data)


def deserialize_envelope[T: BaseModel](
    envelope: Envelope[bytes],
    event_model: type[T],
    connector_name: str,
    priority: int,
) -> T | ConnectorError:
    """Deserialize an Envelope into a typed user model with flow_meta populated.

    The user model T must have a ``flow_meta: FrameworkMeta`` field with a default.
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
    if isinstance(event, HasFlowMeta):
        return event.model_copy(update={"flow_meta": fw})
    return event


class LiveRedisStreamSource[T: BaseModel](BaseThreadSource[T | ConnectorError]):
    """Typed thread source that reads live (new) messages from a Redis Stream.

    Always reads from the latest cursor (``>``). Deserializes raw bytes into
    user model ``T`` and populates ``flow_meta`` on the result.
    On deserialization failure returns ConnectorError.
    """

    def __init__(
        self,
        connector: RedisStreamInputConnector,
        config: SourceConfig,
        event_model: type[T],
        *,
        name: str = "",
    ) -> None:
        """Initialize live source with connector, config, and event model type."""
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
        return deserialize_envelope(results[0], self._event_model, self.name, priority=0)


class AutoRedisSource[T: BaseModel](BaseThreadSource[T | ConnectorError]):
    """Redis stream source derived from config and event type annotation.

    Users subclass with a type annotation only::

        class MySource(AutoRedisSource[MyEvent]): ...

    The event model ``T`` is extracted from the generic parameter at class
    creation time via ``__orig_bases__``. Stream configuration comes from
    the provided ``SourceConfig``.
    """

    _event_model: type[T]

    def __init_subclass__(cls, **kwargs: object) -> None:  # noqa: D105
        super().__init_subclass__(**kwargs)
        for base in getattr(cls, "__orig_bases__", ()):
            origin = getattr(base, "__origin__", None)
            if origin is AutoRedisSource:
                args = getattr(base, "__args__", ())
                if args and isinstance(args[0], type):
                    cls._event_model = args[0]
                    return
        msg = (
            f"{cls.__name__} must specify a concrete BaseModel type parameter, "
            f"e.g. class {cls.__name__}(AutoRedisSource[MyEvent]): ..."
        )
        raise TypeError(msg)

    def __init__(
        self,
        connector: RedisStreamInputConnector,
        config: SourceConfig,
        *,
        name: str = "",
    ) -> None:
        """Initialize from connector and config; event model is inferred from generic param."""
        super().__init__()
        self._connector = connector
        self._config = config
        self.name = name or f"auto_source_{config.stream_name}"

    def __call__(self) -> T | ConnectorError | None:
        """Read the next message, deserialize into ``T``, and return."""
        results = self._connector.read_new(count=1, block_ms=self._config.block_ms)
        if not results:
            return None
        return deserialize_envelope(results[0], self._event_model, self.name, priority=0)


class TypedRedisStreamSink[T: BaseModel](BaseThreadSink[T]):
    """Typed sink that serializes a BaseModel event to Redis Stream output.

    If the event has a ``flow_meta`` field (satisfies HasFlowMeta), it is
    excluded from serialization. All other fields are written to Redis.
    """

    def __init__(
        self,
        output_connector: RedisStreamOutputConnector,
        *,
        name: str = "",
    ) -> None:
        """Initialize typed sink with an output connector."""
        super().__init__()
        self._output = output_connector
        self.name = name or "typed_sink"

    def __call__(self, event: T) -> None:
        """Serialize event and write to Redis, stripping flow_meta if present."""
        exclude: set[str] = {"flow_meta"} if isinstance(event, HasFlowMeta) else set()
        data: dict[str, bytes] = {}
        for key, value in event.model_dump(exclude=exclude).items():
            data[key] = str(value).encode()
        self._output.write(data)

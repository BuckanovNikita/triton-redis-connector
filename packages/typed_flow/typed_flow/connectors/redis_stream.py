"""Redis Stream connectors and pure Source/Sink implementations.

Implements InputConnectorProtocol and OutputConnectorProtocol for Redis Streams,
plus pure Source and Sink classes for the execution variant system.
Concurrency wrapping is handled by framework wrappers at assembly time.
"""

from typing import cast

from pydantic import BaseModel, ValidationError
from redis import Redis  # noqa: TC002

from typed_flow.config.schema import SinkConfig, SourceConfig
from typed_flow.connectors.consumer_group import ConsumerGroupManager
from typed_flow.errors.models import ConnectorError
from typed_flow.protocols import Envelope
from typed_flow.protocols.messaging import FrameworkMeta, HasFlowMeta
from typed_flow.protocols.metrics import TimeMetric


class RedisStreamInputConnector:
    """Reads messages from a Redis Stream via consumer groups.

    Implements InputConnectorProtocol. Handles XREADGROUP for new
    messages, wrapping raw Redis data into Envelopes.
    """

    def __init__(
        self,
        client: "Redis[bytes]",
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
    as configured. Single writes go directly; batch writes use
    Redis pipelining for throughput.
    """

    def __init__(self, client: "Redis[bytes]", config: SinkConfig) -> None:
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

    def write_batch(self, batch: list[dict[str, bytes]]) -> int:
        """Write multiple messages in a single Redis pipeline roundtrip."""
        pipe = self._client.pipeline(transaction=False)
        for data in batch:
            pipe.xadd(
                name=self._config.stream_name,
                fields=data,
                maxlen=self._config.maxlen,
                approximate=self._config.approximate_maxlen,
            )
        pipe.execute()
        return len(batch)


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
            node_name=connector_name,
            connector_name=connector_name,
            stream_name=envelope.source_stream,
            message_id=envelope.message_id,
            error_type=type(exc).__name__,
            error_message=str(exc),
            timestamp_ms=TimeMetric.now(),
        )
    fw = FrameworkMeta(
        source_stream=envelope.source_stream,
        message_id=envelope.message_id,
        priority=priority,
    )
    if isinstance(event, HasFlowMeta):
        return event.model_copy(update={"flow_meta": fw})
    return event

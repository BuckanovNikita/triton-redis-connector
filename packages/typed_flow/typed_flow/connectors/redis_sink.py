"""Redis Stream sink variants for the typed-flow pipeline.

Provides thread and process sink implementations that serialize BaseModel events
and write them to Redis Streams via RedisStreamOutputConnector.
If the event has a flow_meta field (HasFlowMeta), it is stripped before writing.
"""

from __future__ import annotations

from pydantic import BaseModel
from redis import Redis

from typed_flow.config.schema import RedisConnectionConfig, SinkConfig  # noqa: TC001
from typed_flow.connectors.redis_stream import RedisStreamOutputConnector
from typed_flow.protocols import BaseProcessSink, BaseThreadSink
from typed_flow.protocols.messaging import HasFlowMeta


def _build_output_connector(
    redis_cfg: RedisConnectionConfig,
    sink_cfg: SinkConfig,
) -> RedisStreamOutputConnector:
    """Create a Redis client and output connector."""
    client: Redis[bytes] = Redis(
        host=redis_cfg.host,
        port=redis_cfg.port,
        db=redis_cfg.db,
        decode_responses=False,
    )
    return RedisStreamOutputConnector(client=client, config=sink_cfg)


def _serialize_event(event: BaseModel, connector: RedisStreamOutputConnector) -> None:
    """Serialize a BaseModel event and write to Redis.

    Strips flow_meta if the event satisfies HasFlowMeta.
    """
    exclude: set[str] = {"flow_meta"} if isinstance(event, HasFlowMeta) else set()
    data: dict[str, bytes] = {}
    for key, value in event.model_dump(exclude=exclude).items():
        data[key] = str(value).encode()
    connector.write(data)


class RedisThreadSink[T: BaseModel](BaseThreadSink[T]):
    """Thread sink that eagerly connects to Redis on construction."""

    def __init__(
        self,
        redis: RedisConnectionConfig,
        sink: SinkConfig,
        *,
        name: str = "",
    ) -> None:
        """Create Redis client and output connector eagerly."""
        super().__init__()
        self._connector = _build_output_connector(redis, sink)
        self.name = name or f"redis_thread_sink_{sink.stream_name}"

    def __call__(self, event: T) -> None:
        """Serialize and write event to Redis."""
        _serialize_event(event, self._connector)


class RedisProcessSink[T: BaseModel](BaseProcessSink[T]):
    """Process sink that lazily connects to Redis on first call.

    Stores only picklable config in ``__init__``. The Redis client and connector
    are created on the first ``__call__`` invocation inside the child process.
    """

    def __init__(
        self,
        redis: RedisConnectionConfig,
        sink: SinkConfig,
        *,
        name: str = "",
    ) -> None:
        """Store config for lazy initialization in child process."""
        super().__init__()
        self._redis_cfg = redis
        self._sink_cfg = sink
        self._connector: RedisStreamOutputConnector | None = None
        self.name = name or f"redis_process_sink_{sink.stream_name}"

    def __call__(self, event: T) -> None:
        """Lazily initialize connector, then serialize and write."""
        if self._connector is None:
            self._connector = _build_output_connector(self._redis_cfg, self._sink_cfg)
        _serialize_event(event, self._connector)

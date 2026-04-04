"""Redis Stream sink variants for the typed-flow pipeline.

Provides thread and process sink implementations that serialize BaseModel events
and write them to Redis Streams via RedisStreamOutputConnector.
If the event has a flow_meta field (HasFlowMeta), it is stripped before writing.
"""

import time

from loguru import logger
from pydantic import BaseModel
from redis import Redis

from typed_flow.config.schema import RedisConnectionConfig, SinkConfig
from typed_flow.connectors.redis_stream import RedisStreamOutputConnector
from typed_flow.protocols import BaseProcessSink, BaseThreadSink
from typed_flow.protocols.messaging import HasFlowMeta

_FLUSH_INTERVAL_S = 0.05
_MAX_BATCH_SIZE = 64


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


def _serialize_to_dict(event: BaseModel) -> dict[str, bytes]:
    """Serialize a BaseModel event to a dict of bytes for Redis, stripping flow_meta."""
    exclude: set[str] = {"flow_meta"} if isinstance(event, HasFlowMeta) else set()
    return {key: str(value).encode() for key, value in event.model_dump(exclude=exclude).items()}


class RedisThreadSink[T: BaseModel](BaseThreadSink[T]):
    """Thread sink that buffers events and flushes via Redis pipelining.

    Events are buffered until either ``_MAX_BATCH_SIZE`` events accumulate
    or ``_FLUSH_INTERVAL_S`` elapses since the last flush, then written
    to Redis in a single pipeline roundtrip.
    """

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
        self._buffer: list[dict[str, bytes]] = []
        self._last_flush = time.monotonic()

    def __call__(self, event: T) -> None:
        """Buffer event and flush when batch is full or interval elapsed."""
        self._buffer.append(_serialize_to_dict(event))
        now = time.monotonic()
        if len(self._buffer) >= _MAX_BATCH_SIZE or now - self._last_flush >= _FLUSH_INTERVAL_S:
            self._flush()

    def _flush(self) -> None:
        if not self._buffer:
            return
        batch = self._buffer
        self._buffer = []
        self._last_flush = time.monotonic()
        count = self._connector.write_batch(batch)
        logger.trace("Flushed {} events to Redis", count)


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
        exclude: set[str] = {"flow_meta"} if isinstance(event, HasFlowMeta) else set()
        data: dict[str, bytes] = {}
        for key, value in event.model_dump(exclude=exclude).items():
            data[key] = str(value).encode()
        self._connector.write(data)

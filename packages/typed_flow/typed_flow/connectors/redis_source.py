"""Redis Stream source variants for the typed-flow pipeline.

Provides thread and process source implementations that read from Redis Streams,
deserialize into typed user models with flow_meta, and produce them into the pipeline graph.
"""

from pydantic import BaseModel
from redis import Redis

from typed_flow.config.schema import RedisConnectionConfig, SourceConfig
from typed_flow.connectors.consumer_group import ConsumerGroupManager
from typed_flow.connectors.redis_stream import RedisStreamInputConnector, deserialize_envelope
from typed_flow.errors.models import ConnectorError
from typed_flow.protocols import BaseProcessSource, BaseThreadSource


def _build_input_connector(
    redis_cfg: RedisConnectionConfig,
    source_cfg: SourceConfig,
) -> RedisStreamInputConnector:
    """Create a Redis client, consumer group manager, and input connector."""
    client: Redis[bytes] = Redis(
        host=redis_cfg.host,
        port=redis_cfg.port,
        db=redis_cfg.db,
        decode_responses=False,
    )
    group_manager = ConsumerGroupManager()
    return RedisStreamInputConnector(
        client=client,
        config=source_cfg,
        group_manager=group_manager,
    )


class RedisThreadSource[T: BaseModel](BaseThreadSource[T | ConnectorError]):
    """Thread source that eagerly connects to Redis on construction.

    The event model ``T`` is extracted from the generic parameter at subclass
    creation time via ``__orig_bases__``.
    """

    _event_model: type[T]

    def __init_subclass__(cls, **kwargs: object) -> None:  # noqa: D105
        super().__init_subclass__(**kwargs)
        for base in getattr(cls, "__orig_bases__", ()):
            origin = getattr(base, "__origin__", None)
            if origin is RedisThreadSource:
                args = getattr(base, "__args__", ())
                if args and isinstance(args[0], type):
                    cls._event_model = args[0]
                    return
        msg = (
            f"{cls.__name__} must specify a concrete BaseModel type parameter, "
            f"e.g. class {cls.__name__}(RedisThreadSource[MyEvent]): ..."
        )
        raise TypeError(msg)

    def __init__(
        self,
        redis: RedisConnectionConfig,
        source: SourceConfig,
        *,
        name: str = "",
    ) -> None:
        """Create Redis client and input connector eagerly."""
        super().__init__()
        self._config = source
        self._connector = _build_input_connector(redis, source)
        self.name = name or f"redis_thread_source_{source.stream_name}"

    def __call__(self) -> T | ConnectorError | None:
        """Read the next message, deserialize into ``T``, and return."""
        results = self._connector.read_new(count=1, block_ms=self._config.block_ms)
        if not results:
            return None
        return deserialize_envelope(results[0], self._event_model, self.name, priority=0)


class RedisProcessSource[T: BaseModel](BaseProcessSource[T | ConnectorError]):
    """Process source that lazily connects to Redis on first call.

    Stores only picklable config in ``__init__``. The Redis client and connector
    are created on the first ``__call__`` invocation inside the child process.
    """

    _event_model: type[T]

    def __init_subclass__(cls, **kwargs: object) -> None:  # noqa: D105
        super().__init_subclass__(**kwargs)
        for base in getattr(cls, "__orig_bases__", ()):
            origin = getattr(base, "__origin__", None)
            if origin is RedisProcessSource:
                args = getattr(base, "__args__", ())
                if args and isinstance(args[0], type):
                    cls._event_model = args[0]
                    return
        msg = (
            f"{cls.__name__} must specify a concrete BaseModel type parameter, "
            f"e.g. class {cls.__name__}(RedisProcessSource[MyEvent]): ..."
        )
        raise TypeError(msg)

    def __init__(
        self,
        redis: RedisConnectionConfig,
        source: SourceConfig,
        *,
        name: str = "",
    ) -> None:
        """Store config for lazy initialization in child process."""
        super().__init__()
        self._redis_cfg = redis
        self._source_cfg = source
        self._connector: RedisStreamInputConnector | None = None
        self.name = name or f"redis_process_source_{source.stream_name}"

    def __call__(self) -> T | ConnectorError | None:
        """Lazily initialize connector, then read and deserialize."""
        if self._connector is None:
            self._connector = _build_input_connector(self._redis_cfg, self._source_cfg)
        results = self._connector.read_new(count=1, block_ms=self._source_cfg.block_ms)
        if not results:
            return None
        return deserialize_envelope(results[0], self._event_model, self.name, priority=0)

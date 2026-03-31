"""Output stream router — resolves model names to output streams."""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from triton_redis_connector.connectors.redis_stream import RedisStreamOutputConnector

if TYPE_CHECKING:
    from redis import Redis

    from triton_redis_connector.config.schema import OutputRoutingConfig, SinkConfig


class OutputStreamRouter:
    """Resolves model_name to output stream. Creates per-stream connectors lazily."""

    def __init__(
        self,
        config: OutputRoutingConfig,
        redis_client: Redis[bytes],
        sink_config: SinkConfig,
    ) -> None:  # noqa: D107
        self._config = config
        self._redis_client = redis_client
        self._sink_config = sink_config
        self._connectors: dict[str, RedisStreamOutputConnector] = {}

    def resolve_stream(self, model_name: str) -> str:
        """Return the output stream name for a given model."""
        return self._config.model_streams.get(model_name, self._config.default_stream)

    def get_connector(self, model_name: str) -> RedisStreamOutputConnector:
        """Return (or lazily create) the output connector for a model's stream."""
        stream_name = self.resolve_stream(model_name)
        if stream_name not in self._connectors:
            from triton_redis_connector.config.schema import SinkConfig  # noqa: PLC0415

            per_stream_config = SinkConfig(
                stream_name=stream_name,
                maxlen=self._sink_config.maxlen,
                approximate_maxlen=self._sink_config.approximate_maxlen,
            )
            connector = RedisStreamOutputConnector(self._redis_client, per_stream_config)
            self._connectors[stream_name] = connector
            logger.info("Created output connector for stream {}", stream_name)
        return self._connectors[stream_name]

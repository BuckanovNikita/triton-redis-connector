"""Output stream router — resolves model names to output streams."""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from triton_redis_connector.connectors.redis_stream import RedisStreamOutputConnector

if TYPE_CHECKING:
    from redis import Redis

    from triton_redis_connector.config.schema import OutputRoutingConfig, SinkConfig


class UnroutableModelError(ValueError):
    """Raised when a model has no configured output stream and no default is set."""


class OutputStreamRouter:
    """Resolves model_name to output stream. Creates per-stream connectors lazily."""

    def __init__(  # noqa: D107
        self,
        config: OutputRoutingConfig,
        redis_client: Redis[bytes],
        sink_config: SinkConfig,
    ) -> None:
        self._config = config
        self._redis_client = redis_client
        self._sink_config = sink_config
        self._connectors: dict[str, RedisStreamOutputConnector] = {}
        self._validated = False

    def validate_routing(self, known_models: list[str]) -> None:
        """Check that every known model has a routable output stream.

        Call this at startup before processing begins. Logs warnings for
        models that will fall back to the default stream. Raises
        UnroutableModelError if any model has no route and no default
        stream is configured.
        """
        unroutable: list[str] = []
        for model in known_models:
            if model in self._config.model_streams:
                continue
            if self._config.default_stream:
                logger.warning(
                    "Model {} has no explicit output stream, will use default {}",
                    model,
                    self._config.default_stream,
                )
            else:
                unroutable.append(model)

        if unroutable:
            msg = (
                f"No output stream configured for models {unroutable} "
                f"and no default_stream is set in OutputRoutingConfig"
            )
            raise UnroutableModelError(msg)

        self._validated = True
        logger.info(
            "Output routing validated: {} model-specific streams, default={}",
            len(self._config.model_streams),
            self._config.default_stream,
        )

    def resolve_stream(self, model_name: str) -> str:
        """Return the output stream name for a given model."""
        stream = self._config.model_streams.get(model_name)
        if stream is not None:
            return stream
        if self._config.default_stream:
            return self._config.default_stream
        msg = f"No output stream configured for model {model_name!r} and no default_stream is set"
        raise UnroutableModelError(msg)

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

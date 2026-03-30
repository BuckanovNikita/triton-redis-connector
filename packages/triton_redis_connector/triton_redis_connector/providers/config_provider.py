"""ConfigProvider — leaf provider with no upstream dependencies.

Accepts the fully-validated AppConfig at construction and exposes
each sub-config as a separately injectable type.
"""

from __future__ import annotations

from dishka import Provider, provide  # pyright: ignore[reportUnknownVariableType]

from triton_redis_connector.config.schema import (  # noqa: TC001 — dishka needs runtime access
    AppConfig,
    InferenceConfig,
    MetricsConfig,
    PreprocessingConfig,
    QueueConfig,
    RedisConnectionConfig,
)
from triton_redis_connector.providers.scopes import ConnectorScope


class ConfigProvider(Provider):  # noqa: D101
    scope = ConnectorScope.APP

    def __init__(self, config: AppConfig) -> None:  # noqa: D107
        super().__init__()
        self._config = config

    @provide
    def app_config(self) -> AppConfig:  # noqa: D102
        return self._config

    @provide
    def redis_config(self) -> RedisConnectionConfig:  # noqa: D102
        return self._config.redis

    @provide
    def inference_config(self) -> InferenceConfig:  # noqa: D102
        return self._config.inference

    @provide
    def preprocessing_config(self) -> PreprocessingConfig:  # noqa: D102
        return self._config.preprocessing

    @provide
    def queue_config(self) -> QueueConfig:  # noqa: D102
        return self._config.queues

    @provide
    def metrics_config(self) -> MetricsConfig:  # noqa: D102
        return self._config.metrics

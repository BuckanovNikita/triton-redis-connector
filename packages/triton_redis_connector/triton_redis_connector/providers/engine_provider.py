"""EngineProvider — depends on ConfigProvider.

Provides pipeline orchestration objects: PipelineRegistry, QueueBus, Application.
"""

from __future__ import annotations

from dishka import Provider, provide  # pyright: ignore[reportUnknownVariableType]

from triton_redis_connector.config.schema import AppConfig, QueueConfig  # noqa: TC001 — dishka needs runtime access
from triton_redis_connector.engine.app import Application
from triton_redis_connector.engine.queue_bus import QueueBus
from triton_redis_connector.engine.registry import PipelineRegistry
from triton_redis_connector.providers.scopes import ConnectorScope


class EngineProvider(Provider):
    """Provides pipeline engine objects at APP scope."""

    scope = ConnectorScope.APP

    @provide
    def pipeline_registry(self) -> PipelineRegistry:  # noqa: D102
        return PipelineRegistry()

    @provide
    def queue_bus(self, config: QueueConfig) -> QueueBus:  # noqa: D102
        return QueueBus(config=config)

    @provide
    def application(  # noqa: D102
        self,
        config: AppConfig,
        registry: PipelineRegistry,
        queue_bus: QueueBus,
    ) -> Application:
        return Application(config=config, registry=registry, queue_bus=queue_bus)

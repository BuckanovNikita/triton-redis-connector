"""MetricsProvider — depends on ConfigProvider.

Provides MetricBuffer, MetricEmitter, and error-handling objects at APP scope.
"""

from __future__ import annotations

from dishka import Provider, provide

from triton_redis_connector.config.schema import MetricsConfig  # noqa: TC001
from triton_redis_connector.errors.collector import ErrorCollector
from triton_redis_connector.errors.handler import ErrorHandler
from triton_redis_connector.metrics.buffer import MetricBuffer
from triton_redis_connector.providers.scopes import ConnectorScope


class MetricsProvider(Provider):
    """Provides metrics and error collection objects at APP scope."""

    scope = ConnectorScope.APP

    @provide
    def metric_buffer(self) -> MetricBuffer:  # noqa: D102
        return MetricBuffer()

    @provide
    def error_collector(self) -> ErrorCollector:  # noqa: D102
        return ErrorCollector()

    @provide
    def error_handler(  # noqa: D102
        self,
        collector: ErrorCollector,
        config: MetricsConfig,
    ) -> ErrorHandler:
        return ErrorHandler(
            collector=collector,
            flush_interval_s=config.error_flush_interval_s,
        )

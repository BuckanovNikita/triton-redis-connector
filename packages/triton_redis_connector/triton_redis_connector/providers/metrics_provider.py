"""MetricsProvider — depends on ConfigProvider.

Provides MetricBuffer, MetricEmitter, TritonMetricsCollector,
and error-handling objects at APP scope.
"""

from __future__ import annotations

from typing import Any

from dishka import Provider, provide
from redis import Redis  # noqa: TC002 — dishka needs runtime access

from triton_redis_connector.config.schema import AppConfig, InferenceConfig, MetricsConfig  # noqa: TC001
from triton_redis_connector.errors.collector import ErrorCollector
from triton_redis_connector.errors.handler import ErrorHandler
from triton_redis_connector.metrics.buffer import MetricBuffer
from triton_redis_connector.metrics.emitter import MetricEmitter
from triton_redis_connector.metrics.triton_collector import TritonMetricsCollector
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

    @provide
    def triton_metrics_collector(  # noqa: D102
        self,
        inference_config: InferenceConfig,
        metrics_config: MetricsConfig,
        buffer: MetricBuffer,
    ) -> TritonMetricsCollector:
        import tritonclient.grpc as grpcclient  # noqa: PLC0415

        client: Any = grpcclient.InferenceServerClient(url=inference_config.grpc_url)
        return TritonMetricsCollector(
            client=client,
            buffer=buffer,
            poll_interval_s=metrics_config.triton_poll_interval_s,
        )

    @provide
    def metric_emitter(  # noqa: D102
        self,
        buffer: MetricBuffer,
        client: Redis,  # type: ignore[type-arg]
        config: MetricsConfig,
        app_config: AppConfig,
    ) -> MetricEmitter:
        return MetricEmitter(
            buffer=buffer,
            client=client,
            stream=config.output_stream,
            interval_s=config.emit_interval_s,
            app_name=app_config.app_name,
        )

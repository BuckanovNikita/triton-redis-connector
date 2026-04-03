"""Generic event-streaming framework with type-based DAG routing.

Public API re-exports for the typed-flow package.
"""

from __future__ import annotations

from typed_flow.config.schema import AppConfig
from typed_flow.connectors.redis_stream import AutoRedisSource
from typed_flow.engine.app import Application
from typed_flow.engine.metric_aggregator import MetricAggregator, Tracked
from typed_flow.engine.registry import (
    ExecutionVariant,
    GraphNode,
    PipelineRegistry,
)
from typed_flow.errors.models import ConnectorError
from typed_flow.protocols import (
    BaseEvent,
    BaseProcessPoolProcessor,
    BaseProcessSink,
    BaseProcessSource,
    BaseThreadPoolProcessor,
    BaseThreadProcessor,
    BaseThreadSink,
    BaseThreadSource,
    Counter,
    FrameworkMeta,
    Gauge,
    GroupBy,
    HasFlowMeta,
    LifecycleService,
    MetricRecorder,
    ProcessorFactory,
    ProcessPoolProcessor,
    ProcessSink,
    ProcessSource,
    ThreadPoolProcessor,
    ThreadProcessor,
    ThreadSink,
    ThreadSource,
    WrappedNode,
)

__all__ = [
    "AppConfig",
    "Application",
    "AutoRedisSource",
    "BaseEvent",
    "BaseProcessPoolProcessor",
    "BaseProcessSink",
    "BaseProcessSource",
    "BaseThreadPoolProcessor",
    "BaseThreadProcessor",
    "BaseThreadSink",
    "BaseThreadSource",
    "ConnectorError",
    "Counter",
    "ExecutionVariant",
    "FrameworkMeta",
    "Gauge",
    "GraphNode",
    "GroupBy",
    "HasFlowMeta",
    "LifecycleService",
    "MetricAggregator",
    "MetricRecorder",
    "PipelineRegistry",
    "ProcessPoolProcessor",
    "ProcessSink",
    "ProcessSource",
    "ProcessorFactory",
    "ThreadPoolProcessor",
    "ThreadProcessor",
    "ThreadSink",
    "ThreadSource",
    "Tracked",
    "WrappedNode",
]

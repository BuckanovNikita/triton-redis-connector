"""Protocol definitions, Envelope wrapper, and all 6 parallel execution variants.

This is the leaf layer in the import-linter hierarchy. Every other module
in this package may import from protocols, but protocols has zero
internal dependencies (except between its own sub-modules).

All protocols are runtime_checkable for isinstance verification during
graph construction.
"""

from typed_flow.protocols.execution import (
    BaseProcessPoolProcessor,
    BaseProcessSink,
    BaseProcessSource,
    BaseThreadPoolProcessor,
    BaseThreadProcessor,
    BaseThreadSink,
    BaseThreadSource,
    ProcessPoolProcessor,
    ProcessSink,
    ProcessSource,
    ThreadPoolProcessor,
    ThreadProcessor,
    ThreadSink,
    ThreadSource,
)
from typed_flow.protocols.factories import ProcessorFactory
from typed_flow.protocols.framework import (
    AggregationRuleProtocol,
    BucketStats,
    WrappedNode,
)
from typed_flow.protocols.generators import StreamGenerator
from typed_flow.protocols.lifecycle import (
    LifecycleService,
    ObserverProtocol,
    Startable,
    Stoppable,
)
from typed_flow.protocols.messaging import BaseEvent, Envelope, FrameworkMeta, HasFlowMeta
from typed_flow.protocols.metrics import (
    Counter,
    Gauge,
    GroupBy,
    MetricRecorder,
    TimeMetric,
)

__all__ = [
    "AggregationRuleProtocol",
    "BaseEvent",
    "BaseProcessPoolProcessor",
    "BaseProcessSink",
    "BaseProcessSource",
    "BaseThreadPoolProcessor",
    "BaseThreadProcessor",
    "BaseThreadSink",
    "BaseThreadSource",
    "BucketStats",
    "Counter",
    "Envelope",
    "FrameworkMeta",
    "Gauge",
    "GroupBy",
    "HasFlowMeta",
    "LifecycleService",
    "MetricRecorder",
    "ObserverProtocol",
    "ProcessPoolProcessor",
    "ProcessSink",
    "ProcessSource",
    "ProcessorFactory",
    "Startable",
    "Stoppable",
    "StreamGenerator",
    "ThreadPoolProcessor",
    "ThreadProcessor",
    "ThreadSink",
    "ThreadSource",
    "TimeMetric",
    "WrappedNode",
]

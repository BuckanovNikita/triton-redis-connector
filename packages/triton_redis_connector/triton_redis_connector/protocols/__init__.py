"""Protocol definitions, Envelope wrapper, and all 6 parallel execution variants.

This is the leaf layer in the import-linter hierarchy. Every other module
in this package may import from protocols, but protocols has zero
internal dependencies (except between its own sub-modules).

All protocols are runtime_checkable for isinstance verification during
graph construction.

Backward compatibility: all names previously in ``protocols.py`` are
re-exported here so that ``from triton_redis_connector.protocols import X``
continues to work unchanged.
"""

from __future__ import annotations

from triton_redis_connector.protocols.execution import (
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
from triton_redis_connector.protocols.factories import ProcessorFactory
from triton_redis_connector.protocols.framework import (
    AggregationRuleProtocol,
    WrappedNode,
)
from triton_redis_connector.protocols.lifecycle import (
    LifecycleService,
    ObserverProtocol,
    Startable,
    Stoppable,
)
from triton_redis_connector.protocols.messaging import Envelope
from triton_redis_connector.protocols.metrics import (
    Counter,
    Gauge,
    GroupBy,
    MetricRecorder,
)

__all__ = [
    "AggregationRuleProtocol",
    "BaseProcessPoolProcessor",
    "BaseProcessSink",
    "BaseProcessSource",
    "BaseThreadPoolProcessor",
    "BaseThreadProcessor",
    "BaseThreadSink",
    "BaseThreadSource",
    "Counter",
    "Envelope",
    "Gauge",
    "GroupBy",
    "LifecycleService",
    "MetricRecorder",
    "ObserverProtocol",
    "ProcessPoolProcessor",
    "ProcessSink",
    "ProcessSource",
    "ProcessorFactory",
    "Startable",
    "Stoppable",
    "ThreadPoolProcessor",
    "ThreadProcessor",
    "ThreadSink",
    "ThreadSource",
    "WrappedNode",
]

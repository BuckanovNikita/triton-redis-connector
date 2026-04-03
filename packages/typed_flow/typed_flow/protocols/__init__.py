"""Protocol definitions, Envelope wrapper, and all 6 parallel execution variants.

This is the leaf layer in the import-linter hierarchy. Every other module
in this package may import from protocols, but protocols has zero
internal dependencies (except between its own sub-modules).

All protocols are runtime_checkable for isinstance verification during
graph construction.
"""

from __future__ import annotations

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
    WrappedNode,
)
from typed_flow.protocols.lifecycle import (
    LifecycleService,
    ObserverProtocol,
    Startable,
    Stoppable,
)
from typed_flow.protocols.messaging import Envelope
from typed_flow.protocols.metrics import (
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

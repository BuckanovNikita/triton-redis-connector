# Step 07: Split protocols.py into Sub-Package

**Agent:** implementer
**Depends on:** step 05

## Goal

Split the monolithic `protocols.py` (413 lines, 30 names) into a `protocols/` sub-package with logical grouping. Maintain full backward compatibility via re-exports.

## New structure

```
triton_redis_connector/protocols/
├── __init__.py          # Re-exports ALL 30 names (zero import churn)
├── metrics.py           # Counter, Gauge, GroupBy, MetricRecorder
├── messaging.py         # Envelope[T]
├── execution.py         # 6 protocol classes + 6 base classes + Tracked
├── factories.py         # ProcessorFactory
├── lifecycle.py         # LifecycleService, Startable, Stoppable, ObserverProtocol
└── framework.py         # WrappedNode, AggregationRuleProtocol
```

## Tasks

### 7.1 Create `protocols/metrics.py`

Move from protocols.py:
- `Counter` (dataclass, lines 19-30)
- `Gauge` (dataclass, lines 32-43)
- `GroupBy` (dataclass, lines 45-57)
- `MetricRecorder` (Protocol, lines 223-230)

No internal dependencies.

### 7.2 Create `protocols/messaging.py`

Move from protocols.py:
- `Envelope[T]` (Generic class, lines 69-117)

Dependencies: `threading`, `time`, `dataclasses`, `typing`

### 7.3 Create `protocols/execution.py`

Move from protocols.py:
- 6 protocol classes: `ThreadSource`, `ProcessSource`, `ThreadProcessor`, `ThreadPoolProcessor`, `ProcessPoolProcessor`, `ThreadSink`
- 6 base classes: `BaseThreadSource`, `BaseProcessSource`, `BaseThreadProcessor`, `BaseThreadPoolProcessor`, `BaseProcessPoolProcessor`, `BaseThreadSink`
- `Tracked` (if it exists as annotation type)

Dependencies: `threading`, `typing`

### 7.4 Create `protocols/factories.py`

Move from protocols.py:
- `ProcessorFactory[TShared, TProcessor_co]` (Protocol, lines 205-221)

### 7.5 Create `protocols/lifecycle.py`

Move from protocols.py:
- `LifecycleService` (Protocol, lines 232-243)
- `Startable` (Protocol, lines 257-264)
- `Stoppable` (Protocol, lines 266-273)
- `ObserverProtocol[TObserved]` (Protocol, lines 245-255)

### 7.6 Create `protocols/framework.py`

Move from protocols.py:
- `WrappedNode` (Protocol, lines 275-287)
- `AggregationRuleProtocol[TEvent_contra]` (Protocol, lines 289-314)

### 7.7 Create `protocols/__init__.py`

Re-export ALL names for backward compatibility:
```python
from triton_redis_connector.protocols.execution import (
    BaseProcessPoolProcessor, BaseProcessSource, BaseThreadPoolProcessor,
    BaseThreadProcessor, BaseThreadSink, BaseThreadSource,
    ProcessPoolProcessor, ProcessSource, ThreadPoolProcessor,
    ThreadProcessor, ThreadSink, ThreadSource,
)
from triton_redis_connector.protocols.factories import ProcessorFactory
from triton_redis_connector.protocols.framework import AggregationRuleProtocol, WrappedNode
from triton_redis_connector.protocols.lifecycle import (
    LifecycleService, ObserverProtocol, Startable, Stoppable,
)
from triton_redis_connector.protocols.messaging import Envelope
from triton_redis_connector.protocols.metrics import Counter, Gauge, GroupBy, MetricRecorder

__all__ = [
    # ... all 30 names
]
```

### 7.8 Delete old `protocols.py`

After all names are moved and re-exported, delete the original file.

### 7.9 Update import-linter contracts

**File:** root `pyproject.toml`

The leaf layer currently references `(triton_redis_connector.protocols)`. Update to include sub-modules. Options:
- Use `(triton_redis_connector.protocols) | (triton_redis_connector.protocols.metrics) | ...`
- Or check if import-linter supports wildcard like `(triton_redis_connector.protocols)**`

### 7.10 Update mypy/pyright config

If `protocols.py` was excluded or had special treatment, update paths to `protocols/`.

## Key constraint

**NO existing import statements should need changing.** Everything still imports from `triton_redis_connector.protocols` — the `__init__.py` re-exports ensure this.

## Consumer verification

Top consumers (verify they still work):
- `engine/registry.py` — 13 imports
- `engine/app.py` — 8 imports
- `__init__.py` — 19 imports
- `engine/shipped_processors.py` — 2 imports
- `connectors/redis_stream.py` — 3 imports
- `engine/metric_aggregator.py` — 3 imports

## Verification

```bash
uv run pre-commit run --all-files
uv run lint-imports
uv run pytest packages/triton_redis_connector/
```

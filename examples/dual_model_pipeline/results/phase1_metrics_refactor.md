# Phase 1: Metrics Architecture Refactor Results

## Before

```
app.py (595 lines):
  - _make_metric()              : creates MetricEvent (lazy import)
  - _extract_gauge_value()      : extracts float from event fields
  - _is_metric_event()          : checks if result is MetricEvent
  - _apply_metric_rules()       : Counter + Gauge extraction (GroupBy missing!)
  - _emit_node_processing_metric() : per-call timing metric
  - _emit_transit_metric()      : end-to-end latency
  - _emit_per_node_stats()      : node-level metrics for ALL nodes
  - _emit_queue_stats()         : queue depth and drops
  - _log_stats()                : text logging + metric emission

wrappers.py (217 lines):
  - 6 classes with identical __call__ bodies (timing + aggregator)
  - No metric extraction
  - No MetricBuffer access

emitter.py:
  - Flushes buffer to Redis
  - No zero-value guarantee
```

## After

```
app.py (~374 lines):
  - _route_event()   : pure event routing, NO metric logic
  - _log_stats()     : text-only logging (no MetricEvent creation)
  - All _emit_*() methods: REMOVED
  - _stats_counters: REMOVED

wrappers.py (~253 lines):
  - _call_with_metrics()           : shared call logic (DRY)
  - _extract_and_record_metrics()  : Counter + Gauge + GroupBy extraction
  - _make_metric()                 : lazy MetricEvent creation
  - 6 classes delegate to _call_with_metrics()
  - New params: metric_recorder, component_name

emitter.py (~112 lines):
  - register_metric(name)  : NEW - register known metrics
  - _known_metrics set     : NEW - tracks expected metrics
  - _flush_to_redis()      : emits zero-value for missing metrics
  - _xadd_event()          : extracted helper (DRY)
```

## Key Decisions

1. **MetricRecorder protocol** used in wrappers (from protocols.py leaf layer) instead of importing MetricBuffer directly - preserves import-linter architecture

2. **Lazy import** of MetricEvent in wrappers via `_make_metric()` - same pattern app.py used, avoids engine->metrics layer violation

3. **GroupBy format**: `MetricEvent(measure=f"{groupby.name}:{field_value}", value=1.0)` - the field value becomes part of the metric name for per-key counting

4. **Zero-value guarantee**: MetricEmitter emits `MetricEvent(value=0.0)` for any registered metric not seen in the flush cycle

## Lines Changed
- app.py: ~210 lines removed
- wrappers.py: ~36 lines added (net, after removing duplication)
- metric_aggregator.py: ~15 lines added (extract_field_value)
- emitter.py: ~30 lines added (zero-value + register_metric)
- pyproject.toml: 1 line changed (import-linter ignore target)
- test_app.py: ~80 lines added/modified (new wrapper metric tests)

## Verification
- 259 tests pass
- 5 import-linter contracts kept
- vulture clean
- ruff format + check clean

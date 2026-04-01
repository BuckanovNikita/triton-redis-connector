# Phase 0: Code Audit Results

## DRY / SOLID / Architecture Violations Found

### High Severity

| # | Issue | Location | Status |
|---|-------|----------|--------|
| 1 | Wrapper `__call__` duplication - identical timing/aggregator logic in all 6 wrappers | `engine/wrappers.py:37-216` | **Fixed in Phase 1** |
| 2 | Metrics logic embedded in app.py - `_apply_metric_rules()`, `_emit_*()`, `_log_stats()` | `engine/app.py:242-455` | **Fixed in Phase 1** |
| 3 | GroupBy declared but never wired - `extract_field_group_bys()` exists, never called | `engine/metric_aggregator.py:158` | **Fixed in Phase 1** |
| 4 | MetricEmitter has no zero-value guarantee | `metrics/emitter.py:67-86` | **Fixed in Phase 1** |
| 5 | InferenceWorker rejects image_bytes - but InferenceInput supports both modes | `inference/worker.py:34` | **Fixed in Phase 2** |

### Medium Severity

| # | Issue | Location | Status |
|---|-------|----------|--------|
| 6 | MetricBuffer / ErrorCollector identical lock+flush pattern | `metrics/buffer.py`, `errors/collector.py` | Deferred |
| 7 | Redundant `_stats_*` dicts in Application | `app.py:112-120` | Partially fixed (removed `_stats_counters`, `_stats_prev_per_node`) |
| 8 | Router class loaded by string, no interface validation | `config/schema.py:122` | Deferred |
| 9 | Envelope.states list grows unbounded | `protocols.py:85` | Deferred |
| 10 | QueueOverflow return ignored by `_run_parallel_node` | `app.py:518` | Deferred |

## Architecture Observations

### Strengths
- Clean type-based routing via PipelineRegistry - union return types create branching naturally
- Protocol-based execution variant detection (WrappedNode fast path)
- Topological sort for inline dispatch order
- Event-level metric declaration (Counter/Gauge/GroupBy as annotations)

### Weaknesses Addressed
- **Metric coupling**: app.py was 595 lines with ~210 lines of metric logic. Now ~374 lines, pure event routing
- **Wrapper duplication**: 6 identical `__call__` bodies reduced to 1 shared `_call_with_metrics()` function
- **GroupBy gap**: Declared in protocols.py, extraction function existed, but never invoked anywhere
- **Zero-value gap**: MetricEmitter only wrote events that existed in buffer; missing metrics = silence (not zero)

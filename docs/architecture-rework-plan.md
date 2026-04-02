# Architecture Rework Plan — v4

## Context

The current pipeline engine hardcodes a single-model detection flow. Preprocessing
hides model routing inside constructors, inline variants create a main-thread
bottleneck, `build_application()` magically wires processors, and DetectionResult
uses raw JSON strings instead of typed pydantic structures. This rework makes the
engine multi-model, classification-aware, explicit, and type-safe.

**No backward compatibility.** Obsolete code is deleted, not deprecated.

## Design Decisions (locked)

| Decision | Choice |
|---|---|
| Multi-model routing | Typed subclasses per ModelBundle — framework generates `{Name}TritonRequest` / `{Name}InferenceOutput` / `{Name}ImagePayload` |
| Detection/Classification types | Live in `triton_shared_models` (leaf package) |
| Metadata/context | **Composition**: `PipelineEvent[TCtx, TPayload]` — typed user context + typed payload. No `dict[str, Any]` metadata. |
| Inline variants | Dropped — only thread/process variants remain |
| Type extraction | `__orig_bases__` from class hierarchy, no `__call_hints__`, no subtype checking |
| Validation | Structural: every bundle must have inference path + postprocessor + typed sink + triton metrics |
| Source deserialization | `RedisStreamSource[T]` deserializes bytes→T internally. No separate EnvelopeExtractor. |
| Generic matching | **Exact match only** — `PipelineEvent[A, B]` matches only `PipelineEvent[A, B]`, no subtype on generic args |
| Source error handling | Deserialization failure produces `ConnectorError` event (union return type on source) |
| Redis output format | Context fields as top-level hash entries (matching TCtx class signature) + one `result` JSON field for payload |
| Dynamic type generation | `type()` at runtime with subscripted bases for `__orig_bases__` + `TYPE_CHECKING` overloads |
| Backward compatibility | **None.** Drop all obsolete code — no aliases, no deprecation wrappers. |
| Config defaults | **No defaults.** All config fields required. Full explicit config for app start. |
| Hydra usage | Only `hydra.main` allowed. No compose API, no other patterns. |
| Sink routing | Serialization joined into sinks. Each sink is typed: `RedisStreamSink[TCtx, TResult]`. No separate serializer node, no Envelope in pipeline. |
| Parallel dispatch | Runtime type check before queue routing — filter consumers by actual produced type |
| Framework metadata | `Event` base class owns `fw: FrameworkMeta` (single struct). Pydantic prevents user from redefining `fw` field. Not serialized to Redis output — sink strips it. |
| Metrics prefix | `MetricEmitter` prepends one of `{APP_NAME}_` or `{HOSTNAME}_` (app_name preferred, hostname fallback). Only `_` separator. |

---

## Resolved Blockers from Review

### BLOCKER A: Parallel routing dispatches all types to all consumers
**Problem**: `_route_to_consumers()` uses precomputed routing table by producer name. Union returns (`PipelineEvent | InferenceError`) go to ALL consumers, including wrong ones.
**Fix**: `_route_to_consumers()` does **runtime type check** before putting event on queue. Routing table stores `(consumer_name, expected_input_type)` pairs. For each consumer, check `type(result)` matches expected type using exact generic comparison. Only matching consumers receive the event.

### BLOCKER B: Shipped Node classes not generic over payload type
**Problem**: `ClientPreprocessorNode[TCtx]` consumes `PipelineEvent[TCtx, ImagePayload]` but exact matching requires `PipelineEvent[TCtx, DetectionImagePayload]`.
**Fix**: All shipped Node classes are generic over BOTH context and payload:
```python
class ClientPreprocessorNode[TCtx: Event, TPayload: ImagePayload](
    BaseThreadProcessor[PipelineEvent[TCtx, TPayload], PipelineEvent[TCtx, TritonRequest]]
)
class InferenceNode[TCtx: Event, TReq: TritonRequest, TOut: InferenceOutput](
    BaseThreadProcessor[PipelineEvent[TCtx, TReq], PipelineEvent[TCtx, TOut] | InferenceError]
)
class PostProcessorNode[TCtx: Event, TIn: InferenceOutput, TResult: BaseModel](
    BaseThreadProcessor[PipelineEvent[TCtx, TIn], PipelineEvent[TCtx, TResult]]
)
```

### BLOCKER G: Multiple sinks all consume Envelope[bytes]
**Problem**: All serializers produce `Envelope[bytes]`, all sinks consume `Envelope[bytes]`. No type-based routing.
**Fix**: Serialization joined into sinks. No separate serializer. Each sink is typed: `RedisStreamSink[SteelMeta, DetectionResult]` consumes `PipelineEvent[SteelMeta, DetectionResult]`. Exact matching on `PipelineEvent` generic args routes correctly. No Envelope in the pipeline at all.

### MAJOR E: Envelope metadata (source_stream, message_id) lost
**Problem**: `RedisStreamSource[T]` returns bare `T`, discarding envelope metadata. Sink needs source_stream/message_id.
**Fix**: One `FrameworkMeta` struct on `Event` base class:
```python
class FrameworkMeta(BaseModel):
    model_config = ConfigDict(frozen=True)
    source_stream: str = ""
    message_id: str = ""

class Event(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True)
    fw: FrameworkMeta = FrameworkMeta()
```
- `RedisStreamSource[T: Event]` deserializes user event and sets `event.fw` from envelope metadata
- If user defines a field named `fw` in their subclass → **pydantic raises at class definition time** (conflict visible at import)
- `PipelineEvent.context` is `TCtx: Event`, so `fw` travels through the pipeline inside context
- `RedisStreamSink` reads `event.context.fw.source_stream` / `.message_id` for Redis routing
- **`fw` is NOT serialized to Redis output** — sink strips it when writing context fields to hash. Only user-defined fields appear in output.
- Single field to protect (`fw`), not a prefix convention

### MAJOR I: Gap between removing inline and creating Node classes
**Fix**: No gap. Since we drop backward compat, Wave 2 does both: removes inline variants AND creates Node classes in one sequential chain. No need to keep old shipped processors working without Node wrappers.

---

## Wave 1 — All Independent (run in parallel)

### Task 1.1: Detection type in shared_models
**New file**: `packages/triton_shared_models/triton_shared_models/detection/models.py`
- `Detection(BaseModel, frozen=True)`: x1, y1, x2, y2, confidence: float, class_id: int, class_name: str = "", all_class_scores: tuple[float, ...] | None = None
- `packages/triton_shared_models/triton_shared_models/detection/__init__.py` — export Detection
- Update `triton_model_tools/models/detection.py` to re-import from shared_models

### Task 1.2: Classification types in shared_models
**New file**: `packages/triton_shared_models/triton_shared_models/classification/models.py`
- `ClassificationPrediction(BaseModel, frozen=True)`: class_id: int, class_name: str = "", probability: float
- `packages/triton_shared_models/triton_shared_models/classification/__init__.py` — export ClassificationPrediction

### Task 1.3: Classification preprocessing
**New file**: `packages/triton_shared_models/triton_shared_models/classification/preprocessing.py`
- `CenterCropPreprocessor`: resize short edge 256, center crop 224x224, ImageNet normalize
- Existing `LetterBoxPreprocessor` moves to `triton_shared_models/detection/preprocessing.py`

### Task 1.4: Classification postprocessor in model_tools
**Depends on**: 1.2 (imports ClassificationPrediction — soft dep, 1.2 is trivial)
**New files**:
- `triton_model_tools/inference/classification_postprocessing.py` — ClassificationPostprocessor
- `triton_model_tools/models/classification.py` — ImageClassificationResult, FolderClassificationResult

### Task 1.5: Classification comparison tools
**Modify**: `triton_model_tools/benchmarks/comparison.py`
- Add ClassificationComparisonMetrics, ClassificationComparisonSummary, compare_classification()

### Task 1.6: TritonRequest rename (no alias, drop InferenceInput entirely)
**Modify**: `engine/events.py` — rename InferenceInput → TritonRequest. Delete InferenceInput. Remove dead InferenceOutput + TensorSpec.
**Modify**: all imports across codebase — direct replacement, no alias.

---

## Wave 2 — Engine Core (sequential: 2.1→2.2→2.3→2.4, parallel with Wave 1)

### Task 2.1: Drop inline variants + remove obsolete code
**Modify**: `protocols.py` — delete Source, Processor, Sink protocols; delete BaseSource, BaseProcessor, BaseSink
**Modify**: `engine/registry.py` — delete SOURCE, PROCESSOR, SINK from ExecutionVariant; delete INLINE_VARIANTS; delete get_dispatch_order(); delete _infer_variant_from_signature()
**Modify**: `config/schema.py` — delete SOURCE, PROCESSOR, SINK from NodeVariant; **remove all default values** from config fields
**Modify**: `engine/assembly.py` — delete _INLINE_VARIANTS; wrap_with_variant() always wraps
**Modify**: `engine/diagram.py` — remove inline node shapes
**Delete**: EnvelopeExtractor class, old build_application(), _extract_return_type()

### Task 2.2: PipelineEvent + ImagePayload + typed source/sink
**Modify**: `engine/events.py`:
- Add `PipelineEvent[TCtx: Event, TPayload: BaseModel]` (fw_ fields are on Event base, not PipelineEvent)
- Add `ImagePayload(BaseModel)`: image_path, image_bytes
- Remove `metadata: dict` from TritonRequest, InferenceOutput, DetectionResult
- Remove DecodedImage (obsolete)

**Modify**: `connectors/redis_stream.py`:
- `RedisStreamSource[T: BaseModel]` inherits `BaseThreadSource[T | ConnectorError]`
  - Deserializes internally (JSON first, then original_fields fallback)
  - On ValidationError → returns ConnectorError
  - Stores source_stream/message_id in thread-local for business processor access
### Two sources: live + recovery (split, not interleaved)

Instead of one source doing complex interleaving, **split into two independent sources**:

**`LiveRedisStreamSource[T: Event]`** — `BaseThreadSource[T | ConnectorError]`:
- Always reads from latest (`>` cursor) — never blocked by old messages
- New images get **immediate** processing, zero latency from recovery
- Always enabled — this is the primary source

**`RecoveryRedisStreamSource[T: Event]`** — `BaseThreadSource[T | ConnectorError]`:
- On start: **bulk-reads ALL pending messages at once** into memory (fast Redis scan, no processing yet)
- Sorts by message ID descending — **newest first**
- Then emits one by one from newest to oldest, respecting `recovery_max_rps` throttle
- **Fully controllable** via config:
  - `recovery_enabled: bool` — disable entirely
  - `recovery_max_rps: int` — max events per second from recovery source (throttle to not overload Triton). Source sleeps `1000/rps` ms between emits.
  - All required, no defaults
- Drops stale messages older than `pending_ttl_s` (ACK + skip)
- Self-terminates when pending buffer is empty (switches to idle/sleep, stops producing)
- Checks queue depth before pushing — pauses at 80% capacity to prevent overflow
- Runs on its own thread, separate queue — never competes with live source

**Both produce the same `T` type**. Priority enforced at **Triton level only** — our queues stay simple FIFO.

**Triton request priority**: `InferenceNode` sets `priority` on the gRPC inference request based on `fw.priority`:
```python
result = client.infer(
    model_name=req.model_name,
    inputs=[...],
    priority=event.context.fw.priority,   # 0 = highest (live), 1 = lower (recovery)
    timeout=timeout_ms,
)
```
- `LiveRedisStreamSource` sets `fw.priority = 0` (high)
- `RecoveryRedisStreamSource` sets `fw.priority = 1` (low)
- `FrameworkMeta` gains `priority: int = 0` field
- Triton scheduler processes high-priority requests first when queue is contended
- Our pipeline queues stay plain `queue.Queue` FIFO — no PriorityQueue complexity
- Pre/postprocessing run FIFO — but these are fast (<5ms); the bottleneck is inference where Triton handles priority
- Triton must be configured with priority scheduling: `dynamic_batching { priority_levels: 2, default_priority_level: 1 }` in config.pbtxt

**Message acknowledgment** — ACK after sink writes:
- Source stores `fw.message_id` on the event
- `fw.message_id` travels through pipeline in `PipelineEvent.context.fw`
- Sink writes result to Redis, then ACKs via shared ACK queue: `ack_queue.put(fw.message_id)`
- Source thread (or dedicated ACK thread) drains ACK queue and calls `connector.acknowledge(msg_id)`
- If processing fails → message NOT ACK'd → stays pending for RecoverySource on next restart
- **At-least-once delivery**: never silently lost

**Config**:
```yaml
sources:
  - stream_name: steel:input
    consumer_group: steel-pipeline
    recovery_enabled: true       # false to skip pending entirely
    recovery_max_rps: 10         # max 10 recovery events/sec — won't overload Triton
    pending_ttl_s: 60.0          # drop stale messages older than 60s
```
User controls how aggressively recovery runs — from disabled to full-speed. Low `recovery_max_rps` ensures Triton capacity reserved for live traffic.

**Example pipeline with both sources**:
```python
app = Application.build(config, processors=[
    LiveRedisStreamSource(connector, src_cfg, SteelInput, name="live_source"),
    RecoveryRedisStreamSource(connector, src_cfg, SteelInput, name="recovery_source"),  # optional
    CameraBranchRouter(name="router"),
    # ... rest of pipeline
])
```
If `recovery_enabled: false`, user simply doesn't add RecoveryRedisStreamSource to the list.

**Negative tests**:
- Both sources active → live messages processed immediately, pending messages processed in background
- Recovery disabled → zero pending messages processed, live unaffected
- 1000 stale messages → all dropped by recovery source, live unaffected
- Queue overflow from recovery → recovery source pauses, live source continues unblocked
- Crash → unACK'd messages reappear as pending on restart
- Recovery source empties pending buffer → goes idle, no CPU waste

- `RedisStreamSink[TCtx: Event, TResult: BaseModel]` inherits `BaseThreadSink[PipelineEvent[TCtx, TResult]]`
  - **Serialization is built into the sink** — no separate EventSerializerNode
  - Writes context fields as top-level Redis hash entries (matching TCtx signature)
  - Writes payload as single `result` JSON field (serialized TResult via `model_dump_json()`)
  - `fw` struct stripped from output — only user fields appear
  - Each model gets its own typed sink: `RedisStreamSink[SteelMeta, DetectionResult]`

### Task 2.2b: Mermaid diagram saved to file at startup
**Modify**: `engine/app.py`
- After generating mermaid diagram, save it to file `flow.md` in working directory (or path from config)
- Still log to loguru as before
- File is overwritten each start (always reflects current graph)
- Filename: `config.debug.mermaid_path: str` (required, no default — user must set explicitly, e.g. `flow.md`).

### Task 2.3: New type extraction + exact generic matching + runtime dispatch
**Modify**: `engine/registry.py`:
- Add `_extract_types_from_generics(node)` — walks __orig_bases__
- register() uses only generic extraction (drop _get_call_hints, _eval_annotations_with_bounds)
- Exact generic matching: compare via `typing.get_origin()` + `typing.get_args()` equality
- Routing table stores `list[tuple[str, type]]` (consumer_name, expected_input_type)
- `compute_routing_table()` returns typed pairs

**Modify**: `engine/app.py`:
- Remove _dispatch_cycle(), _route_event(), _drain_inline_queues(), _route_to_parallel_consumers()
- run() = validate → diagram → build routing table → wire queues → start all → supervisor loop (health check + stats + error summary)

### Error visibility on main thread

Main thread **must** log every error event — not delegate to worker threads only. The supervisor loop drains a dedicated error queue on the main thread:

1. **Error queue on QueueBus**: `_error_queue` — all error events (ConnectorError, InferenceError, PreprocessingError) are routed to this queue IN ADDITION to LogErrorSinkNode's queue. Runtime type check in `_route_to_consumers` puts errors on both queues.

2. **Supervisor loop drains error queue each cycle**:
   ```python
   def run(self):
       ...
       while not self._shutdown.is_set():
           self._drain_error_queue()      # log errors on main thread
           self._check_parallel_health()   # restart dead threads
           if self._should_log_stats():
               self._log_stats()           # periodic stats
           time.sleep(0.01)
   
   def _drain_error_queue(self):
       while True:
           try:
               error = self._error_queue.get_nowait()
           except queue.Empty:
               break
           self._log_error_on_main_thread(error)
           self._stats_errors += 1
           if self._config.debug.stop_on_error:
               self.shutdown()
   
   def _log_error_on_main_thread(self, event):
       if isinstance(event, ConnectorError):
           logger.error("[MAIN] Connector error on {}: {}", event.stream_name, event.error_message)
       elif isinstance(event, InferenceError):
           logger.error("[MAIN] Inference error on {} v{}: {}", event.model_name, event.model_version, event.error_message)
       elif isinstance(event, PreprocessingError):
           logger.error("[MAIN] Preprocessing error: {} ({})", event.error_message, event.source_node)
   ```

3. **`stop_on_error` works again** — was broken when `_log_error_event` was deleted from inline dispatch. Now the main thread sees every error and can trigger shutdown.

4. **`_log_stats` includes per-node error counts** from MetricAggregator (unchanged)
5. **`_check_parallel_health` logs restarts** on main thread (unchanged)
6. **Circuit breaker** checks error rate on main thread (unchanged)
7. **LogErrorSinkNode** still logs + feeds ErrorCollector on its worker thread (unchanged — dual logging is intentional: worker thread logs immediately, main thread logs with `[MAIN]` prefix for monitoring)

### Metrics logged to stdout (not just Redis)

Current `MetricEmitter` only writes to Redis stream (debug-level "Flushed N events" but no values). Metrics must also be visible in loguru stdout on main thread.

**Fix**: `_log_stats` on supervisor thread expanded to include MetricAggregator data per node:
```
[STATS] processed=150 errors=3 restarts=0
  inference: calls=150 errors=2 filtered=0 avg_ms=12.3
  preprocessor: calls=152 errors=1 filtered=0 avg_ms=3.1
  postprocessor: calls=148 errors=0 filtered=0 avg_ms=0.8
  source: calls=155 errors=0
  sink_det: calls=80 errors=0
  sink_cls: calls=68 errors=0
[COUNTERS] detection_done=80 classification_done=68 connector_error=1 inference_error=2 preprocessing_error=1
[GAUGES] inference_time_ms=12.3 classification_inference_time_ms=8.1
```

Implementation:
- `_log_stats()` reads from each node's `MetricAggregator` for calls/errors/filtered/avg_latency
- Also reads Counter/Gauge values from `MetricBuffer` snapshot (not flush — non-destructive read)
- Logged at INFO level on main thread via loguru
- Same data goes to Redis via `MetricEmitter` — two output paths, one data source
- Interval configurable via `config.metrics.stats_log_interval_s`
- `_route_to_consumers()` does **runtime type check**: for each `(consumer, expected_type)`, check `type(result) == expected_type` (exact, using origin+args for generics). Only matching consumers get the event.

### Processor resilience after errors

**Per-call exception handling in `_run_parallel_node`**:
- Catch **all exceptions** (`except Exception`) instead of hardcoded list — any unexpected error in `process()` must not kill the thread
- Log with `logger.exception` (full traceback)
- Record error in MetricAggregator (`node._aggregator.record_error()`)
- Sleep 0.1s before retry (prevent tight crash loops)
- The event that caused the error is **dropped** (not re-queued) — avoid poison pill loops

**Base* class `__call__` wraps `process()` with error boundary**:
```python
def __call__(self, event: TIn) -> TOut | None:
    start = time.monotonic()
    try:
        result = self.process(event)
    except Exception:
        elapsed = (time.monotonic() - start) * 1000
        self._aggregator.record_error()
        logger.exception("Error in {}.process()", self.name)
        raise  # re-raised, caught by _run_parallel_node's outer try/except
    elapsed = (time.monotonic() - start) * 1000
    self._aggregator.record_call(elapsed)
    if result is None:
        self._aggregator.record_filtered()
    elif self._metric_recorder is not None:
        _extract_and_record_metrics(result, self._metric_recorder, self.name, elapsed)
    return result
```

**Thread restart with limits and backoff**:
- `_check_parallel_health()` restarts dead threads with **exponential backoff**: delay = min(0.1 * 2^restart_count, 30s)
- **Max restarts per node**: `config.debug.max_restarts_per_node: int` (required, no default). After max, log CRITICAL and stop restarting.
- Emit metric: `{prefix}_{node_name}_restarts` counter

**`_get_stop_event` update**: With deleted inline protocols, use `BaseThread*` classes directly:
```python
def _get_stop_event(self, instance: object) -> threading.Event | None:
    if hasattr(instance, 'stop') and isinstance(instance.stop, threading.Event):
        return instance.stop
    return None
```
All Base* classes have `self.stop = threading.Event()` — this check is sufficient.

**Negative tests for resilience (add to Task 4.4)**:
- Processor that raises `ValueError` on 3rd call → thread stays alive, processes 4th call
- Processor that raises `KeyError` → thread stays alive
- Processor that raises `ValidationError` → thread stays alive
- Processor that always raises → thread restarts up to max_restarts, then stops
- After restart, processor resumes reading from its input queue (no lost messages except the poison pill)
- Metrics show error count matching number of exceptions raised

**Circuit breaker — global stop on error rate threshold**:
- Configurable per-app: `config.debug.max_errors_per_minute: int` (required, no default)
- `MetricAggregator` tracks errors in a sliding 60s window per node
- `_check_parallel_health()` (supervisor loop) checks each node's error rate:
  - If any node's errors in last 60s > `max_errors_per_minute` → log CRITICAL with node name and error count → call `self.shutdown()`
- This is a **global stop** — all nodes stop, pipeline exits
- Prevents runaway error loops from burning resources or filling logs
- Metric emitted before shutdown: `{prefix}_circuit_breaker_triggered` with node name

**Negative test**: Processor that raises on every call → after N errors in 60s, pipeline shuts down gracefully. Assert: shutdown completed, error metric present, no zombie threads.

### Task 2.4: Metrics instrumentation in Node classes + shipped Node classes

**Critical**: Current metrics work via wrappers in `engine/wrappers.py`. The `_call_with_metrics()` function handles:
1. Counter/Gauge/GroupBy extraction from result events (via `_extract_and_record_metrics`)
2. MetricAggregator per-call timing (throughput, latency, errors)
3. Recording MetricEvent to MetricBuffer

With Node classes replacing wrappers, this logic must move. Two options:
- **Option A**: Base* classes gain built-in metrics. `BaseThreadProcessor.__call__` wraps the user's logic with timing + metric extraction. Requires `metric_recorder` passed at construction.
- **Option B**: `Application.build()` wraps each node in a thin metrics decorator before registration. Node classes stay pure, metrics are applied externally.

**Choose Option A** — metrics are part of the node lifecycle, not an external concern. Base* classes get:
```python
class BaseThreadProcessor[TIn, TOut]:
    def __init__(self, name: str = "", *, metric_recorder: MetricRecorder | None = None):
        self.stop = threading.Event()
        self.name = name or type(self).__name__
        self._metric_recorder = metric_recorder
        self._aggregator = MetricAggregator()

    def __call__(self, event: TIn) -> TOut | None:
        # Definitive implementation — handles errors, metrics, and filtering
        start = time.monotonic()
        try:
            result = self.process(event)  # user overrides this
        except Exception:
            self._aggregator.record_error()
            logger.exception("Error in {}.process()", self.name)
            raise  # re-raised, caught by _run_parallel_node
        elapsed_ms = (time.monotonic() - start) * 1000
        self._aggregator.record_call(elapsed_ms)
        if result is None:
            self._aggregator.record_filtered()
        elif self._metric_recorder is not None:
            _extract_and_record_metrics(result, self._metric_recorder, self.name, elapsed_ms)
        return result

    def process(self, event: TIn) -> TOut | None:
        raise NotImplementedError
```
This is the **single definitive `__call__`** — all earlier code snippets in this plan are drafts superseded by this one.

`BaseThreadSink` and `BaseThreadSource` follow the same pattern:
- `BaseThreadSink.__call__` wraps `process()` with timing + error recording (no result to extract metrics from)
- `BaseThreadSource.__call__` wraps `produce()` with timing + error recording
- User overrides `process()` / `produce()`, never `__call__()`

- User overrides `process()` not `__call__()` — `__call__` handles metrics
- `Application.build()` passes `metric_recorder` to each node at construction
- `InstrumentedNode` protocol still works — Base* classes expose `aggregator` property
- Counter/Gauge/GroupBy on event classes continue to work via `_extract_and_record_metrics`
- MetricAggregator integrates into Base* classes for per-node throughput/latency

**Also**: `engine/wrappers.py` is **deleted** — its logic is now in Base* classes. The `_call_with_metrics()` and `_extract_and_record_metrics()` functions move to `engine/metric_aggregator.py` (or a new `engine/metrics_integration.py`).

**Modify**: `engine/shipped_processors.py` — delete old classes (EnvelopeExtractor, etc.). Add Node classes:
```python
class DaliPreprocessorNode[TCtx: Event, TPayload: ImagePayload](
    BaseThreadProcessor[PipelineEvent[TCtx, TPayload], PipelineEvent[TCtx, TritonRequest]]
)
class ClientPreprocessorNode[TCtx: Event, TPayload: ImagePayload](
    BaseThreadProcessor[PipelineEvent[TCtx, TPayload], PipelineEvent[TCtx, TritonRequest]]
)
class InferenceNode[TCtx: Event, TReq: TritonRequest, TOut: InferenceOutput](
    BaseThreadProcessor[PipelineEvent[TCtx, TReq], PipelineEvent[TCtx, TOut] | InferenceError]
)
class PostProcessorNode[TCtx: Event, TIn: InferenceOutput, TResult: BaseModel](
    BaseThreadProcessor[PipelineEvent[TCtx, TIn], PipelineEvent[TCtx, TResult]]
)
class LogErrorSinkNode(BaseThreadSink[InferenceError | ConnectorError | PreprocessingError])
```

### Error handling architecture

**Problem in current plan**: errors are silently lost or uncounted.

1. **All error types must have Counters**:
```python
class ConnectorError(BaseModel):
    _metrics_ = [Counter("connector_error")]
class InferenceError(BaseModel):
    _metrics_ = [Counter("inference_error")]
class PreprocessingError(BaseModel):
    _metrics_ = [Counter("preprocessing_error")]  # was decode_failure
```

2. **Per-processor error counting via MetricAggregator**: Base* classes track errors in `process()`:
```python
class BaseThreadProcessor[TIn, TOut]:
    def __call__(self, event: TIn) -> TOut | None:
        start = time.monotonic()
        try:
            result = self.process(event)
        except Exception:
            self._aggregator.record_error()  # per-node error count
            raise
        elapsed = (time.monotonic() - start) * 1000
        self._aggregator.record_call(elapsed)  # throughput + latency
        if result is None:
            self._aggregator.record_filtered()  # dropped events count
        return result
```

3. **Errors carry context** — when InferenceNode or PreprocessorNode produces an error, it should include which input caused it. Error events gain `source_node: str` field:
```python
class InferenceError(BaseModel):
    _metrics_ = [Counter("inference_error")]
    source_node: str        # which processor produced this error
    model_name: str
    model_version: str
    error_type: str
    error_message: str
    retry_count: int
    timestamp_ms: int
```

4. **Preprocessor errors are NOT silent drops** — when `ClientPreprocessorNode.process()` fails to read/decode an image, it must return `PreprocessingError` instead of `None`. Node signature becomes:
```python
class ClientPreprocessorNode[TCtx: Event, TPayload: ImagePayload](
    BaseThreadProcessor[
        PipelineEvent[TCtx, TPayload],
        PipelineEvent[TCtx, TritonRequest] | PreprocessingError
    ]
)
```

5. **LogErrorSinkNode feeds ErrorCollector** — not just logging. ErrorCollector aggregates counts, ErrorHandler logs periodic summaries:
```python
class LogErrorSinkNode(BaseThreadSink[InferenceError | ConnectorError | PreprocessingError]):
    def __init__(self, name: str = "", *, error_collector: ErrorCollector | None = None):
        self._collector = error_collector

    def process(self, event: ...):
        logger.error("Pipeline error: {}: {}", type(event).__name__, event.error_message)
        if self._collector:
            self._collector.record(event)
```

6. **Metrics emitted for errors**:
- `{prefix}_connector_error` — counter per occurrence
- `{prefix}_inference_error` — counter per occurrence
- `{prefix}_preprocessing_error` — counter per occurrence
- `{prefix}_{node_name}_errors` — per-processor error count from MetricAggregator
- `{prefix}_{node_name}_filtered` — per-processor dropped/None count

---

## Wave 3 — Two parallel tracks (after Wave 1 + Wave 2)

**Track A** — engine integration (needs Wave 2):
### Task 3.1: Typed DetectionResult + ClassificationResult
**Modify**: `engine/events.py` — DetectionResult with `detections: tuple[Detection, ...]` (no detections_json), ClassificationResult with `predictions: tuple[ClassificationPrediction, ...]`

### Task 3.2: Application.build() classmethod
**Modify**: `engine/app.py` — see API below

### Task 3.3: Build validation
**New file**: `engine/validation.py`

### Task 3.4: ModelBundle + expand_bundle
**New file**: `engine/bundle.py`

**Track B** — model_tools (needs only Wave 1, NO engine deps, runs in parallel with Track A):

**Track B1 — independent tasks (all parallel)**:
### Task 3.5: ClassificationPostProcessorBridge
**New file**: `triton_model_tools/inference/classification_bridge.py`

### Task 3.6: Update YoloPostProcessorBridge for typed DetectionResult
**Modify**: `triton_model_tools/inference/yolo_bridge.py` — return tuple[Detection, ...]

### Task 3.7: ResNet18 ONNX export + Triton repository
**New file**: `triton_model_tools/export/classification_exporter.py`

### Task 3.8: All-class scores — extended processor variants (full vertical slice)

All-scores is NOT a flag on existing processors. It's a separate set of extended processors that carry raw class scores end-to-end. Requires changes at every layer:

**1. Export layer** — ONNX model must NOT apply NMS internally (no end2end). Raw class scores must be in the output tensor.
- `triton_model_tools/export/exporter.py` — add `export_raw_scores: bool` option that disables end2end NMS at export time
- `config.pbtxt` output shape includes full class dimension

**2. Shared models** — extended Detection type:
- `triton_shared_models/detection/models.py` — `Detection.all_class_scores` already in plan (Task 1.1). No extra changes.
- New: `DetailedClassificationPrediction(ClassificationPrediction)` with `all_probabilities: tuple[float, ...]` if needed per-prediction

**3. Postprocessor layer** — extended postprocessor:
- `triton_model_tools/inference/postprocessing.py` — new `YoloDetailedPostprocessor(YoloPostprocessor)`:
  - Inherits base NMS logic
  - After NMS, attaches raw class score vector to each surviving Detection's `all_class_scores`
  - Separate class, not a flag — base `YoloPostprocessor` stays unchanged
- `triton_model_tools/inference/classification_postprocessing.py` — `DetailedClassificationPostprocessor`:
  - Returns full probability vector alongside top-K

**4. Bridge layer** — extended bridge:
- `triton_model_tools/inference/yolo_bridge.py` — new `YoloDetailedPostProcessorBridge`:
  - Uses `YoloDetailedPostprocessor` internally
  - Returns `DetectionResult` with `all_class_scores` populated on each Detection
- `triton_model_tools/inference/classification_bridge.py` — `DetailedClassificationPostProcessorBridge`:
  - Returns `ClassificationResult` with `all_probabilities` populated

**5. Pipeline Node layer** — no changes needed. `PostProcessorNode` is generic over result type. User chooses which bridge to pass.

**6. CLI** — extended commands:
- `uv run triton-model-tools infer --all-scores` uses `YoloDetailedPostprocessor`
- `uv run triton-model-tools local-infer --all-scores` same

**Key principle**: base processors stay simple. Extended variants inherit and add. User explicitly opts in by choosing the detailed bridge when building the pipeline.

### Task 3.8c: TensorRT local inference support (optional dependency)

True TensorRT support — local inference without Triton, same pattern as DALI optional dep.

**1. Optional dependency** in `packages/triton_model_tools/pyproject.toml`:
```toml
[project.optional-dependencies]
dali = ["nvidia-dali-cuda120>=1.35.0"]
tensorrt = ["tensorrt>=10.0", "cuda-python>=12.0"]
```
Install: `uv pip install triton-model-tools[tensorrt]`

**2. Local TensorRT inference client**:
**New file**: `triton_model_tools/inference/tensorrt_inference.py`
- `TensorRTInferenceClient`: loads `.plan` file, runs inference via TensorRT Python API
  - `__init__(model_path: Path)` — loads engine, allocates buffers
  - `infer(tensor: NDArray) -> dict[str, NDArray]` — same interface as `OnnxInferenceClient`
  - Handles FP16/FP32 input based on engine binding dtype
- Import guarded: `try: import tensorrt` with clear error if missing
- Skip-aware: all tests/CLI using TRT client check availability first

**3. Local TensorRT folder inference**:
**New file**: `triton_model_tools/inference/local_trt_folder_inference.py`
- `run_local_trt_folder_inference(...)` — same interface as `run_local_folder_inference` but uses `TensorRTInferenceClient`
- For both detection and classification

**4. CLI extension**:
- `uv run triton-model-tools local-infer --backend tensorrt --model-path model.plan`
- Falls back to ONNX if TensorRT unavailable with clear message

**5. Python API extension** (in `triton_model_tools/api.py`):
```python
def run_local_detection(images, onnx_path=None, trt_path=None, ...) -> list[ImageDetectionResult]
def run_local_classification(images, onnx_path=None, trt_path=None, ...) -> list[ImageClassificationResult]
```
Accepts either ONNX or TRT path. Selects client accordingly.

**6. Tests** (`packages/triton_model_tools/tests/test_tensorrt_inference.py`):
- `trt_available` fixture — skip if tensorrt not installed
- Test: load .plan model, run inference, compare output against ONNX reference
- Test: FP16 .plan model produces results within tolerance of FP32 ONNX
- Test: invalid .plan file → clear error, not crash

**7. Benchmarks** — add TRT backend:
- `BACKENDS: list[str] = ["onnx", "tensorrt"]` (already exists in pipeline_benchmark.py)
- perf_analyzer tasks (6.1, 6.2) run both ONNX and TRT variants
- Dynamic batching drift test (6.6) includes TRT variant

**8. Drift verification ONNX vs TRT**:
- Export same model to both ONNX and .plan
- Run inference on same images with both clients
- `compare_inference()` / `compare_classification()` — assert precision/recall/accuracy ≥ 0.99
- This catches export bugs where TRT conversion introduces drift

**9. Test fixtures**: `restore_models.py` also generates TRT variants (if tensorrt available):
```
test_fixtures/models/
  yolo26s/1/model.onnx          # restored by script
  yolo26s_trt/1/model.plan      # restored by script (skipped if no tensorrt)
  resnet18/1/model.onnx
  resnet18_trt/1/model.plan     # .plan files are GPU-specific, always generated locally
```
All weights .gitignore'd. `restore_models.py` is the single source for all test models.

### Task 3.8d: Export variant verification tests
**Depends on**: 3.7 (ResNet18 export), 3.8c (TRT client), 3.12 (test fixtures)
Verify all YOLO export combinations produce correct models:

**Export matrix to test**:
| Variant | Format | FP16 | end2end | rect | Expected |
|---|---|---|---|---|---|
| yolo26s | ONNX | no | no | no | Standard NMS-free output, square input |
| yolo26s | ONNX | yes | no | no | FP16 ONNX via ultralytics `half=True` export |
| yolo26s | ONNX | no | yes | no | end2end output (batch, dets, 6), NMS inside model |
| yolo26s | ONNX | no | no | yes | Rectangular input (e.g., 960x608) |
| yolo26s | TRT | yes | no | no | FP16 .plan, standard output |
| yolo26s | TRT | yes | yes | no | FP16 .plan, end2end |
| resnet18 | ONNX | no | n/a | no | Classification output (1, 1000) |
| resnet18 | ONNX | yes | n/a | no | FP16 ONNX via `half=True` |
| resnet18 | TRT | yes | n/a | no | FP16 classification |

**Checks per variant**:
1. Export succeeds without error
2. config.pbtxt has correct input/output shapes and dtypes
3. **No double NMS**: if `end2end=True`, model includes NMS — postprocessor must NOT apply NMS again. Test: export with end2end, run inference, check output shape is `(batch, num_dets, 6)` not raw class scores
4. **Rect support**: if `rect=True`, input tensor shape matches non-square dims. Preprocessing must use correct target_size, not assume square
5. **FP16 ONNX**: exported via ultralytics `half=True` — verify ONNX file has FP16 weights, inference produces valid results within tolerance of FP32 ONNX
6. **FP16 TRT**: TRT FP16 .plan — verify inference within tolerance of FP32
7. **Drift check**: for each format pair (ONNX FP32 vs ONNX FP16, ONNX vs TRT, FP32 vs FP16), run same images, compare results, assert precision/recall ≥ 0.99

**New test file**: `packages/triton_model_tools/tests/test_export_variants.py`
- Parametrized over the matrix above
- Uses `test_fixtures/images/` for inference verification
- Marks TRT tests with `@pytest.mark.skipif` if tensorrt unavailable

### Task 3.8b: Preprocessing verification tools
**Depends on**: 3.12 (test fixtures with images)
**New files**:
- `triton_model_tools/testing/preprocessing_verify.py`:
  - `verify_yolo_preprocessing(weights_path, config, images) -> list[PreprocessingResult]`
    - Loads YOLO weights + config, runs preprocessing (letterbox + normalize), returns tensor + scale + pad_offset per image
    - Compares against reference ultralytics preprocessing — asserts pixel-level match
  - `verify_dali_preprocessing(model_repo, images) -> list[PreprocessingResult]`
    - Runs DALI ensemble preprocessing via Triton (requires running Triton)
    - Compares against client-side preprocessing — asserts output tensor match within tolerance
    - Gracefully skips if DALI/Triton unavailable (`@pytest.mark.skipif` with DALI availability check)
  - `verify_classification_preprocessing(images) -> list[PreprocessingResult]`
    - Runs CenterCropPreprocessor, compares against torchvision reference transforms
  - `PreprocessingResult(BaseModel)`: image_path, tensor_shape, tensor_hash, scale, pad_offset

- `triton_model_tools/testing/conftest.py` — add `dali_available` fixture:
  ```python
  @pytest.fixture
  def dali_available():
      try:
          import nvidia.dali  # noqa: F401
          return True
      except ImportError:
          pytest.skip("DALI not available")
  ```

- `packages/triton_model_tools/tests/test_preprocessing_drift.py`:
  - Test YOLO preprocessing matches ultralytics reference
  - Test DALI preprocessing matches client-side (skip if unavailable)
  - Test classification preprocessing matches torchvision reference
  - All tests compare on real images, assert max pixel diff < threshold

### Task 3.9: Python API for model_tools + examples
**New file**: `triton_model_tools/api.py` — public API wrapping internal modules
**Depends on**: 3.5, 3.6, 3.7 (bridges + export for API to wrap)

**Track B2 — tests that depend on B1 + Track C fixtures (run after B1 + C complete)**:

### Task 3.10: Drift test (detection) — broken ONNX model
**Depends on**: 3.6 (YoloBridge), 3.12 (test fixtures with corrupted models)

### Task 3.11: Drift test (classification) — broken ResNet18 ONNX
**Depends on**: 3.5 (ClassificationBridge), 3.7 (ResNet18 export), 3.12 (test fixtures)

### Task 3.11b: Full pipeline drift test against original ultralytics
**Depends on**: 3.5, 3.6, 3.7, 3.8c (TRT client), 3.12 (test fixtures)

End-to-end drift verification: our entire pipeline (preprocessing → export → Triton inference → postprocessing) must produce results identical to running ultralytics directly. Any drift means a bug in our pipeline.

**Detection drift test**:
1. Run `ultralytics.YOLO(model).predict(images)` — get reference detections (bboxes, classes, confidences)
2. Export same model via our exporter (ONNX FP32, ONNX FP16, TRT FP16)
3. For each export variant:
   a. Run through our full pipeline: `LetterBoxPreprocessor → TritonClient.infer → YoloPostprocessor`
   b. Run through Redis pipeline: `RedisStreamSource → ClientPreprocessorNode → InferenceNode → PostProcessorNode → RedisStreamSink` (E2E, requires Docker)
   c. Compare against ultralytics reference: `compare_inference()` — assert precision ≥ 0.99, recall ≥ 0.99, avg_iou ≥ 0.99
4. Also compare our standalone tools: `run_detection(images)` from Python API vs ultralytics reference

**Classification drift test**:
1. Run `torchvision.models.resnet18(pretrained=True)` on images — get reference top-K predictions
2. Export via our classification exporter (ONNX FP32, ONNX FP16, TRT FP16)
3. For each variant:
   a. Tool path: `CenterCropPreprocessor → OnnxInferenceClient/TensorRTInferenceClient → ClassificationPostprocessor`
   b. Pipeline path: full Redis pipeline E2E
   c. Compare against torchvision reference: `compare_classification()` — assert top-1 accuracy ≥ 0.99, top-5 accuracy ≥ 0.99

**What this catches** (that other tests don't):
- Preprocessing drift: our letterbox/center-crop differs from ultralytics/torchvision
- Export drift: ONNX/TRT conversion introduces numerical errors beyond tolerance
- Postprocessing drift: our NMS implementation differs from ultralytics NMS
- Pipeline integration drift: serialization/deserialization changes tensor values
- FP16 quantization drift: half-precision introduces too much error

**Manual verification step** (in E2E guide):
```bash
# 1. Run ultralytics reference
uv run python -c "
from ultralytics import YOLO
model = YOLO('yolo11s.pt')
results = model.predict('test_fixtures/images/', save_txt=True, project='test_fixtures', name='ultralytics_ref')
"

# 2. Run our pipeline on same images
uv run triton-model-tools infer --model yolo11s --images test_fixtures/images/ --output test_fixtures/our_results/

# 3. Compare
uv run triton-model-tools compare-folders \
    --baseline test_fixtures/ultralytics_ref/labels/ \
    --candidate test_fixtures/our_results/ \
    --iou-threshold 0.5

# What to look for:
# - precision ≥ 0.99
# - recall ≥ 0.99
# - avg_iou ≥ 0.99
# - class_mismatch_rate = 0.0
# - If ANY metric is below threshold → our pipeline has a bug
```

**Test file**: `packages/triton_model_tools/tests/test_pipeline_drift_vs_ultralytics.py`
- `@pytest.mark.slow` — runs full inference
- Parametrized over: model variant x export format x FP precision
- Requires `test_fixtures/models/` restored
- TRT tests skip if tensorrt unavailable

**Track C** — config/infra (no deps on engine or model_tools):
### Task 3.12: Consolidate test fixtures to repo root
- Create `test_fixtures/` at repo root
- Add 8-10 COCO8 images to `test_fixtures/images/` (committed to git)
- Create `test_fixtures/restore_models.py` (committed) — script that exports yolo26s + resnet18 ONNX, creates corrupted variants
- Add `test_fixtures/.gitignore` — ignores `models/` and `corrupted_models/` (weights not in git, no lfs)
- Root `conftest.py` — images always available, models skip with clear "Run restore_models.py" message
- Update all package `conftest.py` to use root fixtures (delete scattered coco8 candidate lists)
- Update all examples to reference `test_fixtures/images/`

### Task 3.13: Config documentation generation (was 3.12)
**New file**: `triton_redis_connector/config/docs_generator.py`
- Introspect pydantic config models via model_json_schema()
- Generate markdown with all fields, types, required markers (no defaults)

### Task 3.14: Simplify Hydra — only hydra.main
**Modify**: `config/hydra_app.py` — delete compose API, only `@hydra.main`

### Task 3.15: Metrics naming convention
Update MetricEmitter to prepend one of `{APP_NAME}_` or `{HOSTNAME}_` at flush time (app_name preferred, hostname fallback). Only `_` separator.

---

## Wave 4 — Examples + Integration Tests (after Wave 3)

### Application.build() API (defined in Task 3.2):
```python
@classmethod
def build(
    cls,
    config: AppConfig,
    processors: list[object],
    *,
    triton_collector: LifecycleService | None = None,
) -> Application:
```
- Node name from `__init__(name=...)` or class name
- Auto-creates MetricBuffer, MetricEmitter from config
- Runs validation (Task 3.3)
- Deletes `providers/app_factory.py` entirely

### Build validation (defined in Task 3.3):
1. At least one source  2. At least one sink
3. Every source output has consumer  4. Every TritonRequest → InferenceNode
5. Every InferenceOutput → postprocessor  6. Every result PipelineEvent → typed sink
7. triton_collector present  8. Graph connected  9. No orphans
10. TCtx inherits from Event (has `fw: FrameworkMeta`)  11. TCtx no field named `result`

### Task 4.1: Rewrite dual_model_pipeline
Uses Application.build(), PipelineEvent[SteelMeta, ...], typed sinks

### Task 4.2: New multi_task_pipeline (detection + classification)
Two ModelBundles, shared InferenceWorker, separate typed sinks

### Task 4.3: Fix existing tests
### Task 4.4: Negative tests (break everything intentionally)
### Task 4.5: Metrics resend verification test

---

## Wave 5 — E2E Verification (after Wave 4, requires Docker)

### Task 5.1: E2E test — single model detection pipeline
1. Start Redis + Triton (Docker)
2. Export YOLO model to Triton repository
3. Create pipeline: RedisStreamSource[ImageEvent] → BusinessProcessor → ClientPreprocessorNode → InferenceNode → PostProcessorNode → RedisStreamSink[TCtx, TResult]
4. Populate input stream with 50+ test images
5. Read output stream, deserialize context + result
6. Assert: all images processed, DetectionResult has detections, context fields preserved exactly
7. Assert: no `fw` field in Redis output (framework data stripped by sink)
8. Assert: no data override — user fields match input exactly

### Task 5.2: E2E test — single model classification pipeline
Same as 5.1 but with ResNet18 + ClassificationResult

### Task 5.3: E2E test — multi-model pipeline (detection + classification)
1. Two ModelBundles (YOLO + ResNet18) sharing Triton
2. TaskRouter splits by type
3. Assert: detection results go to detection sink only
4. Assert: classification results go to classification sink only
5. Assert: no cross-contamination between sinks
6. Assert: context preserved in both branches

### Task 5.4: E2E test — error handling + error metrics
1. Send malformed JSON to input stream → ConnectorError produced, pipeline continues
2. Send valid event but Triton down → InferenceError produced, error sink receives it
3. Send image with non-existent path → PreprocessingError produced (not silent None)
4. Assert: pipeline recovers from all errors without crash
5. Assert: metrics stream has `connector_error`, `inference_error`, `preprocessing_error` counters > 0
6. Assert: per-node error counts in metrics (`{node_name}_errors` > 0 for failing nodes)
7. Assert: ErrorCollector summary shows correct error type distribution

### Task 5.5: E2E test — metrics verification
1. Run pipeline with Triton metrics collector
2. Assert: metrics Redis stream has events with `{prefix}_*` format (underscore only, prefix = app_name or hostname)
3. Assert: Triton metrics (inference_count, queue_time) present and non-zero
4. Assert: Counter metrics (detection_done, classification_done) present
5. Assert: Gauge metrics (inference_time_ms) present with reasonable values

### Task 5.6: E2E test — user data preservation round-trip
1. Create event with fields named: source_stream, message_id, created_at, receipt_ts, timestamp, result_data
2. Push through full pipeline
3. Assert: ALL user fields arrive at output unchanged (no framework override)
4. Assert: no `fw` field in Redis output (stripped by sink)

### Task 5.7: E2E test — build validation catches broken configs
1. Try building with missing source → assert raises with clear message
2. Try building with missing sink → assert raises
3. Try building with unconnected InferenceNode → assert raises
4. Try building without triton_collector → assert raises
5. Try defining TCtx that redefines `fw` field → pydantic raises at class definition time
6. Try building with TCtx field named `result` → assert raises

---

## Final: E2E Verification Guide for User

**New file**: `docs/e2e-verification-guide.md` — step-by-step document for the user to verify everything works.

### Section 0: Test Fixtures — Committed to Repo at Top Level
All test data lives at repo root in `test_fixtures/`. **Committed to git** — tests work immediately after clone, no downloads.

```
test_fixtures/
  images/                     # 8-10 real JPEGs (~2MB), committed to git
    000000000009.jpg
    000000000025.jpg
    000000000030.jpg
    ...
  models/                     # .gitignore'd — restored via script
    yolo26s/1/model.onnx
    resnet18/1/model.onnx
  corrupted_models/           # .gitignore'd — restored via script
    yolo26s_broken/1/model.onnx
    resnet18_broken/1/model.onnx
  restore_models.py           # committed — exports/creates all models
  .gitignore                  # ignores models/ and corrupted_models/
```

- **Images committed to git** — always available
- **Images committed** — 8-10 JPEGs (~2MB total)
- **Models NOT committed** — ONNX/TRT weights are too large for git (no git-lfs). `.gitignore`'d.
- **Models restored via script**: `test_fixtures/restore_models.py` — exports/downloads models into `test_fixtures/models/` and `test_fixtures/corrupted_models/`
  ```bash
  uv run python test_fixtures/restore_models.py
  ```
  - Exports yolo26s ONNX via ultralytics
  - Exports resnet18 ONNX via torchvision
  - Creates corrupted variants (zero out conv weights)
  - Idempotent — skips if models already exist
- **`test_fixtures/.gitignore`**: ignores `models/` and `corrupted_models/`, commits `images/` and `restore_models.py`
- If models missing, tests that need them print clear error: `"Run: uv run python test_fixtures/restore_models.py"` then skip
- Benchmarks that need more images cycle through `test_fixtures/images/`

**Root `conftest.py`** — single fixture path for all packages:
```python
TEST_FIXTURES = Path(__file__).resolve().parent / "test_fixtures"
RESTORE_HINT = "Run: uv run python test_fixtures/restore_models.py"

@pytest.fixture
def test_images_dir() -> Path:
    return TEST_FIXTURES / "images"

@pytest.fixture
def test_models_dir() -> Path:
    path = TEST_FIXTURES / "models"
    if not path.is_dir() or not any(path.rglob("*.onnx")):
        pytest.skip(f"Models not found. {RESTORE_HINT}")
    return path

@pytest.fixture
def corrupted_models_dir() -> Path:
    path = TEST_FIXTURES / "corrupted_models"
    if not path.is_dir() or not any(path.rglob("*.onnx")):
        pytest.skip(f"Corrupted models not found. {RESTORE_HINT}")
    return path
```

**All references unified** — replace every `datasets/coco8/`, `~/datasets/`, `/workspace/test_data/` with `test_fixtures/`.

### Section 1: Prerequisites
```
- Docker running (Redis + Triton containers)
- uv sync completed
- test_fixtures/images/ present (committed to repo)
- Model weights restored via script
- Models exported to model_repository/ for Triton
```
Commands:
```bash
docker compose -f .devcontainer/docker-compose.debug.yml up -d
uv sync
uv run python test_fixtures/restore_models.py              # restore ONNX weights (not in git)
uv run triton-model-tools convert --model yolo26s --output model_repository/
uv run python -m triton_model_tools.export.classification_exporter --output model_repository/
```

### Section 2: Run Single-Model Detection Pipeline
```bash
# 1. Populate input stream
uv run triton-model-tools populate-stream --stream steel:input --images test_fixtures/images/ --count 50

# 2. Start pipeline
uv run python examples/dual_model_pipeline/run_pipeline.py

# 3. Verify: check flow.md was created
cat flow.md

# 4. Verify: read output stream in another terminal
uv run python -c "
from redis import Redis
r = Redis()
results = r.xrange('steel:even_results', count=5)
for msg_id, fields in results:
    print(msg_id, {k.decode(): v.decode()[:80] for k, v in fields.items()})
"

# What to look for:
# - flow.md file exists with mermaid graph
# - Output stream has entries
# - Each entry has context fields (camera_no, metal_type, image_path) as top-level keys
# - Each entry has 'result' field with JSON DetectionResult (detections array)
# - No 'fw' key in output (framework data stripped by sink)
```

### Section 3: Run Multi-Model Pipeline (Detection + Classification)
```bash
# 1. Populate
uv run triton-model-tools populate-stream --stream multi:input --images test_fixtures/images/ --count 50

# 2. Start
uv run python examples/multi_task_pipeline/run_pipeline.py

# 3. Verify detection output
uv run python -c "
from redis import Redis
r = Redis()
det = r.xrange('multi:detection_results', count=3)
cls = r.xrange('multi:classification_results', count=3)
print(f'Detection results: {len(det)}')
print(f'Classification results: {len(cls)}')
# Verify no cross-contamination:
for _, fields in det:
    result = fields[b'result'].decode()
    assert 'detections' in result, 'Detection result must have detections'
    assert 'predictions' not in result, 'Detection stream must not have classification results'
for _, fields in cls:
    result = fields[b'result'].decode()
    assert 'predictions' in result, 'Classification result must have predictions'
    assert 'detections' not in result, 'Classification stream must not have detection results'
print('No cross-contamination: PASS')
"
```

### Section 4: Verify Metrics
```bash
# Read metrics stream
uv run python -c "
from redis import Redis
r = Redis()
metrics = r.xrange('metrics:pipeline', count=20)
for msg_id, fields in metrics:
    name = fields.get(b'component_name', b'').decode()
    measure = fields.get(b'measure', b'').decode()
    value = fields.get(b'value', b'').decode()
    print(f'{name}_{measure} = {value}')
"

# What to look for:
# - Metrics with app_name prefix (e.g., steel_inspector_detection_done)
# - Only '_' separator, no dots
# - triton_* metrics present (inference_count, queue_time)
# - Counter values > 0
# - Gauge values (inference_time_ms) are reasonable (1-500ms range)
```

### Section 5: Verify Preprocessing Consistency
```bash
# YOLO preprocessing vs ultralytics reference
uv run pytest packages/triton_model_tools/tests/test_preprocessing_drift.py -v -k yolo

# Classification preprocessing vs torchvision reference
uv run pytest packages/triton_model_tools/tests/test_preprocessing_drift.py -v -k classification

# DALI preprocessing (requires Triton with DALI model loaded)
uv run pytest packages/triton_model_tools/tests/test_preprocessing_drift.py -v -k dali
# (skips automatically if DALI unavailable)
```

### Section 6: Verify Model Drift Detection
```bash
# Broken YOLO model — drift tools must detect it
uv run pytest packages/triton_model_tools/tests/test_model_drift.py -v -k detection --tb=short

# Broken ResNet18 — drift tools must detect it
uv run pytest packages/triton_model_tools/tests/test_model_drift.py -v -k classification --tb=short

# What to look for:
# - Tests PASS (meaning drift WAS detected in broken models)
# - Comparison metrics show precision/recall/accuracy dropped significantly
```

### Section 7: Verify Build Validation
```bash
# All negative cases must raise clear errors
uv run pytest tests/test_engine/test_build_validation.py -v --tb=short

# What to look for:
# - Every test PASSES (meaning build correctly rejected invalid configs)
# - Error messages are clear and actionable
```

### Section 8: Verify User Data Preservation
```bash
# Round-trip test: user fields must not be overwritten by framework
uv run pytest -v -k "user_data_preservation or round_trip" --tb=short

# What to look for:
# - Fields like source_stream, message_id, receipt_ts survive pipeline unchanged
# - Framework fw struct stripped from output, never appears in Redis
```

### Section 9: Run Full Test Suite
```bash
uv run pre-commit run --all-files   # all linters + type checkers
uv run pytest                        # all unit tests
uv run pytest -m slow               # drift + preprocessing tests
uv run pytest -m e2e                # E2E tests (requires Docker)
```

### Section 10: Python API Smoke Test
```bash
# Detection via Triton
uv run python examples/python_api/detect_images.py --images test_fixtures/images/ --model yolo26s

# Classification via Triton
uv run python examples/python_api/classify_images.py --images test_fixtures/images/ --model resnet18

# Compare two models for drift
uv run python examples/python_api/compare_models.py --baseline yolo26s --candidate yolo26s_trt

# What to look for:
# - Detection prints bounding boxes with confidence > 0
# - Classification prints top-5 class predictions
# - Comparison shows precision/recall ≥ 0.99 for same model
```

### Quick Checklist
| Check | Command | Expected |
|---|---|---|
| Linters pass | `uv run pre-commit run --all-files` | All green |
| Unit tests | `uv run pytest` | All pass |
| flow.md created | Run any example, check file | Mermaid diagram present |
| Detection pipeline | Example + read output stream | Entries with detections |
| Classification pipeline | Example + read output stream | Entries with predictions |
| No cross-contamination | Multi-model + check both streams | Each stream has only its type |
| Metrics in Redis | Read metrics stream | Prefixed, underscore-only |
| Triton metrics | Read metrics stream | triton_* present |
| YOLO preprocessing | pytest test_preprocessing_drift | Match reference |
| Drift detection works | pytest test_model_drift | Broken model detected |
| Build rejects bad config | pytest test_build_validation | All rejections work |
| User data preserved | pytest round_trip | No overwrite |

---

## GPU/Memory Budget for Tests and Benchmarks

**Never load all models into Triton simultaneously.** Each test/benchmark stage loads only the models it needs, then unloads before the next stage.

- YOLO ONNX + ResNet18 ONNX — fits in 20GB VRAM together (both small)
- YOLO TRT FP16 + ResNet18 TRT FP16 — fits together
- **Never**: YOLO ONNX + YOLO TRT + ResNet18 ONNX + ResNet18 TRT all at once
- Use Triton model control API (`--model-control-mode=explicit`) to load/unload per test
- E2E tests (Wave 5): load only models needed for that specific test, unload after
- Benchmarks (Wave 6): one model at a time for perf_analyzer, two max for concurrent test
- Dynamic batching test: same model loaded once, not both batched and non-batched variants simultaneously
- If test needs to compare ONNX vs TRT: run ONNX first, collect results, unload, load TRT, run, compare offline

---

## Wave 5b — Code Quality Review (after Wave 5, before benchmarks)

Full codebase review after all functional changes are done. Catch what linters miss.

### Task 5b.1: Dead code and unused imports sweep
- `uv run vulture .` — fix all findings
- Manual review: any functions/classes that survived deletion but are no longer called
- Check all `__init__.py` exports match actual public API
- Verify no re-exports of deleted classes remain

### Task 5b.2: Type consistency review
- `uv run mypy . && uv run pyright` — must be zero errors
- Manual review: any `type: ignore` comments that were added as workarounds — resolve properly
- Verify all generic params are `Event` not `BaseModel` where required
- Verify all Node classes have correct `__orig_bases__` (test with `typing.get_args`)
- Check no `Any` leaked into signatures that should be typed

### Task 5b.3: Import layer contract verification
- `uv run lint-imports` — must pass
- Review: new files (detection/, classification/, bundle.py, validation.py) added to correct layers
- Verify shared_models has zero imports from connector or model_tools
- Verify connector has zero imports from model_tools
- Check no circular imports introduced

### Task 5b.4: Naming and structure review
- Consistent naming: all Node classes end with `Node`, all events end with `Result`/`Error`/`Input`
- No orphan files — every .py file is imported somewhere or is a test/example entry point
- No duplicate logic — shipped Node classes don't duplicate code from each other (use base methods)
- Config field names consistent with code (no mismatched snake_case)

### Task 5b.5: Documentation accuracy review
- Read every doc file, verify it matches actual code:
  - `docs/architecture.md` — data flow diagram matches real pipeline
  - `docs/parallelism.md` — variant list matches actual ExecutionVariant enum
  - `CLAUDE.md` — shipped processors table matches actual Node classes
  - `README.md` — usage examples actually run
  - `docs/config-reference.md` — regenerate and diff against committed version
- Agent/skill instructions — verify code examples compile
- Delete any doc sections that reference removed concepts

### Task 5b.6: Test coverage gaps
- Run `uv run pytest --cov` — identify uncovered lines in new code
- Every new Node class has at least: happy path test, error path test, None/filtered test
- Every new config field has validation test (required, out-of-range)
- PipelineEvent serialization round-trip covered
- FrameworkMeta field collision test covered
- Recovery source: all modes tested (enabled, disabled, empty pending, stale TTL)

### Task 5b.7: Security and resource review
- No bare `except Exception` that swallows silently (ruff BLE001 catches this but verify)
- No unbounded memory growth (queues have maxsize, pending buffer bounded)
- No thread leaks (all threads are daemon or properly joined on shutdown)
- Redis connections properly closed on shutdown
- gRPC channels properly closed
- No secrets in config examples or test fixtures

---

## Wave 6 — Performance Benchmarking (resource constraint: 30GB RAM / 20GB VRAM)

### Task 6.1: perf_analyzer YOLO
### Task 6.2: perf_analyzer ResNet18
### Task 6.3: E2E pipeline perf — detection only
### Task 6.4: E2E pipeline perf — classification only
### Task 6.5: E2E pipeline perf — both models concurrent
### Task 6.6: Dynamic batching — no drift verification
- Re-export with dynamic_batching, cycle images to 1000+, compare against baseline, assert precision/recall/accuracy ≥ 0.99

---

## Parallelism Summary

```
Wave 1 (6 tasks ALL parallel) + Wave 2 (4 sequential, parallel with Wave 1):
  1.1-1.6: shared_models types, preprocessing, comparison, TritonRequest rename
  2.1→2.2→2.3→2.4: drop inline → PipelineEvent → type extraction → Node classes

Wave 3 — 3 tracks:
  Track A (engine, sequential): 3.1→3.2→3.3→3.4
  Track B1 (model_tools, parallel): 3.5, 3.6, 3.7, 3.8, 3.8c (bridges, export, all_scores, TRT)
  Track C (config/infra, parallel): 3.12 test fixtures, 3.13 config docs, 3.14 hydra, 3.15 metrics
  Track B2 (after B1+C): 3.8b, 3.8d, 3.9, 3.10, 3.11, 3.11b (preproc verify, export verify, API, drift tests)

Wave 4 (5 tasks, ALL parallel):
  4.1-4.2 Examples  4.3 Fix tests  4.4 Negative tests  4.5 Metrics verify

Wave 5 (7 tasks, E2E verification, requires Docker):
  5.1-5.7 Full E2E tests

Wave 5b (7 tasks, ALL parallel, code quality review):
  5b.1-5b.7 Dead code, types, imports, naming, docs, coverage, security

Wave 6 (6 tasks, performance):
  6.1-6.6 Benchmarks + dynamic batching drift check
```

**Total: 42 tasks, 6 waves.** Wave 3 consolidates into 3 parallel tracks (engine, model_tools, config/infra).

---

## Documentation Updates (mandatory after each wave)

- **CLAUDE.md** — architecture, shipped processors, config schema, base classes, import contracts
- **README.md** (Russian) — usage, examples, metrics table
- **docs/architecture.md** — data flow, type hierarchy
- **docs/parallelism.md** — variant list, base class examples
- **docs/config-reference.md** — auto-generated from pydantic schemas (Task 3.13)
- **Agent/skill instructions** — update if conventions changed
- **pyproject.toml** — mypy exclude, vulture, import-linter contracts

---

## Metrics Naming Convention

Format: `{prefix}_{measure_name}` — only `_` separator allowed, no dots.

`MetricEmitter` builds prefix from **one** identifier — whichever is available:
- `app_name` from config (preferred) → `{APP_NAME}`
- `hostname` from `socket.gethostname()` (fallback if no app_name)
- Only one is used, never both concatenated

Prepended at flush time. Declaration sites (Counter, Gauge, GroupBy) use simple measure names with `_` only.

Examples:
```
steel_inspector_inference_latency_ms
steel_inspector_detection_done
steel_inspector_triton_gpu_utilization
```

### Triton metrics resend
TritonMetricsCollector scrapes `/metrics`, formats as `{prefix}_triton_{metric}` (prefix = app_name or hostname). Writes to Redis metrics stream. Verified in Task 5.5.

---

## Key Contracts

### Event base class + PipelineEvent[TCtx, TPayload]
- `Event(BaseModel)` — base for all user events, has single `fw: FrameworkMeta` field
- `FrameworkMeta` — frozen struct with `source_stream`, `message_id` (framework routing data)
- If user subclass redefines `fw` → **pydantic raises at class definition** (conflict at import)
- `PipelineEvent[TCtx: Event, TPayload: BaseModel]` — context carries `fw` from Event
- context: TCtx (user's typed domain data + fw struct, frozen)
- payload: TPayload (swapped by shipped processors)
- `fw` is **stripped from Redis output** — only user fields + `result` JSON appear in stream
- TCtx must not have field named `result` (reserved for Redis payload JSON key)

### Pipeline data flow
```
RedisStreamSource[SteelInput] → SteelInput | ConnectorError
UserBusinessProcessor → PipelineEvent[SteelMeta, EvenCameraPayload]
ClientPreprocessorNode[SteelMeta, EvenCameraPayload] → PipelineEvent[SteelMeta, TritonRequest]
InferenceNode[SteelMeta, YoloTritonRequest, YoloInferenceOutput] → PipelineEvent[...] | InferenceError
PostProcessorNode[SteelMeta, YoloInferenceOutput, DetectionResult] → PipelineEvent[SteelMeta, DetectionResult]
RedisStreamSink[SteelMeta, DetectionResult] → Redis (serializes + writes)
```

### Redis output format
Sink serializes internally. User context fields as top-level hash entries + `result` JSON field for payload. `fw` struct is **stripped** — not written to Redis. No framework data in output stream. No Envelope in the pipeline.

### Static analysis catches contract violations
The entire pipeline flow is typed so mypy/pyright catch errors at lint time, not runtime:
- Wrong payload type to preprocessor → mypy error on `PipelineEvent[Ctx, WrongPayload]` mismatch
- Missing generic args on Node class → pyright error on unparameterized generic
- Business processor returning wrong PipelineEvent payload → type checker catches union mismatch
- Passing DetectionResult sink where ClassificationResult expected → type error
- `Application.build()` accepts `list[object]` at runtime but each node's `__orig_bases__` carries full type info for registry validation
- expand_bundle with TYPE_CHECKING overloads ensures static analysis sees concrete types even for runtime-generated subclasses

### Framework never overrides user data
- `Event` base class owns `fw: FrameworkMeta` — pydantic prevents redefinition in subclasses
- Source sets `event.fw` from envelope metadata after deserialization
- PipelineEvent.context is frozen, shipped processors copy verbatim
- Sink reads `context.fw` for Redis routing, then **strips `fw` from output** — only user-defined fields appear as top-level Redis hash entries
- Round-trip negative tests verify all user fields preserved unchanged and `fw` absent from output

---

## Obsolete Files — Delete or Rewrite After Completion

No obsolete files may remain in the repo. This is a mandatory checklist after all waves.

### Files to DELETE entirely
| File | Reason |
|---|---|
| `providers/app_factory.py` | Replaced by Application.build() |
| `providers/node_assembly_provider.py` | Config-driven node assembly replaced by explicit processor list |
| `providers/engine_provider.py` | Wired Application construction — replaced by Application.build() |
| `engine/context.py` | EventContext (contextvars) replaced by Event.fw: FrameworkMeta |
| `engine/wrappers.py` | Metrics logic moved to Base* classes (Task 2.4). Wrappers obsolete. |
| `routers/default.py` | Dynamic field-based routing replaced by typed PipelineEvent DAG routing |
| `routers/multi_model.py` | BroadcastRouter replaced by typed per-model PipelineEvent routing |
| `routers/protocol.py` | RoutingTarget/RouterProtocol obsolete — DAG routes by type |
| `routers/__init__.py` | Entire routers/ directory deleted |
| `connectors/output_router.py` | Output routing done by typed sinks, not by router |

### DI providers — KEEP (dishka stays)
DI is NOT dropped. Only the magic pipeline wiring is removed. DI is still used for infrastructure:
| File | Status | Reason |
|---|---|---|
| `providers/config_provider.py` | **Keep** | Hydra config loading into DI container |
| `providers/connector_provider.py` | **Keep** | Redis client creation from config |
| `providers/inference_provider.py` | **Keep** | gRPC client setup from config |
| `providers/metrics_provider.py` | **Keep** | MetricBuffer/Emitter lifecycle |
| `providers/scopes.py` | **Keep** | Request/app DI scopes |
| `providers/preprocessing_provider.py` | **Keep but update** | LetterBoxPreprocessor import path changes |
| `providers/shipped_factories.py` | **Delete** | ConfigurableClientPreprocessor replaced by ClientPreprocessorNode |

### Files to EVALUATE — keep or delete
| File | Question |
|---|---|
| `engine/assembly.py` | **Keep for expand_bundle()** — `import_class()` and type resolution still needed by `expand_bundle()` for dynamic subclass creation. Delete `wrap_with_variant()`, `assemble_node()`, `assemble_all()`. |
| `engine/metric_aggregator.py` | **Keep + update.** `MetricAggregator`, `InstrumentedNode`, `extract_event_counters`, `_extract_and_record_metrics` (moved from wrappers) all stay. Base* classes integrate with them. |
| `engine/queue_bus.py` | **Keep** — still needed for inter-thread queue communication. |
| `engine/middleware.py` | **Keep** — RetryMiddleware, ConcurrencyLimitMiddleware are generic wrappers. Node classes can compose with them. No obsolete patterns. |
| `connectors/base.py` | **Keep** — InputConnectorProtocol/OutputConnectorProtocol still used by RedisStreamSource/Sink internally for Redis I/O. |
| `connectors/consumer_group.py` | **Keep** — Redis consumer group management. |
| `preprocessing/decoder.py` | **Keep but update** — image decoders. Used by ClientPreprocessorNode internally. |
| `preprocessing/image_decode_processor.py` | **Delete** — returns DecodedImage (deleted). Decoding logic moves into ClientPreprocessorNode which uses decoder.py directly. |
| `preprocessing/protocols.py` | **Keep** — decoder/preprocessor protocols used by Node classes. |
| `preprocessing/letterbox.py` | **Move** to triton_shared_models/detection/preprocessing.py (Task 1.3). Delete original. |
| `inference/model_registry.py` | **Keep** — model version tracking, used by InferenceNode. |
| `errors/collector.py` | **Keep** — error collection. |
| `errors/handler.py` | **Keep** — error handling. |
| `errors/models.py` | **Keep** — ConnectorError, InferenceError, PreprocessingError. |
| `metrics/aggregation.py` | **Keep** — time-windowed aggregation for GroupBy. |
| `providers/preprocessing_provider.py` | **Keep but update** — LetterBoxPreprocessor import path changes to shared_models. |

### Files to REWRITE (major changes, not just import updates)
| File | What changes |
|---|---|
| `main.py` | Rewrite: uses EngineProvider (deleted) + dishka container.get(Application). Must use Application.build() instead. Keep hydra.main entry point but remove DI container for Application resolution. DI still used for infrastructure (Redis client, metrics). |
| `protocols.py` | Delete Source, Processor, Sink protocols + BaseSource, BaseProcessor, BaseSink. Add `name` param to remaining Base* classes. |
| `engine/shipped_processors.py` | Delete EnvelopeExtractor, EventSerializer, DaliInferenceInputBuilder, ClientPreprocessor, LogErrorSink. Replace with Node classes. |
| `engine/registry.py` | Delete inline variants, _infer_variant_from_signature, _get_call_hints, _eval_annotations_with_bounds. Add _extract_types_from_generics, exact generic matching. |
| `engine/app.py` | Delete _dispatch_cycle, _route_event, _drain_inline_queues. Add Application.build(), runtime type-checked routing. |
| `engine/events.py` | Delete InferenceInput (renamed TritonRequest), dead InferenceOutput/TensorSpec, DecodedImage. Add PipelineEvent, ImagePayload, ClassificationResult. Rewrite DetectionResult. |
| `engine/assembly.py` | Delete _INLINE_VARIANTS, inline passthrough in wrap_with_variant. |
| `connectors/redis_stream.py` | RedisStreamSource → generic BaseThreadSource[T\|ConnectorError]. RedisStreamSink → generic BaseThreadSink[PipelineEvent[TCtx,TResult]]. |
| `config/schema.py` | Delete SOURCE/PROCESSOR/SINK from NodeVariant. Remove ALL default values. Delete RouterConfig + router_class (class path in config — violates convention, replaced by typed DAG routing). |
| `engine/diagram.py` | Remove inline node shapes. |

### Files to UPDATE (import paths, type changes)
| File | What changes |
|---|---|
| `__init__.py` (connector pkg) | Remove exports: EnvelopeExtractor, EventSerializer, build_application, Source, Processor, Sink, BaseSource, BaseProcessor, BaseSink. Add new exports. |
| `providers/__init__.py` | Remove build_application export. |
| `providers/shipped_factories.py` | Update metadata handling → PipelineEvent pattern. |
| `providers/preprocessing_provider.py` | Update LetterBoxPreprocessor import → triton_shared_models.detection.preprocessing. |
| `inference/request.py` | Update InferenceInput → TritonRequest re-export. |
| `inference/worker.py` | InferenceInput → TritonRequest in __call__ signature. |
| `triton_model_tools/inference/yolo_bridge.py` | detections_json → tuple[Detection, ...]. Import Detection from shared_models. |
| `triton_model_tools/inference/preprocessing.py` | LetterBoxPreprocessor import from shared_models. |
| `triton_model_tools/models/detection.py` | Re-import Detection from shared_models. |
| `triton_model_tools/tests/test_dali.py` | LetterBoxPreprocessor import path. |
| `triton_model_tools/tests/conftest.py` | Delete scattered coco8 candidate paths → use root test_fixtures/. |

### Docs to REWRITE
| File | What changes |
|---|---|
| `docs/architecture.md` | Rewrite execution variant section, shipped processors table, data flow diagram, config schema. |
| `docs/parallelism.md` | Rewrite variant decision tree, base class examples, remove inline variant references. |
| `CLAUDE.md` (root) | Update architecture overview, shipped processors, base classes, config schema, import contracts. |
| `CLAUDE.md` (connector pkg) | Same as root — architecture, shipped processors, config, base classes. |
| `README.md` (connector pkg, Russian) | Update usage examples, architecture overview, metrics table. |
| `README.md` (model_tools, Russian) | Update if CLI commands changed. |

### Agent/Skill instructions to UPDATE
| File | What changes |
|---|---|
| `.claude/agents/architect.md` | Remove execution variant model (9 variants), update to 6 variants. Remove build_application factory docs. Add PipelineEvent, Application.build(), ModelBundle. |
| `.claude/agents/implementer.md` | Remove variant declaration examples, wrapper examples, build_application example. Add Node class examples, PipelineEvent pattern. |
| `.claude/agents/reviewer.md` | Update review checklist for new patterns. |
| `.claude/agents/tester.md` | Update test patterns for PipelineEvent, typed sinks, negative tests. |
| `.claude/skills/commit/SKILL.md` | **Update** — pre-commit hooks unchanged but verify commit workflow still matches. |
| `.claude/skills/check-infra/SKILL.md` | Update if Docker services changed. |
| `.claude/skills/scaffold-module/SKILL.md` | Update if package structure changed (new submodules in shared_models). |
| `.claude/skills/using-nvidia-dali/SKILL.md` | **No change** — external tool reference, no connector API refs. |
| `.claude/skills/using-triton-inference-server/SKILL.md` | **No change** — external tool reference. |
| `.claude/skills/using-ultralytics/SKILL.md` | **No change** — external tool reference. |

### Example files to REWRITE
| File | What changes |
|---|---|
| `examples/dual_model_pipeline/run_pipeline.py` | Replace build_application() → Application.build() with explicit processor list. |
| `examples/dual_model_pipeline/processors.py` | Replace metadata dict → PipelineEvent[SteelMeta, ...]. Delete ModelResultRouter (typed sinks handle routing). Delete EvenModelSink/OddModelSink (replaced by RedisStreamSink[TCtx, TResult]). |
| `examples/dual_model_pipeline/events.py` | Replace metadata: dict → SteelMeta context. Delete EvenModelResult/OddModelResult. Delete detections_json. |
| `examples/dual_model_pipeline/config.yaml` | Remove defaults, make all fields required. |
| `examples/dual_model_pipeline/test_deterministic.py` | Update for new event types. |

### Config files to UPDATE
| File | What changes |
|---|---|
| `pyproject.toml` (root) | Update import-linter contracts if layer boundaries changed. |
| `pyproject.toml` (connector) | Update mypy exclude if files moved. |
| `pyrightconfig.json` | Update exclude paths if files moved. |
| Root `conftest.py` | Add test_fixtures path fixtures. Remove old coco8 candidate logic. |

### Verify: zero references to deleted patterns
After all waves, run each grep — **must return 0 results**:
```bash
uv run ruff check .                                          # no import errors
uv run vulture .                                             # no dead code

# Deleted classes/functions
grep -r "build_application" packages/                        # zero
grep -r "EnvelopeExtractor" packages/                        # zero
grep -r "EventSerializer" packages/                          # zero
grep -r "EventContext\|current_context\|set_context\|reset_context" packages/  # zero
grep -r "DaliInferenceInputBuilder\b" packages/              # zero (replaced by DaliPreprocessorNode)
grep -r "ClientPreprocessor\b" packages/triton_redis_connector/  # zero (replaced by ClientPreprocessorNode)
grep -r "LogErrorSink\b" packages/triton_redis_connector/    # zero (replaced by LogErrorSinkNode)
grep -r "ConfigurableClientPreprocessor" packages/           # zero

# Deleted protocols/base classes
grep -r "class Source\b\|class Processor\b\|class Sink\b" packages/triton_redis_connector/triton_redis_connector/protocols.py  # zero
grep -r "\bBaseSource\b\|\bBaseProcessor\b\|\bBaseSink\b" packages/triton_redis_connector/  # zero

# Deleted patterns
grep -r "detections_json" packages/                          # zero
grep -r "metadata: dict" packages/triton_redis_connector/    # zero in connector pkg
grep -r "InferenceInput\b" packages/                         # zero (renamed to TritonRequest)
grep -r "__call_hints__" packages/                           # zero
grep -r "_infer_variant_from_signature" packages/            # zero
grep -r "get_dispatch_order" packages/                       # zero
grep -r "INLINE_VARIANTS" packages/                          # zero
grep -r "RoutingTarget\|RouterProtocol\|BroadcastRouter\|FieldBasedRouter" packages/  # zero
grep -r "router_class" packages/                             # zero (was class path in config)

# Deleted files must not exist
test ! -f packages/triton_redis_connector/triton_redis_connector/providers/app_factory.py
test ! -f packages/triton_redis_connector/triton_redis_connector/providers/node_assembly_provider.py
test ! -f packages/triton_redis_connector/triton_redis_connector/providers/engine_provider.py
test ! -f packages/triton_redis_connector/triton_redis_connector/engine/context.py
test ! -f packages/triton_redis_connector/triton_redis_connector/engine/wrappers.py
test ! -f packages/triton_redis_connector/triton_redis_connector/routers/default.py
test ! -f packages/triton_redis_connector/triton_redis_connector/routers/multi_model.py
test ! -f packages/triton_redis_connector/triton_redis_connector/routers/protocol.py
test ! -f packages/triton_redis_connector/triton_redis_connector/connectors/output_router.py
test ! -f packages/triton_redis_connector/triton_redis_connector/providers/shipped_factories.py
test ! -f packages/triton_redis_connector/triton_redis_connector/preprocessing/image_decode_processor.py
```

---

## Post-Completion Cleanup

After all waves complete successfully:
- **Keep** `docs/architecture-rework-plan.md` — architectural decision record for the team
- **Keep** `docs/e2e-verification-guide.md` — user-facing guide for verifying the system
- Run the full obsolete pattern grep (above) one final time

---

## Verification

1. `uv run ruff check . && uv run ruff format .`
2. `uv run mypy .`
3. `uv run pyright`
4. `uv run lint-imports`
5. `uv run vulture` — must find zero dead code
6. `uv run pytest`
7. Obsolete pattern grep (above) — must return zero results
8. Wave 5 E2E tests (Docker required)
9. Wave 6 benchmarks

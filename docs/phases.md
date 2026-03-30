# Implementation Phases

This document splits the [design](design.md) into 5 phases. Each phase produces a
testable system that builds on the previous one. Phases are logical groupings — PRs
can span or split them as needed.

---

## Phase 1 — Foundation: Shared Models, Config, Envelope, Protocols

**Goal**: Extract shared types, set up Hydra config, define all Protocol interfaces,
and build the Envelope event wrapper. After this phase the project has a compilable
skeleton with no runtime behavior but all public API contracts locked in.

### Deliverables

#### 1.1 Package: `triton_shared_models`

Create `packages/triton_shared_models/` with types extracted from `triton_model_tools`:

| Module | Contents |
|---|---|
| `triton_types.py` | `TritonDataType`, `TritonInstanceKind`, `TritonTensorSpec` |
| `triton_config.py` | `TritonModelConfig`, `to_pbtxt()` |

Update `triton_model_tools` to import from `triton_shared_models` instead of
defining its own copies. Verify existing `triton_model_tools` tests still pass.

#### 1.2 Config schema (`config/schema.py`)

Pydantic models: `RedisConnectionConfig`, `StreamInputConfig`, `StreamOutputConfig`,
`InferenceConfig`, `PreprocessingConfig`, `MetricsConfig`, `AppConfig`.

#### 1.3 Hydra integration (`config/hydra_app.py`)

Hydra initialization, default YAML files under `conf/`. Environment variable
interpolation via OmegaConf (`${oc.env:REDIS_HOST}`).

Add `hydra-core` dependency to `triton_redis_connector/pyproject.toml`.

#### 1.4 Envelope (`engine/event.py`)

`Envelope[T]` dataclass with `track_state()`, `transit_time_ms()`, `__lt__` for
PriorityQueue ordering. `message_id_to_priority()` helper.

#### 1.5 All Protocol classes

| File | Protocols |
|---|---|
| `protocols.py` | `ComponentProtocol`, `InferenceRequestBuilderProtocol`, `ResultHandlerProtocol`, `ObserverProtocol`, `AggregationRuleProtocol` |
| `connectors/base.py` | `InputConnectorProtocol`, `OutputConnectorProtocol` |
| `preprocessing/protocols.py` | `ImageDecoderProtocol`, `ImagePreprocessorProtocol` |

#### 1.6 Error models (`errors/models.py`)

`ConnectorError`, `InferenceError`, `PreprocessingError`.

#### 1.7 Inference request models (`inference/request.py`)

`InferenceRequest`, `InferenceResult`.

#### 1.8 Metric event model (`metrics/collector.py`)

`MetricEvent` pydantic model.

### Tests

- Unit tests for all pydantic models (config, errors, metric event, inference request)
- Unit tests for Envelope (state tracking, priority calculation, ordering)
- Verify `triton_model_tools` tests pass after shared model extraction
- Hydra config loading tests (YAML → pydantic validation)
- Protocol classes are `runtime_checkable` — test isinstance checks on dummy impls

### Verification

```bash
uv run pytest
uv run pre-commit run --all-files
```

All existing tests pass + new unit tests for foundation types.

---

## Phase 2 — Connectors, Queues, Engine Loop

**Goal**: Read from Redis Streams, route events through queues, dispatch to
components via type-annotation wiring, and write results to output streams.
After this phase a minimal end-to-end flow works: Redis in → component → Redis out
(no inference yet, just passthrough).

### Deliverables

#### 2.1 Consumer group management (`connectors/consumer_group.py`)

`ConsumerGroupManager`: `ensure_group_exists()`, `read_pending_newest_first()`,
`read_new()`, `acknowledge()`.

#### 2.2 Redis input connector (`connectors/redis_stream.py`)

`RedisStreamInputConnector`: runs in dedicated thread, blocking XREADGROUP,
decodes to pydantic models, wraps in Envelope, puts on output queue. Error
resilience — catches `ConnectionError`, `TimeoutError`, `ValidationError`, routes
to error queue.

#### 2.3 Redis output connector (`connectors/redis_stream.py`)

`RedisStreamOutputConnector`: runs in dedicated thread, reads from input queue,
XADD to stream, triggers ACK callback.

#### 2.4 Queue bus (`engine/queue_bus.py`)

`QueueBus`: manages `live_queue`, `pending_queue`, `output_queue`, `error_queue`
per stream. Combined reader logic: live-first, then pending (newest-first), block
on live when both empty.

#### 2.5 Component registry (`engine/registry.py`)

`ComponentRegistry`: `register()` inspects `process()` method type hints.
`build_graph()` wires components by matching input/output types. Raises on
missing sources, cycles, dangling outputs.

#### 2.6 Application skeleton (`engine/app.py`)

`Application.from_config()` and `run()` — startup sequence through step 8
(connector threads running). Main loop: drain queues → dispatch to components →
route results to output queue. Graceful shutdown on SIGINT/SIGTERM.

#### 2.7 Recovery with newest-first (`connectors/consumer_group.py`)

On startup: read pending messages, sort newest-first, feed into `pending_queue`.
Dual-queue reader in main loop.

### Tests

- `ConsumerGroupManager`: mock Redis client, test XGROUP CREATE idempotency,
  pending read ordering, XACK
- `RedisStreamInputConnector`: mock Redis, test pydantic decode, error routing,
  thread start/stop
- `RedisStreamOutputConnector`: mock Redis, test XADD, ACK callback
- `ComponentRegistry`: test type-hint extraction, graph wiring, error on bad graphs
- `Application`: integration test with mock Redis — send event in, verify event out
  through a passthrough component
- Recovery: test pending messages processed newest-first, live messages take priority

### Verification

```bash
uv run pytest
uv run pre-commit run --all-files
```

Integration test: start Application with mock Redis, push events to input stream,
verify they appear on output stream in correct order.

---

## Phase 3 — Inference: Worker Pool, Retry, Rolling Release

**Goal**: Add the multiprocessing inference worker pool. After this phase the
full pipeline works: Redis in → preprocess → Triton inference → Redis out.

### Deliverables

#### 3.1 Inference worker pool (`inference/worker.py`)

`InferenceWorkerPool`: spawns N worker processes. Each worker creates its own
`tritonclient.grpc.InferenceServerClient`, runs `async_infer` with callbacks,
applies retry policy. Communication via `multiprocessing.Queue`.

`_inference_worker_main()`: worker process entry point. Reads from request queue,
uses semaphore for concurrency control, puts results/errors on result queue.

#### 3.2 Retry policy (`inference/retry.py`)

`RetryPolicy`: 3 attempts, exponential backoff (0.1s, 0.4s, 1.6s).
`execute_with_callback()` wraps async_infer calls inside worker processes.

#### 3.3 Model version tracker (`inference/model_registry.py`)

`ModelVersionTracker`: per-worker instance. Polls `get_model_repository_index()`,
checks `is_model_ready()`. `tracked_inference()` context manager for safe in-flight
counting. Runs as timer thread within each worker process.

#### 3.4 Image preprocessing (`preprocessing/`)

- `TurboJPEGDecoder` (`jpeg_decoder.py`) — process-safe via per-worker init
- `CV2AvifDecoder` (`avif_decoder.py`)
- `DecoderDispatcher` (`dispatcher.py`) — magic byte detection, content type routing
- `ParallelDecoderPool` (`pool.py`) — `ProcessPoolExecutor`, `decode()`, `decode_batch()`

Add `PyTurboJPEG`, `opencv-python-headless` dependencies.

#### 3.5 Wire inference into Application

Update `Application.run()`:
- Start `InferenceWorkerPool` (step 9 in bootstrap)
- Start `ParallelDecoderPool`
- Main loop: when component returns `InferenceRequest`, put on `mp.request_queue`
- Drain `mp.result_queue`, route `InferenceResult` to result handler / output queue
- Route `InferenceError` to error handler
- Shutdown: poison pills → join workers

### Tests

- `RetryPolicy`: test 1/2/3 attempt scenarios, backoff delays, final failure
- `ModelVersionTracker`: mock Triton client, test version switch, in-flight tracking,
  context manager
- `DecoderDispatcher`: test magic byte detection, routing to correct decoder
- `ParallelDecoderPool`: test with real JPEG/PNG bytes, verify output shape/dtype
- `InferenceWorkerPool`: integration test with mock Triton (or identity model from
  `model_repository/identity/`) — submit request, verify result comes back
- End-to-end: Redis → decode → Triton identity model → Redis out

### Verification

```bash
uv run pytest
uv run pre-commit run --all-files
```

End-to-end test with devcontainer services (Redis + Triton with identity model):
push image event to input stream, verify inference result on output stream.

---

## Phase 4 — Metrics, Errors, Observability

**Goal**: Add the complete metrics pipeline and error handling system. After this
phase the application emits operational metrics to a Redis stream and handles all
error scenarios gracefully.

### Deliverables

#### 4.1 Metric buffer and emitter (`metrics/collector.py`, `metrics/emitter.py`)

`MetricBuffer`: thread-safe, `push()` / `drain()`.
`MetricEmitter`: periodic timer thread, drains buffer, XADD to metrics stream.

#### 4.2 Transit time tracking

Wire `Envelope.track_state()` calls at each pipeline stage (received, decoded,
queued, processing_start, inference_submitted, inference_complete, output_sent,
acknowledged). Auto-emit `event_total_transit_ms`, `event_queue_wait_ms`,
`event_inference_ms`.

#### 4.3 Triton metrics collector (`metrics/triton_metrics.py`)

`TritonMetricsCollector`: periodic timer thread, calls
`get_inference_statistics()` via gRPC, converts to `MetricEvent` instances.
Metrics: inference count, queue time, compute times per model.

#### 4.4 Error collector (`metrics/error_collector.py`)

`ErrorCollector`: buckets errors by type, flushes as `MetricEvent` on interval.
`record()` / `flush()` methods.

#### 4.5 Error handler (`errors/handler.py`)

`ErrorHandler`: drains error queue in main thread, logs immediately via
`logger.error()`, feeds `ErrorCollector.record()` for aggregation.

#### 4.6 Wire into Application

Update `Application.run()`:
- Start `MetricEmitter` timer
- Start `TritonMetricsCollector` timer
- Start `ErrorCollector` flush timer
- Main loop: drain error queue → ErrorHandler

### Tests

- `MetricBuffer`: concurrent push from multiple threads, drain correctness
- `MetricEmitter`: mock Redis, verify XADD format for MetricEvent
- `TritonMetricsCollector`: mock Triton stats response, verify MetricEvent output
- `ErrorCollector`: record errors, flush, verify counts and metric names
- `ErrorHandler`: verify logging + collection routing
- Transit time: verify all states tracked, metrics emitted at correct stages
- Integration: run pipeline, verify metrics appear on metrics stream

### Verification

```bash
uv run pytest
uv run pre-commit run --all-files
```

Integration test: run full pipeline, verify metrics stream contains transit times,
Triton stats, and error counts.

---

## Phase 5 — Aggregation Library, Polish, Documentation

**Goal**: Add the business metrics aggregation package, finalize the user-facing
API, and ensure production readiness.

### Deliverables

#### 5.1 Package: `triton_metrics_aggregator`

Create `packages/triton_metrics_aggregator/`:

- `AggregationRule` pydantic model
- `TimeWindowAggregator[TEvent]`: validates field paths against pydantic model_fields
  at registration time, `observe()` records events, `flush()` returns aggregated
  `MetricEvent` instances
- Built-in rules: `count_by`, `sum_by`, `avg_by`

#### 5.2 Aggregator integration as ObserverProtocol

Register `TimeWindowAggregator` as `ObserverProtocol` component. Framework feeds
events without expecting output. Flushed metrics go to shared `MetricBuffer`.

#### 5.3 User-facing API cleanup

- Public exports in `__init__.py` for all packages
- Verify all Protocol classes are importable from top-level
- Entry point: `__main__.py` with Hydra `@hydra.main`

#### 5.4 Example application

Create `examples/` directory with a minimal camera inference example:
- `CameraImageEvent` model
- `CameraToInference` component
- `DetectionResultHandler` component
- Hydra config YAML
- Camera-per-minute aggregation rule

#### 5.5 README update

Update `README.md` with:
- Architecture overview
- Quick start with example
- Configuration reference
- Extension guide (implementing Protocols)

### Tests

- `AggregationRule`: validation of field paths, window config
- `TimeWindowAggregator`: observe events, flush at window boundary, verify
  per-key counts, multi-rule support
- Integration: full pipeline with aggregator → verify business metrics on stream
- Example app: smoke test that it starts and processes one event

### Verification

```bash
uv run pytest
uv run pre-commit run --all-files
```

Full end-to-end: example app running in devcontainer with Redis + Triton,
processing sample images, emitting inference results, operational metrics,
and business aggregation metrics.

---

## Phase Summary

| Phase | Scope | Key Output |
|---|---|---|
| **1** | Shared models, config, protocols, envelope | Compilable skeleton, all API contracts |
| **2** | Connectors, queues, engine, recovery | Redis in → component → Redis out (no inference) |
| **3** | Inference workers, preprocessing, retry, rolling release | Full pipeline: Redis → Triton → Redis |
| **4** | Metrics, error handling, observability | Operational metrics, error resilience |
| **5** | Aggregation library, examples, docs | Business metrics, production-ready |

### Dependency Graph

```
Phase 1 ──► Phase 2 ──► Phase 3
                │              │
                └──► Phase 4 ◄─┘
                         │
                         ▼
                     Phase 5
```

Phase 4 can start after Phase 2 (metrics don't need inference to work), but
inference metrics (Phase 4.3) need Phase 3. In practice, implement sequentially
1 → 2 → 3 → 4 → 5 for simplest dependency management.

# Triton Redis Connector — Architectural Design

## 1. Overview

### Problem Statement

ML inference pipelines need a reliable bridge between data sources (camera streams,
file uploads) and NVIDIA Triton Inference Server. The system must handle:

- Reading events from Redis Streams with consumer group semantics
- Decoding and preprocessing images for YOLO models
- Sending async inference requests to Triton via gRPC
- Writing results back to Redis Streams
- Collecting metrics (transit times, Triton stats, errors, business aggregations)
- Recovering after restarts with newest-first priority and TTL-based staleness
- Safe rolling model version updates

### System Context

```
Camera/Sensors → Redis Streams → [triton-redis-connector] → Triton Server
                                        ↓                        ↓
                                  Redis Streams ←── results ──────┘
                                  (results, metrics)
```

The connector is a multi-process, multi-threaded Python application.

- **Main process**: orchestrates sources, processors, sinks, queues, dispatch
- **Inference worker pool**: N separate processes, each with its own Triton gRPC
  client. Requests distributed via `multiprocessing.Queue`. Each worker uses
  `async_infer` with callback-based retry internally for concurrency.
- **Decoder pool**: configurable `ThreadPoolExecutor` or `ProcessPoolExecutor`
  for CPU-bound image decoding (single decoder per config).
- **Source threads**: blocking I/O (e.g. Redis XREADGROUP) in dedicated threads
  within the main process.

---

## 2. Package Structure

The monorepo contains four packages:

```
packages/
  triton_shared_models/              # Shared Triton types (extracted from model_tools)
    pyproject.toml
    triton_shared_models/
      __init__.py
      py.typed
      triton_types.py               # TritonDataType, TritonInstanceKind, TritonTensorSpec
      triton_config.py              # TritonModelConfig, to_pbtxt()

  triton_redis_connector/            # Core connector framework
    pyproject.toml
    triton_redis_connector/
      __init__.py
      py.typed
      protocols.py                  # Source, Processor, Sink protocols

      config/
        __init__.py
        schema.py                   # Pydantic models for Hydra-loaded config
        hydra_app.py                # Hydra initialization, config composition

      connectors/
        __init__.py
        redis_stream.py             # RedisStreamSource, RedisStreamSink
        consumer_group.py           # Consumer group management, position tracking

      engine/
        __init__.py
        app.py                      # Application — main entry point
        registry.py                 # Pipeline registry, type-annotation wiring
        queue_bus.py                # Thread-safe queues connecting stages
        event.py                    # Envelope[T] — timestamped event wrapper
        context.py                  # EventContext, current_context() via contextvars

      inference/
        __init__.py
        worker.py                   # InferenceWorkerPool, _inference_worker_main
        request.py                  # InferenceInput / InferenceOutput models
        model_registry.py           # ModelVersionTracker — rolling release (per-worker)

      preprocessing/
        __init__.py
        protocols.py                # ImageDecoder Protocol, ImagePreprocessor Protocol
        decoder.py                  # OpenCVDecoder, TurboJPEGDecoder
        pool.py                     # DecoderPool — thread or process pool wrapper

      metrics/
        __init__.py
        collector.py                # MetricEvent model, MetricBuffer
        triton_metrics.py           # TritonMetricsCollector — polls Triton stats
        error_collector.py          # ErrorCollector — interval-based error aggregation
        emitter.py                  # MetricEmitter — writes to output stream

      errors/
        __init__.py
        models.py                   # ConnectorError, InferenceError, PreprocessingError
        handler.py                  # ErrorHandler — routes errors from error queue

    conf/                            # Hydra config directory
      config.yaml
      sources/
        camera_images.yaml
      sinks/
        detection_results.yaml
        metrics.yaml
      inference/
        default.yaml
      preprocessing/
        default.yaml
      metrics/
        default.yaml

    tests/
      conftest.py
      test_config/
      test_connectors/
      test_engine/
      test_inference/
      test_preprocessing/
      test_metrics/

  triton_metrics_aggregator/         # Aggregation metrics library
    pyproject.toml
    triton_metrics_aggregator/
      __init__.py
      py.typed
      aggregator.py                 # TimeWindowAggregator
      rules.py                      # Built-in rules (count_by, sum_by, avg_by)
    tests/

  triton_model_tools/                # Existing — model export tools (unchanged logic)
    ...                              # Now depends on triton_shared_models for types
```

### Package Dependencies

```
triton_shared_models        — pydantic, loguru (no external deps)
triton_redis_connector      — triton_shared_models, redis, tritonclient[all],
                              pydantic, loguru, numpy, hydra-core,
                              PyTurboJPEG, opencv-python-headless
triton_model_tools          — triton_shared_models, ultralytics, onnx,
                              onnxruntime, tritonclient[all], pydantic, loguru,
                              numpy, Pillow
triton_metrics_aggregator   — triton_shared_models, triton_redis_connector,
                              pydantic, loguru
```

---

## 3. Layer Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                        CONFIGURATION LAYER                           │
│  Hydra YAML files + env var interpolation → pydantic AppConfig       │
└──────────────────────────┬───────────────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────────────┐
│  MAIN PROCESS                                                        │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │                    APPLICATION LAYER (engine/)                  │  │
│  │  Application       PipelineRegistry       QueueBus             │  │
│  │  (orchestrator)    (type-hint wiring)     (thread-safe queues) │  │
│  └──┬────────────────────┬────────────────────────┬───────────────┘  │
│     │                    │                        │                  │
│  ┌──▼────────────────────▼────────────────────────▼───────────────┐  │
│  │                    SOURCE / SINK LAYER                          │  │
│  │  ┌─────────────────┐        ┌──────────────────┐              │  │
│  │  │ Source[T]        │ Thread  │ Sink[T]           │ Thread      │  │
│  │  │ (Redis XREAD)    │───────► │ (Redis XADD)      │              │  │
│  │  └─────────────────┘ Queue  └──────────────────┘              │  │
│  │  Protocol-based: Kafka can be added later.                     │  │
│  └────────────────────────────────────────────────────────────────┘  │
│     │                                       ▲                        │
│     │ Queue[Envelope[T]]                    │ Queue[T]               │
│     ▼                                       │                        │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │                    PROCESSING LAYER (user processors)           │  │
│  │  Processor[TIn, TOut].__call__(event) wired by type annotations│  │
│  │  Envelope hidden: runtime unwraps payload, wraps result        │  │
│  │  Metadata access via current_context() contextvars             │  │
│  └──────────────────────────┬─────────────────────────────────────┘  │
│                              │ InferenceInput (type-based intercept) │
│                              ▼ mp.request_queue                      │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │                    METRICS LAYER (metrics/)                     │  │
│  │  MetricBuffer → MetricEmitter → Redis Stream "metrics"         │  │
│  │  TritonMetricsCollector | ErrorCollector | TimeWindowAggregator │  │
│  └────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────┬───────────────────────────────────────┘
                               │ mp.request_queue
    ┌──────────────────────────┼───────────────────────┐
    ▼                          ▼                       ▼
┌──────────────────────────────────────────────────────────────────────┐
│  INFERENCE WORKER PROCESSES (N separate OS processes)                 │
│                                                                      │
│  Each worker has:                                                    │
│  - Own gRPC InferenceServerClient (independent connection)           │
│  - async_infer with callback-based retry (threading.Timer)           │
│  - ModelVersionTracker (per-worker, polls independently)             │
│  - Semaphore limiting concurrent in-flight requests                  │
│                                                                      │
│  Results → mp.result_queue → Main process                            │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│  DECODER POOL (ThreadPoolExecutor or ProcessPoolExecutor, via config)│
│                                                                      │
│  Single decoder type per config (opencv or turbojpeg)               │
│  Both release GIL during decode — ThreadPool is default             │
│  Results (numpy arrays) returned to main process                     │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 4. Data Flow

### 4.1 Happy Path: Image Event through Inference

```
Step 1: Redis Stream "camera_images"
        XREADGROUP GROUP app_group consumer_1 COUNT 10 BLOCK 1000

Step 2: RedisStreamSource (Thread-1)
        - Decodes Redis fields into pydantic model (CameraImageEvent)
        - Wraps in Envelope[CameraImageEvent] with receive timestamp
        - Calls emit(event) to push onto queue
        - Does NOT XACK yet (tracked in pending set)

Step 3: Runtime unwraps Envelope, calls Processor.__call__(event)
        - Registry resolved: CameraImageEvent → UserPreprocessor
        - current_context() provides metadata if needed

Step 4: UserPreprocessor (user code, implements Processor[TIn, TOut])
        - Extracts image bytes from event
        - Submits to DecoderPool (single decoder type per config)
        - Returns InferenceInput(raw_image=ndarray, model_name="yolo26s")

Step 5: Framework intercepts InferenceInput return type
        - Pipeline is split at the InferenceInput boundary
        - Puts InferenceInput onto mp.request_queue (non-blocking)

Step 6: Inference worker process (one of N) picks up request
        - ModelVersionTracker resolves latest ready version (per-worker)
        - Builds tritonclient InferInput from ndarray
        - Calls async_infer() with callback
        - On failure: callback schedules retry via threading.Timer
          with exponential delay, re-calls async_infer with new callback
        - Semaphore released only on final success or final failure
        - On success: puts InferenceOutput onto mp.result_queue

Step 7: Main process drains mp.result_queue
        - Framework wraps InferenceOutput in Envelope
        - Feeds to next Processor in pipeline (post-inference stage)

Step 8: Post-inference Processor returns result
        - If output type matches a Sink, framework routes to Sink

Step 9: RedisStreamSink (Thread-2)
        - Gets result from output queue
        - Serializes to Redis fields
        - XADD to output stream "detection_results"

Step 10: Acknowledgment
         - After successful output write, XACK on input stream
         - Envelope.track_state("acknowledged") records final timestamp
         - Transit time metric emitted
```

### 4.2 Newest-First Recovery with Interleave Ratio + TTL

```
1. Startup: XGROUP CREATE stream group $ MKSTREAM (no-op if exists)

2. Read pending (unacked) messages:
   XREADGROUP GROUP group consumer STREAMS stream 0
   Sort by message ID DESCENDING (newest first)
   Feed into pending_queue (PriorityQueue, newest = highest priority)
   Drop any pending message older than pending_ttl_s (default 60s):
     XACK without processing, log as dropped

3. Start live reading: XREADGROUP ... STREAMS stream >
   Feed into live_queue (regular Queue)

4. Combined reader with interleave ratio (default 4:1):
   - Process 4 live messages, then 1 pending message
   - If live_queue empty, process pending
   - If pending_queue empty, process live only
   - Both configurable via recovery_interleave_ratio and pending_ttl_s
   → Live data is never starved by recovery backlog
```

---

## 5. Configuration

### 5.1 Hydra Directory Layout

```
conf/
  config.yaml                     # Main config with defaults list
  sources/
    camera_images.yaml            # Source definition
  sinks/
    detection_results.yaml        # Sink definition
    metrics.yaml                  # Metrics output sink
  inference/
    default.yaml                  # Triton connection + retry config
  preprocessing/
    default.yaml                  # Decoder type + pool config
  metrics/
    default.yaml                  # Collection intervals
```

### 5.2 Pydantic Config Schema (`config/schema.py`)

```python
class RedisConnectionConfig(BaseModel):
    host: str = "localhost"           # env: REDIS_HOST
    port: int = 6379                  # env: REDIS_PORT
    db: int = 0
    decode_responses: bool = False


class SourceConfig(BaseModel):
    stream_name: str
    consumer_group: str
    consumer_name: str = ""           # auto-generated if empty
    event_model: str                  # fully-qualified pydantic model class path
    batch_size: int = 10
    block_ms: int = 1000
    newest_first_on_recovery: bool = True
    recovery_interleave_ratio: int = 4  # N live : 1 pending
    pending_ttl_s: float = 60.0         # drop pending older than this


class SinkConfig(BaseModel):
    stream_name: str
    maxlen: int | None = 10_000       # XADD MAXLEN ~ value
    approximate_maxlen: bool = True


class DecoderType(str, Enum):
    OPENCV = "opencv"
    TURBOJPEG = "turbojpeg"


class DecoderPoolType(str, Enum):
    THREAD = "thread"
    PROCESS = "process"


class InferenceConfig(BaseModel):
    grpc_url: str = "localhost:8001"  # env: TRITON_GRPC_URL
    num_inference_workers: int = 2
    max_concurrent_per_worker: int = 8
    max_retry_attempts: int = 3
    retry_base_delay_s: float = 0.1
    retry_backoff_multiplier: float = 4.0
    model_version_poll_interval_s: float = 5.0


class PreprocessingConfig(BaseModel):
    decoder: DecoderType = DecoderType.OPENCV
    decoder_pool_type: DecoderPoolType = DecoderPoolType.THREAD
    decoder_pool_workers: int = 4
    default_target_size: tuple[int, int] = (640, 640)


class QueueConfig(BaseModel):
    maxsize: int = 1000               # applied to ALL queues


class MetricsConfig(BaseModel):
    output_stream: str = "metrics"
    triton_poll_interval_s: float = 10.0
    error_flush_interval_s: float = 30.0
    transit_time_enabled: bool = True


class AppConfig(BaseModel):
    app_name: str = "triton-connector"  # env: APP_NAME
    redis: RedisConnectionConfig = RedisConnectionConfig()
    sources: list[SourceConfig] = []
    sinks: list[SinkConfig] = []
    processors: list[str] = []          # fully-qualified class paths
    inference: InferenceConfig = InferenceConfig()
    preprocessing: PreprocessingConfig = PreprocessingConfig()
    queues: QueueConfig = QueueConfig()
    metrics: MetricsConfig = MetricsConfig()
```

### 5.3 Example Main Config (`conf/config.yaml`)

```yaml
defaults:
  - inference: default
  - preprocessing: default
  - metrics: default
  - _self_

app_name: ${oc.env:APP_NAME,triton-connector}

redis:
  host: ${oc.env:REDIS_HOST,localhost}
  port: ${oc.env:REDIS_PORT,6379}

sources:
  - stream_name: camera_images
    consumer_group: inference_group
    event_model: myapp.models.CameraImageEvent
    batch_size: 10
    block_ms: 1000
    newest_first_on_recovery: true
    recovery_interleave_ratio: 4
    pending_ttl_s: 60.0

sinks:
  - stream_name: detection_results
    maxlen: 10000
  - stream_name: metrics
    maxlen: 50000

processors:
  - myapp.processors.CameraToInference
  - myapp.processors.DetectionResultHandler
```

---

## 6. Protocols & Extension Points

All extension points use `typing.Protocol` with `@runtime_checkable`. Users implement
these interfaces without inheriting from framework classes (structural subtyping).

### 6.1 Core Pipeline Protocols (`protocols.py`)

Three primitives define the user-facing pipeline:

```python
@runtime_checkable
class Source(Protocol[TOut]):
    """Produces events. Runs in own thread, pushes events via emit().

    Framework creates the stop Event, assigns it before starting the
    source thread, and calls source.stop.set() at shutdown.
    """

    stop: threading.Event

    def __call__(self, emit: Callable[[TOut], None]) -> None: ...


@runtime_checkable
class Processor(Protocol[TIn, TOut]):
    """Transforms events. Return None to filter (skip this event).

    Runtime unwraps Envelope.payload before calling __call__,
    wraps result in new Envelope after return.
    Users access metadata via current_context() contextvars.
    """

    def __call__(self, event: TIn) -> TOut | None: ...


@runtime_checkable
class Sink(Protocol[TIn]):
    """Consumes events (terminal pipeline node)."""

    def __call__(self, event: TIn) -> None: ...
```

Envelope is hidden from users. The runtime:
1. Unwraps `envelope.payload` before calling `Processor.__call__`
2. Wraps the return value in a new Envelope inheriting metadata
3. Sets `current_context()` via `contextvars` for opt-in metadata access

### 6.2 Image Protocols (`preprocessing/protocols.py`)

```python
@runtime_checkable
class ImageDecoderProtocol(Protocol):
    """Protocol for image decoders. Must be thread-safe."""

    def decode(self, raw_bytes: bytes) -> npt.NDArray[np.uint8]: ...


@runtime_checkable
class ImagePreprocessorProtocol(Protocol):
    """Protocol for image preprocessing (resize, normalize, pad)."""

    def to_raw(
        self, image: npt.NDArray[np.uint8],
    ) -> npt.NDArray[np.float32]: ...

    def to_bytes(self, tensor: npt.NDArray[np.float32]) -> bytes: ...
```

### 6.3 Observer Protocol (`protocols.py`)

```python
@runtime_checkable
class ObserverProtocol(Protocol[TEvent]):
    """Observes events without producing output (metrics, logging)."""

    def observe(self, envelope: Envelope[TEvent]) -> None: ...
```

### 6.4 Aggregation Rule Protocol (`protocols.py`)

```python
@runtime_checkable
class AggregationRuleProtocol(Protocol[TEvent]):
    """Extracts grouping key from events for time-windowed aggregation."""

    @property
    def metric_name(self) -> str: ...

    @property
    def window_seconds(self) -> int: ...

    def extract_key(self, event: TEvent) -> str: ...

    def aggregate(self, count: int) -> float: ...
```

---

## 7. Source & Sink Layer

### 7.1 RedisStreamSource

Implements `Source[T]`. Runs in a dedicated thread. Performs blocking
`XREADGROUP` calls and pushes events via `emit()`.

```python
class RedisStreamSource:
    """Reads from a Redis Stream using consumer group semantics.

    Runs in a dedicated thread. The framework assigns stop: Event
    before starting the thread. Calls emit(event) for each decoded
    message. Errors go to the error queue — source never crashes.
    """

    stop: threading.Event

    def __init__(
        self,
        config: SourceConfig,
        redis_config: RedisConnectionConfig,
        event_model_class: type[BaseModel],
    ) -> None: ...

    def __call__(self, emit: Callable[[Any], None]) -> None:
        """Main loop running in dedicated thread.

        while not self.stop.is_set():
            try:
                messages = redis.xreadgroup(...)
                for message_id, fields in messages:
                    try:
                        event = event_model_class.model_validate(fields)
                        emit(event)
                    except ValidationError:
                        error_queue.put(ConnectorError(...))
            except redis.ConnectionError:
                error_queue.put(ConnectorError(...))
                self.stop.wait(timeout=1.0)  # backoff
            except redis.TimeoutError:
                pass  # normal for BLOCK reads
        """
```

### 7.2 Consumer Group Management

```python
class ConsumerGroupManager:
    """Manages consumer group lifecycle and acknowledgment."""

    def ensure_group_exists(self) -> None:
        """XGROUP CREATE with $ and MKSTREAM. Catches BUSYGROUP."""

    def read_pending_newest_first(
        self, ttl_s: float = 60.0,
    ) -> list[tuple[str, dict[str, str]]]:
        """XREADGROUP with id=0, sort by message ID descending.
        Drop messages older than ttl_s: XACK without processing."""

    def read_new(self) -> list[tuple[str, dict[str, str]]]:
        """XREADGROUP with id='>' (new messages, blocking)."""

    def acknowledge(self, message_id: str) -> None:
        """XACK a processed message."""
```

### 7.3 RedisStreamSink

Implements `Sink[T]`. Runs in a dedicated thread. Reads from a queue and
writes via `XADD`.

```python
class RedisStreamSink:
    """Writes events to a Redis Stream.

    Runs in a dedicated thread. On successful XADD, triggers
    acknowledgment callback.
    """

    stop: threading.Event

    def __call__(self, event: dict[str, str | bytes]) -> None:
        """XADD to stream, trigger ack callback."""
```

---

## 8. Inference Layer

### 8.1 InferenceInput as Framework Intercept Marker

`InferenceInput` is not a real Processor — it is a type-based intercept point.
When a Processor returns `InferenceInput`, the framework intercepts it, routes
to the worker pool via `mp.request_queue`. `InferenceOutput` comes back from
`mp.result_queue`, and the framework feeds it to the next Processor in the
pipeline. The pipeline is split at the `InferenceInput` boundary.

```python
class InferenceInput(BaseModel, arbitrary_types_allowed=True):
    model_name: str
    model_version: str = ""           # empty = latest
    raw_image: npt.NDArray[np.float32]
    input_name: str = "images"
    output_names: tuple[str, ...] = ("output0",)


class InferenceOutput(BaseModel, arbitrary_types_allowed=True):
    model_name: str
    model_version: str
    outputs: dict[str, npt.NDArray[np.float32]]
    inference_time_ms: float
```

### 8.2 Multiprocessing Worker Pool

Triton inference runs in a **pool of N worker processes**. Each worker process
creates its own `tritonclient.grpc.InferenceServerClient` and uses
`client.async_infer()` with callback-based retry. The main process distributes
`InferenceInput` objects via `multiprocessing.Queue` and collects results via
a separate result queue.

This design avoids Python GIL contention for gRPC serialization/deserialization,
allows true parallel utilization of Triton's gRPC endpoint, and isolates inference
failures from the main orchestrator.

```
Main Process                        Worker Processes
┌──────────────────┐               ┌─────────────────────────┐
│                  │  request_queue │  Worker-0               │
│  Application     │──────────────►│  gRPC client (own conn) │
│  (orchestrator)  │               │  async_infer + callback  │
│                  │  request_queue │  retry via Timer         │
│                  │──────────────►├─────────────────────────┤
│                  │               │  Worker-1               │
│                  │◄──────────────│  gRPC client (own conn) │
│                  │  result_queue  │  async_infer + callback  │
│                  │◄──────────────│  retry via Timer         │
│                  │  result_queue  ├─────────────────────────┤
│                  │               │  Worker-N               │
│                  │◄──────────────│  ...                     │
└──────────────────┘  result_queue  └─────────────────────────┘
```

```python
class InferenceWorkerPool:
    """Manages a pool of inference worker processes.

    Each worker has its own Triton gRPC client for true parallelism.
    Requests/results flow through multiprocessing.Queue.
    All queues have configurable maxsize.
    """

    def __init__(
        self, config: InferenceConfig, queue_config: QueueConfig,
    ) -> None:
        self._request_queue: mp.Queue[InferenceInput] = mp.Queue(
            maxsize=queue_config.maxsize,
        )
        self._result_queue: mp.Queue[InferenceOutput | InferenceError] = (
            mp.Queue(maxsize=queue_config.maxsize)
        )
        self._workers: list[mp.Process] = []
        self._config = config

    def start(self) -> None:
        """Spawn N worker processes."""
        for i in range(self._config.num_inference_workers):
            worker = mp.Process(
                target=_inference_worker_main,
                args=(
                    i,
                    self._config,
                    self._request_queue,
                    self._result_queue,
                ),
                daemon=True,
            )
            worker.start()
            self._workers.append(worker)

    def submit(self, request: InferenceInput) -> bool:
        """Put request onto the shared request queue.
        Returns False and logs + emits queue_overflow metric if full.
        """
        try:
            self._request_queue.put_nowait(request)
        except queue.Full:
            logger.error(
                "request_queue full, dropping InferenceInput for {}",
                request.model_name,
            )
            return False
        return True

    def get_results(
        self, timeout: float = 0.1,
    ) -> list[InferenceOutput | InferenceError]:
        """Drain available results from result queue (non-blocking)."""
        results: list[InferenceOutput | InferenceError] = []
        while True:
            try:
                results.append(self._result_queue.get_nowait())
            except queue.Empty:
                break
        return results

    def shutdown(self) -> None:
        """Send poison pills and join workers."""
        for _ in self._workers:
            self._request_queue.put(None)  # sentinel
        for worker in self._workers:
            worker.join(timeout=10)
```

### 8.3 Inference Worker with Async Retry via Callback

Each worker uses `client.async_infer()` (non-blocking). Retry happens inside
the callback via `threading.Timer` — on failure, schedule retry with exponential
delay, re-call `async_infer` with a new callback. The semaphore is released only
on final success or final failure.

```python
def _inference_worker_main(
    worker_id: int,
    config: InferenceConfig,
    request_queue: mp.Queue[InferenceInput | None],
    result_queue: mp.Queue[InferenceOutput | InferenceError],
) -> None:
    """Entry point for each inference worker process.

    Creates its own gRPC client. Reads requests from queue.
    Uses async_infer with callback-based retry for concurrency.
    """
    client = tritonclient.grpc.InferenceServerClient(
        url=config.grpc_url,
    )
    semaphore = threading.Semaphore(config.max_concurrent_per_worker)

    def _make_callback(
        req: InferenceInput,
        attempt: int,
        start_time_ns: int,
    ) -> Callable:
        def _on_complete(result, error):
            if error is None:
                elapsed_ms = (
                    (time.monotonic_ns() - start_time_ns) / 1_000_000
                )
                result_queue.put(InferenceOutput(
                    model_name=req.model_name,
                    model_version=req.model_version,
                    outputs=_extract_outputs(result, req.output_names),
                    inference_time_ms=elapsed_ms,
                ))
                semaphore.release()
                return

            if attempt >= config.max_retry_attempts:
                result_queue.put(InferenceError(
                    model_name=req.model_name,
                    model_version=req.model_version,
                    error_type=type(error).__name__,
                    error_message=str(error),
                    retry_count=attempt,
                    timestamp_ms=int(time.time() * 1000),
                ))
                semaphore.release()
                return

            delay = (
                config.retry_base_delay_s
                * config.retry_backoff_multiplier ** attempt
            )

            def _retry():
                client.async_infer(
                    model_name=req.model_name,
                    model_version=req.model_version,
                    inputs=_build_inputs(req),
                    callback=_make_callback(
                        req, attempt + 1, start_time_ns,
                    ),
                    outputs=_build_requested_outputs(req),
                )

            timer = threading.Timer(delay, _retry)
            timer.start()

        return _on_complete

    while True:
        request = request_queue.get()
        if request is None:  # poison pill
            break
        semaphore.acquire()

        start_ns = time.monotonic_ns()
        client.async_infer(
            model_name=request.model_name,
            model_version=request.model_version,
            inputs=_build_inputs(request),
            callback=_make_callback(request, attempt=1, start_time_ns=start_ns),
            outputs=_build_requested_outputs(request),
        )
```

### 8.4 Rolling Release — ModelVersionTracker (`inference/model_registry.py`)

Triton uses `model-control-mode=poll` with 10s interval. When a new model version
appears in the repository, Triton loads it automatically.

Each inference worker process runs its own `ModelVersionTracker` instance. Since
every worker has its own gRPC client, it can independently query Triton for model
readiness. Version state is per-worker (no shared memory needed) — all workers
converge to the same version within one poll interval.

```python
class ModelVersionTracker:
    """Tracks model versions and provides safe version resolution.

    Runs inside each inference worker process with its own gRPC client.
    Polls Triton's model repository index periodically.
    Ensures in-flight requests complete on their version before switching.
    """

    def resolve_version(self, model_name: str) -> str:
        """Return the latest ready version for a model.

        Queries is_model_ready() if cache is stale.
        """

    def track_inflight(self, model_name: str, version: str) -> None:
        """Increment in-flight counter (thread-safe via Lock)."""

    def release_inflight(self, model_name: str, version: str) -> None:
        """Decrement in-flight counter."""

    def _poll_versions(self) -> None:
        """Periodic task (runs in a timer thread within worker process):
        1. Query get_model_repository_index()
        2. For each model, check is_model_ready(model, new_version)
        3. If ready, update current version mapping
        4. Old version usable until in-flight count drops to zero
        5. Emit metric: model_version_switch (via result_queue)
        """
```

**Version switch sequence:**

```
T=0:   Model "yolo26s" v1 active, 5 in-flight across all workers
T=1:   New v2 uploaded to model_repository/yolo26s/2/
T=10:  Triton polls, starts loading v2
T=15:  Worker-0 _poll_versions detects v2, is_model_ready → True
       Worker-0 switches to v2, Worker-1 still on v1 (polls independently)
T=20:  Worker-1 _poll_versions detects v2, switches to v2
       All workers now on v2. In-flight v1 requests complete naturally.
```

**Context manager for safe inference (used inside worker):**

```python
@contextmanager
def tracked_inference(
    tracker: ModelVersionTracker, model_name: str,
) -> Iterator[str]:
    version = tracker.resolve_version(model_name)
    tracker.track_inflight(model_name, version)
    try:
        yield version
    finally:
        tracker.release_inflight(model_name, version)
```

---

## 9. Image Preprocessing

### 9.1 Single Decoder via Config

User picks one decoder via config (`decoder: opencv | turbojpeg`). There is no
dispatcher, no magic bytes detection, no multi-decoder routing.

- **OpenCV** (`opencv`): Handles JPEG, PNG, AVIF, BMP natively via
  `cv2.imdecode()`. Default choice for format flexibility.
- **TurboJPEG** (`turbojpeg`): Faster for JPEG-only workloads via PyTurboJPEG.
  Thread-safe with thread-local instances.

```python
class OpenCVDecoder:
    def decode(self, raw_bytes: bytes) -> npt.NDArray[np.uint8]:
        buf = np.frombuffer(raw_bytes, dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)


class TurboJPEGDecoder:
    def decode(self, raw_bytes: bytes) -> npt.NDArray[np.uint8]:
        jpeg = self._get_thread_local()
        return jpeg.decode(raw_bytes)  # returns BGR numpy array
```

### 9.2 Configurable Decoder Pool

The pool type is configurable: `decoder_pool_type: thread | process` in config,
defaulting to `thread` since both OpenCV and TurboJPEG release GIL during decode.
Uses `ThreadPoolExecutor` or `ProcessPoolExecutor` accordingly.

```python
class DecoderPool:
    """Configurable pool for image decoding.

    Pool type (thread/process) and decoder type (opencv/turbojpeg)
    determined by config. Decoded numpy arrays returned via Future.

    Note: for ProcessPoolExecutor, raw_bytes (input) and ndarray (output)
    are serialized across process boundary via pickle. For typical camera
    images (100KB-2MB in, ~3.7MB out at 640x640x3 uint8) this overhead
    is acceptable compared to decode time.
    """

    def __init__(self, config: PreprocessingConfig) -> None:
        executor_cls = (
            ThreadPoolExecutor
            if config.decoder_pool_type == DecoderPoolType.THREAD
            else ProcessPoolExecutor
        )
        self._executor = executor_cls(
            max_workers=config.decoder_pool_workers,
        )

    def decode(
        self, raw_bytes: bytes,
    ) -> Future[npt.NDArray[np.uint8]]:
        """Submit decode to pool, return Future."""
        return self._executor.submit(_decode_in_worker, raw_bytes)

    def decode_batch(
        self,
        items: Sequence[bytes],
    ) -> list[Future[npt.NDArray[np.uint8]]]:
        """Submit multiple decodes, return list of Futures."""
```

### 9.3 DALI Ensemble Alternative

For YOLO models with DALI preprocessing, decoding happens server-side in Triton.
The `ImagePreprocessorProtocol.to_bytes()` passes raw JPEG bytes through, and
Triton's DALI backend handles decode + preprocess. The `InferenceInput.raw_image`
field carries raw bytes as a 1-D uint8 array.

---

## 10. Model Repository

### 10.1 NMS as Separate Triton Python Backend Model

Non-maximum suppression runs as a shared Triton model using the PyTorch backend
with `torchvision.ops.nms`. This model is reused across all non-end2end ensembles.

```
model_repository/
  nms_postprocess/                  # Shared NMS model (PyTorch backend)
    config.pbtxt
    1/
      model.py                      # torchvision.ops.nms

  yolo11s/                          # Standard detection (needs NMS)
    config.pbtxt
    1/
      model.onnx

  yolo12s/                          # Standard detection (needs NMS)
    config.pbtxt
    1/
      model.onnx

  yolo26s/                          # End2end (built-in NMS, no ensemble)
    config.pbtxt
    1/
      model.onnx

  yolo11s_ensemble/                 # Ensemble: yolo11s → nms_postprocess
    config.pbtxt

  yolo12s_ensemble/                 # Ensemble: yolo12s → nms_postprocess
    config.pbtxt
```

### 10.2 Ensemble Topology

End2end models (like `yolo26s`) export with NMS baked into the ONNX graph via the
`end2end_export` flag in the model registry. They run standalone without an ensemble.

Standard detection models route through a shared `nms_postprocess` model:

```
yolo11s → nms_postprocess → yolo11s_ensemble
yolo12s → nms_postprocess → yolo12s_ensemble
yolo26s (standalone, end2end, no ensemble needed)
```

The `nms_postprocess` model uses `torchvision.ops.nms` to filter overlapping
detections. One shared instance serves all non-end2end ensembles.

---

## 11. Metrics

### 11.1 MetricEvent Model

```python
class MetricEvent(BaseModel):
    component_name: str               # from APP_NAME env var
    measure: str                      # metric name
    value: float
    timestamp_ms: int                 # unix epoch ms
    tags: dict[str, str] = {}         # queue name, model name, etc.
```

### 11.2 Metrics Flow

```
Event transit tracking ──┐
                          │
Triton stats polling ─────┤   MetricEvent
                          ├──────────────► MetricBuffer ──► MetricEmitter ──► Redis "metrics"
Error aggregation ────────┤
                          │
Queue overflow tracking ──┤
                          │
Business aggregation ─────┘
  (TimeWindowAggregator)
```

### 11.3 Transit Time Tracking

Each `Envelope` records timestamps at key states:

| State | When |
|---|---|
| `received` | Source reads from Redis |
| `decoded` | Pydantic model deserialized |
| `queued` | Placed on processing queue |
| `processing_start` | Processor begins processing |
| `inference_submitted` | Triton request sent |
| `inference_complete` | Triton response received |
| `output_sent` | Result written to output stream |
| `acknowledged` | XACK sent |

Auto-emitted metrics: `event_total_transit_ms`, `event_queue_wait_ms`,
`event_inference_ms`.

### 11.4 TritonMetricsCollector

Periodic task (runs in a timer thread). Polls Triton stats via gRPC
`get_inference_statistics()`:

- `triton_inference_count` — cumulative inference count per model
- `triton_inference_queue_time_us` — cumulative queue time
- `triton_inference_compute_infer_time_us` — cumulative inference compute time
- `triton_inference_compute_input_time_us` — cumulative input compute time
- `triton_inference_compute_output_time_us` — cumulative output compute time

### 11.5 MetricBuffer & MetricEmitter

```python
class MetricBuffer:
    """Thread-safe buffer accumulating MetricEvent instances.

    Multiple producers (transit tracker, Triton collector, error collector,
    queue overflow tracker) write concurrently. MetricEmitter drains
    periodically.
    """

    def push(self, event: MetricEvent) -> None: ...
    def drain(self) -> list[MetricEvent]: ...


class MetricEmitter:
    """Drains MetricBuffer on interval, writes to Redis output stream.

    Runs as a periodic timer thread.
    Serializes each MetricEvent to Redis fields and XADD.
    """
```

---

## 12. Aggregation Metrics Library (`triton_metrics_aggregator`)

### 12.1 Purpose

Analyze incoming events with pydantic validators and send aggregation results as
metrics. Example: input event has `camera` attribute, aggregate by camera value per
minute, emit "images_per_camera_per_minute" metric.

### 12.2 Core Types

```python
class AggregationRule(BaseModel):
    metric_name: str                  # "images_per_camera_per_minute"
    field_path: str                   # dot-separated path, "camera"
    window_seconds: int = 60
    aggregation: str = "count"        # "count" | "sum" | "avg"


class TimeWindowAggregator(Generic[TEvent]):
    """Aggregates events by field values within time windows.

    Validates field_path exists on TEvent at registration time
    using pydantic model_fields (fail fast).

    Usage:
        aggregator = TimeWindowAggregator[CameraImageEvent](
            rules=[AggregationRule(
                metric_name="images_per_camera_per_minute",
                field_path="camera",
                window_seconds=60,
            )],
        )
        aggregator.observe(event)       # on each event
        metrics = aggregator.flush()    # periodically
        # → [MetricEvent(
        #     measure="images_per_camera_per_minute.cam_01", value=42,
        # )]
    """

    def __init__(self, rules: Sequence[AggregationRule]) -> None:
        self._validate_field_paths()    # fail fast

    def observe(self, event: TEvent) -> None: ...

    def flush(self) -> list[MetricEvent]: ...
```

### 12.3 Integration

`TimeWindowAggregator` integrates as an `ObserverProtocol` component.
The framework feeds it events without expecting output routing.
Flushed metrics go to the shared `MetricBuffer`.

---

## 13. Error Handling

### 13.1 Error Model Hierarchy

```python
class ConnectorError(BaseModel):
    connector_name: str
    stream_name: str
    message_id: str | None
    error_type: str                   # exception class name
    error_message: str
    timestamp_ms: int

class InferenceError(BaseModel):
    model_name: str
    model_version: str
    error_type: str
    error_message: str
    retry_count: int
    timestamp_ms: int

class PreprocessingError(BaseModel):
    message_id: str
    decoder_type: str
    error_type: str
    error_message: str
    timestamp_ms: int
```

### 13.2 Error Flow

```
Source Threads ──────► error_queue ──────────┐  (intra-process queue)
                                             │
Decoder Pool ────────► error_queue ──────────┤  (intra-process queue)
                                             │
Inference Workers ──► mp.result_queue ───────┤  (cross-process, InferenceError)
  (errors returned as InferenceError         │
   via multiprocessing result queue)         │
                                             ▼
                                      ErrorHandler (main thread)
                                             │
                                   ┌─────────┴──────────┐
                                   ▼                     ▼
                            ErrorCollector          logger.error()
                            (buckets by type)       (immediate log)
                                   │
                                   ▼ (on flush interval)
                            MetricBuffer → MetricEmitter → Redis "metrics"
```

### 13.3 ErrorCollector

```python
class ErrorCollector:
    """Collects errors by interval, emits aggregated metric counts.

    _buckets: dict[error_type, count] within current window.
    Flushed every error_flush_interval_s (default 30s).
    """

    def record(self, error_type: str) -> None:
        """Increment counter for error type in current window."""

    def flush(self) -> list[MetricEvent]:
        """Drain buckets → MetricEvent per type."""
```

### 13.4 Source and Sink Resilience

Sources and sinks catch specific exception types and route errors to the error
queue. They **never crash**:

- `redis.ConnectionError` → error_queue + backoff sleep
- `redis.TimeoutError` → normal for BLOCK reads, continue
- `pydantic.ValidationError` → error_queue, skip message, continue
- Any other specific exception → error_queue, continue

---

## 14. Recovery & Position Tracking

### 14.1 Consumer Group Semantics

Redis consumer groups provide built-in position tracking:

- `XGROUP CREATE ... $` — start reading from new messages only
- `XREADGROUP ... >` — read only new (undelivered) messages
- `XREADGROUP ... 0` — read pending (delivered but unacked) messages
- `XACK` — acknowledge successful processing

Messages remain pending until acknowledged. On restart, pending messages
are redelivered automatically.

### 14.2 At-Least-Once Delivery

XACK is sent **only after** the result is successfully written to the output
stream. If the app crashes between processing and output write, the message
remains pending and will be reprocessed on restart. Trade-off: potential duplicate
processing, acceptable for inference workloads (idempotent results).

### 14.3 Newest-First Priority with Interleave Ratio + TTL

On restart recovery, pending messages are sorted by Redis message ID descending
(newest first). Two separate queues:

- `pending_queue` (PriorityQueue) — drained with newest-first ordering
- `live_queue` (Queue) — normal FIFO for new messages

Reader logic with configurable interleave ratio (default 4:1) and TTL (default
60s):

```python
def _combined_reader(
    live_queue,
    pending_queue,
    handler,
    stop_event,
    interleave_ratio: int = 4,
    pending_ttl_s: float = 60.0,
):
    live_count = 0
    while not stop_event.is_set():
        # 1. Try live_queue (up to interleave_ratio messages)
        try:
            envelope = live_queue.get_nowait()
            handler(envelope)
            live_count += 1
            if live_count < interleave_ratio:
                continue
            live_count = 0
        except queue.Empty:
            live_count = 0

        # 2. Interleave: process 1 pending message
        try:
            envelope = pending_queue.get_nowait()
            age_s = _envelope_age_seconds(envelope)
            if age_s > pending_ttl_s:
                # Stale: XACK without processing, drop
                acknowledge(envelope.message_id)
                logger.warning(
                    "Dropped stale pending message {} (age={}s)",
                    envelope.message_id,
                    age_s,
                )
                continue
            handler(envelope)
            continue
        except queue.Empty:
            pass

        # 3. Both empty — block on live briefly
        try:
            envelope = live_queue.get(timeout=0.1)
            handler(envelope)
            live_count = 1
        except queue.Empty:
            continue
```

---

## 15. Process & Thread Model

### 15.1 Process Map

```
┌────────────────────────────────────────────────────────────────────────┐
│  MAIN PROCESS                                                          │
│                                                                        │
│  Main Thread:                                                          │
│  - Application orchestrator                                            │
│  - Queue reader (drains live + pending queues, interleave ratio)       │
│  - Pipeline dispatch: unwrap Envelope → call Processor.__call__        │
│  - Intercept InferenceInput → put onto mp.request_queue                │
│  - Drain mp.result_queue → wrap in Envelope → feed next Processor      │
│  - ErrorHandler (drains error queue)                                   │
│  - current_context() set via contextvars before each __call__          │
│                                                                        │
│  ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐       │
│  │ Thread: Source-1  │ │ Thread: Source-2  │ │ Thread: Sink-1   │       │
│  │ XREADGROUP        │ │ XREADGROUP        │ │ XADD             │       │
│  │ stream_a          │ │ stream_b          │ │ results_stream   │       │
│  │ → emit(event)     │ │ → emit(event)     │ │ ← output_queue   │       │
│  └──────────────────┘ └──────────────────┘ └──────────────────┘       │
│                                                                        │
│  ┌──────────────────┐ ┌──────────────────┐                             │
│  │ Thread: Sink-2    │ │ Timer: Metrics    │                             │
│  │ XADD              │ │ - TritonMetrics   │                             │
│  │ metrics_stream    │ │ - Error flush     │                             │
│  │ ← metrics_queue   │ │ - Metric emit     │                             │
│  └──────────────────┘ └──────────────────┘                             │
└────────────────────────────────────────────────────────────────────────┘
         │ mp.request_queue                    ▲ mp.result_queue
         ▼                                     │
┌──────────────────────────────────────────────────────────────────────┐
│  INFERENCE WORKER PROCESSES (N separate processes)                    │
│                                                                      │
│  ┌─────────────────────────┐  ┌─────────────────────────┐           │
│  │  Worker-0                │  │  Worker-1                │           │
│  │  gRPC client (own conn)  │  │  gRPC client (own conn)  │           │
│  │  async_infer + callback  │  │  async_infer + callback  │           │
│  │  retry via Timer         │  │  retry via Timer         │           │
│  │  ModelVersionTracker     │  │  ModelVersionTracker     │           │
│  │  Semaphore(8)            │  │  Semaphore(8)            │           │
│  └─────────────────────────┘  └─────────────────────────┘           │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│  DECODER POOL (ThreadPoolExecutor default, ProcessPoolExecutor opt.) │
│                                                                      │
│  Single decoder type per config:                                     │
│  - opencv: cv2.imdecode() — JPEG/PNG/AVIF/BMP                       │
│  - turbojpeg: PyTurboJPEG — JPEG-only, faster                       │
│                                                                      │
│  Both release GIL → ThreadPool is default and sufficient             │
│  (configurable via decoder_pool_type: thread | process)              │
└──────────────────────────────────────────────────────────────────────┘
```

### 15.2 Queue Topology

All queues have configurable `maxsize` (via `QueueConfig`). When a queue is full,
the item is dropped (not blocked), loguru logs an error, and a `queue_overflow`
metric with the queue name tag is emitted.

| Queue | Type | Boundary | maxsize | Producer | Consumer |
|---|---|---|---|---|---|
| `live_queue[stream]` | `Queue[Envelope[T]]` | intra | config | Source thread | Main thread |
| `pending_queue[stream]` | `PriorityQueue[Envelope[T]]` | intra | config | Startup recovery | Main thread |
| `output_queue[stream]` | `Queue[T]` | intra | config | Main thread | Sink thread |
| `error_queue` | `Queue[Error]` | intra | config | All threads | Main thread |
| `metric_buffer` | `MetricBuffer` | intra | config | All metric producers | MetricEmitter |
| `request_queue` | `mp.Queue[InferenceInput]` | **cross** | config | Main process | Workers |
| `result_queue` | `mp.Queue[InferenceOutput\|Error]` | **cross** | config | Workers | Main process |

### 15.3 Backpressure: Drop on Full + Metrics

When any queue is full:
1. Drop the item (never block the producer)
2. Log via loguru: `logger.error("queue {} full, dropping item", queue_name)`
3. Emit `queue_overflow` metric with `tags={"queue": queue_name}`

This prevents cascading backpressure from stalling the entire pipeline. Dropped
items are visible in metrics dashboards for capacity planning.

### 15.4 Serialization Across Process Boundary

Objects crossing the `multiprocessing.Queue` boundary are serialized via pickle:

- **InferenceInput**: contains `model_name` (str), `model_version` (str),
  `raw_image` (numpy ndarray), `input_name` (str), `output_names` (tuple of str).
  Numpy arrays pickle efficiently — a 640x640x3 float32 tensor is ~4.7MB.
- **InferenceOutput**: contains `outputs` dict of numpy arrays.
  Same efficient numpy pickle.
- **InferenceError**: contains only strings and ints — trivial to pickle.

### 15.5 Phase 4C — Shared Memory IPC Optimization

After Phase 5 benchmarks confirm pickle IPC is the bottleneck, optimize with
`multiprocessing.shared_memory` for numpy arrays across the process boundary.
This eliminates the pickle serialization overhead for large tensors by passing
only a shared memory handle and metadata (shape, dtype) through the queue.

### 15.6 Inference Concurrency (per worker)

Each worker process runs `tritonclient.grpc.InferenceServerClient.async_infer()`.
This is non-blocking within the worker — it accepts a callback and returns
immediately. The callback fires in a gRPC internal thread when the response arrives
and puts the result onto the shared `result_queue` (multiprocessing-safe).
A `threading.Semaphore` per worker limits concurrent in-flight requests
(`max_concurrent_per_worker`, default 8).

---

## 16. Envelope Type (`engine/event.py`)

```python
@dataclass(slots=True)
class Envelope(Generic[T]):
    payload: T
    stream_name: str
    message_id: str
    priority: int                     # for PriorityQueue ordering
    timestamps: dict[str, int] = field(default_factory=dict)

    def track_state(self, state: str) -> None:
        self.timestamps[state] = time.monotonic_ns()

    def transit_time_ms(self, from_state: str, to_state: str) -> float:
        return (
            (self.timestamps[to_state] - self.timestamps[from_state])
            / 1_000_000
        )

    def derive(self, payload: U) -> Envelope[U]:
        """Create new Envelope with different payload, inheriting metadata."""
        return Envelope(
            payload=payload,
            stream_name=self.stream_name,
            message_id=self.message_id,
            priority=self.priority,
            timestamps=dict(self.timestamps),
        )

    def __lt__(self, other: Envelope[Any]) -> bool:
        return self.priority < other.priority
```

Priority calculation for newest-first:

```python
def message_id_to_priority(message_id: str, is_pending: bool) -> int:
    if not is_pending:
        return 0                      # live messages — highest priority
    timestamp_str = message_id.split("-")[0]
    return -int(timestamp_str)        # newer = smaller negative = higher priority
```

---

## 17. Pipeline Registry & Type-Annotation Wiring

### 17.1 How It Works

When a user registers a Processor class, the registry:

1. Calls `typing.get_type_hints()` on the `__call__` method
2. Extracts `event: X` parameter type
3. Extracts return type `Y | None` (strips `None` — means filter)
4. Builds directed graph: `X → Processor → Y`
5. Connects Sources whose output type matches Processor input types
6. Connects Processors whose output type matches Sink input types
7. Recognizes `InferenceInput` output as the framework intercept point —
   pipeline is split here, routed through worker pool

```python
class PipelineRegistry:
    def register(self, processor_class: type) -> None:
        hints = typing.get_type_hints(processor_class.__call__)
        input_type = hints["event"]
        output_type = hints["return"]
        # Strip None from Optional/Union for wiring
        if typing.get_origin(output_type) is types.UnionType:
            output_type = next(
                a for a in typing.get_args(output_type) if a is not type(None)
            )
        self._graph.add_node(processor_class, input_type, output_type)

    def build_graph(self) -> ProcessingGraph:
        """Wire processors, sources, sinks.
        Raises ValueError on missing sources, cycles, or dangling outputs.
        """
```

### 17.2 Wiring Example

```
SourceConfig(event_model=CameraImageEvent)
    │
    │  Source.__call__(emit) pushes CameraImageEvent
    ▼
CameraToInference: Processor[CameraImageEvent, InferenceInput]
    │
    │  Framework intercepts InferenceInput → mp.request_queue
    ║  ═══════════════ process boundary ═══════════════
    │
    ▼
InferenceWorkerPool (Worker-N picks up request)
    │  gRPC async_infer + callback → InferenceOutput
    │
    ║  ═══════════════ process boundary ═══════════════
    │  mp.result_queue
    ▼
DetectionResultHandler: Processor[InferenceOutput, DetectionResult]
    │
    │  type match: DetectionResult → Sink
    ▼
SinkConfig(stream_name="detection_results")
```

### 17.3 Example User Code

```python
class CameraImageEvent(BaseModel):
    camera: str
    image_data: bytes
    timestamp: int


class DetectionResult(BaseModel):
    camera: str
    detections: list[dict[str, float]]
    timestamp: int


class CameraToInference:
    """Implements Processor[CameraImageEvent, InferenceInput]."""

    def __init__(self, decoder_pool: DecoderPool) -> None:
        self._decoder_pool = decoder_pool

    def __call__(
        self, event: CameraImageEvent,
    ) -> InferenceInput:
        future = self._decoder_pool.decode(event.image_data)
        decoded = future.result()     # block until decoded
        raw = decoded.astype(np.float32) / 255.0
        return InferenceInput(
            model_name="yolo26s", raw_image=raw,
        )


class DetectionResultHandler:
    """Implements Processor[InferenceOutput, DetectionResult]."""

    def __call__(
        self, event: InferenceOutput,
    ) -> DetectionResult:
        # Extract detections from model output tensors
        detections = _parse_detections(event.outputs)
        ctx = current_context()
        return DetectionResult(
            camera=ctx.metadata.get("camera", "unknown"),
            detections=detections,
            timestamp=int(time.time() * 1000),
        )
```

---

## 18. Application Bootstrap

```python
class Application:
    @classmethod
    def from_config(cls, config: AppConfig) -> Application: ...

    def run(self) -> None:
        """Main run sequence:
        1. Create Redis connections
        2. Create InferenceWorkerPool (spawns N worker processes)
        3. Create DecoderPool (Thread or ProcessPoolExecutor per config)
        4. Import and register user processors (from config.processors)
        5. Build processing graph via PipelineRegistry
        6. Create consumer groups for sources
        7. Load pending messages (newest-first, drop stale via TTL)
           into pending queues
        8. Create stop Events, assign to each Source/Sink
        9. Start source threads (each Source.__call__(emit) in own thread)
        10. Start sink threads
        11. Start inference worker pool (worker processes)
        12. Start timer threads:
            - TritonMetricsCollector (periodic)
            - ErrorCollector flush (periodic)
            - MetricEmitter (periodic)
        13. Run main loop:
            - Combined reader: drain live + pending queues
              (interleave ratio, TTL check)
            - Unwrap Envelope → set current_context() → Processor.__call__
            - Intercept InferenceInput → submit to worker pool
            - Drain results from worker pool (mp.result_queue)
            - Wrap InferenceOutput in Envelope → feed next Processor
            - Route final output to Sink queue
            - Handle errors
        14. On shutdown signal (SIGINT/SIGTERM):
            - Call source.stop.set() for each Source
            - Call sink.stop.set() for each Sink
            - Drain remaining queue items
            - Shutdown inference worker pool (poison pills + join)
            - Shutdown decoder pool
            - Flush final metrics
            - Close Redis connections
        """

    def shutdown(self) -> None: ...
```

**Entry point (`__main__.py`):**

```python
@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    app_config = AppConfig(**OmegaConf.to_container(cfg, resolve=True))
    app = Application.from_config(app_config)
    app.run()
```

---

## 19. New Dependencies

### triton_redis_connector/pyproject.toml additions

```
hydra-core>=1.3.0
PyTurboJPEG>=1.7.0
opencv-python-headless>=4.9.0
```

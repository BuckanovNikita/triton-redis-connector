# Triton Redis Connector — Architectural Design

## 1. Overview

### Problem Statement

ML inference pipelines need a reliable bridge between data sources (camera streams, file uploads) and NVIDIA Triton Inference Server. The system must handle:

- Reading events from Redis Streams with consumer group semantics
- Decoding and preprocessing images for YOLO models
- Sending async inference requests to Triton via gRPC
- Writing results back to Redis Streams
- Collecting metrics (transit times, Triton stats, errors, business aggregations)
- Recovering after restarts with newest-first priority
- Safe rolling model version updates

### System Context

```
Camera/Sensors → Redis Streams → [triton-redis-connector] → Triton Server
                                        ↓                        ↓
                                  Redis Streams ←── results ──────┘
                                  (results, metrics)
```

The connector is a multi-threaded Python application. Connector threads perform
blocking Redis I/O. A dedicated inference module uses Triton's async gRPC client
where results return as futures. All other code is synchronous.

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
      protocols.py                  # All user-facing Protocol classes

      config/
        __init__.py
        schema.py                   # Pydantic models for Hydra-loaded config
        hydra_app.py                # Hydra initialization, config composition

      connectors/
        __init__.py
        base.py                     # InputConnector / OutputConnector Protocols
        redis_stream.py             # RedisStreamInputConnector, RedisStreamOutputConnector
        consumer_group.py           # Consumer group management, position tracking

      engine/
        __init__.py
        app.py                      # Application — main entry point
        registry.py                 # Component registry, type-annotation wiring
        queue_bus.py                # Thread-safe queues connecting threads
        event.py                    # Envelope[T] — timestamped event wrapper

      inference/
        __init__.py
        client.py                   # TritonInferenceClient — async gRPC with futures
        request.py                  # InferenceRequest / InferenceResult models
        model_registry.py           # ModelVersionTracker — rolling release
        retry.py                    # RetryPolicy with exponential backoff

      preprocessing/
        __init__.py
        protocols.py                # ImageDecoder Protocol, ImagePreprocessor Protocol
        jpeg_decoder.py             # TurboJPEGDecoder
        avif_decoder.py             # CV2AvifDecoder
        dispatcher.py               # DecoderDispatcher — routes by content type
        pool.py                     # ParallelDecoderPool — thread pool wrapper

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
      streams/
        input/
          camera_images.yaml
        output/
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
│                        APPLICATION LAYER (engine/)                    │
│                                                                      │
│  Application          ComponentRegistry        QueueBus              │
│  (orchestrator)       (type-hint wiring)       (thread-safe queues)  │
└────┬─────────────────────┬───────────────────────┬───────────────────┘
     │                     │                       │
┌────▼─────────────────────▼───────────────────────▼───────────────────┐
│                        CONNECTOR LAYER (connectors/)                  │
│                                                                      │
│  ┌─────────────────┐          ┌──────────────────┐                   │
│  │ InputConnector   │ Thread   │ OutputConnector   │ Thread            │
│  │ (Redis XREAD)    │────────► │ (Redis XADD)      │                   │
│  └─────────────────┘  Queue   └──────────────────┘                   │
│                                                                      │
│  Protocol-based: InputConnectorProtocol / OutputConnectorProtocol     │
│  Redis is the implementation. Kafka can be added later.              │
└──────────────────────────────────────────────────────────────────────┘
     │                                        ▲
     │ Queue[Envelope[T]]                     │ Queue[dict]
     ▼                                        │
┌──────────────────────────────────────────────────────────────────────┐
│                        PROCESSING LAYER (user components)            │
│                                                                      │
│  User-defined components implement ComponentProtocol[TIn, TOut]      │
│  Framework wires them by matching type annotations.                  │
│                                                                      │
│  ParallelDecoderPool → ImageDecoder → ImagePreprocessor              │
└──────────────────────────────┬───────────────────────────────────────┘
                               │ InferenceRequest
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│                        INFERENCE LAYER (inference/)                   │
│                                                                      │
│  TritonInferenceClient    ModelVersionTracker    RetryPolicy         │
│  (gRPC async/futures)     (rolling release)      (3x exp backoff)   │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
┌──────────────────────────────▼───────────────────────────────────────┐
│                        METRICS LAYER (metrics/)                      │
│                                                                      │
│  MetricBuffer → MetricEmitter → Redis Stream "metrics"               │
│  TritonMetricsCollector (polls Triton stats)                         │
│  ErrorCollector (interval-based error counts)                        │
│  TimeWindowAggregator (business metric aggregations)                 │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 4. Data Flow

### 4.1 Happy Path: Image Event through Inference

```
Step 1: Redis Stream "camera_images"
        XREADGROUP GROUP app_group consumer_1 COUNT 10 BLOCK 1000

Step 2: RedisStreamInputConnector (Thread-1)
        - Decodes Redis fields into pydantic model (CameraImageEvent)
        - Wraps in Envelope[CameraImageEvent] with receive timestamp
        - Puts onto Queue[Envelope[CameraImageEvent]]
        - Does NOT XACK yet (tracked in pending set)

Step 3: QueueBus delivers to registered Component (main thread)
        - Registry resolved: CameraImageEvent → UserPreprocessComponent
        - component.process(envelope) called

Step 4: UserPreprocessComponent (user code, implements ComponentProtocol)
        - Extracts image bytes from event
        - Submits to ParallelDecoderPool:
          JPEG → TurboJPEGDecoder | AVIF → CV2AvifDecoder
        - Creates InferenceRequest(raw_image=ndarray, model_name="yolo26s")

Step 5: Inference Layer
        - ModelVersionTracker resolves latest ready version
        - Builds tritonclient InferInput from ndarray
        - Calls async_infer() → returns Future[InferenceResult]
        - RetryPolicy: 3 attempts, exponential backoff (0.1s, 0.4s, 1.6s)
        - On all retries exhausted: InferenceError → error queue

Step 6: Future callback fires on completion
        - Wraps result in InferenceResult
        - Routes to user ResultHandler or directly to output queue

Step 7: RedisStreamOutputConnector (Thread-2)
        - Gets result from output queue
        - Serializes to Redis fields
        - XADD to output stream "detection_results"

Step 8: Acknowledgment
        - After successful output write, XACK on input stream
        - Envelope.track_state("acknowledged") records final timestamp
        - Transit time metric emitted
```

### 4.2 Newest-First Recovery (after restart)

```
1. Startup: XGROUP CREATE stream group $ MKSTREAM (no-op if exists)

2. Read pending (unacked) messages:
   XREADGROUP GROUP group consumer STREAMS stream 0
   Sort by message ID DESCENDING (newest first)
   Feed into pending_queue (PriorityQueue, newest = highest priority)

3. Start live reading: XREADGROUP ... STREAMS stream >
   Feed into live_queue (regular Queue)

4. Main thread reader:
   - Always try live_queue first (non-blocking get)
   - If empty, try pending_queue (non-blocking get, newest first)
   - If both empty, block on live_queue with short timeout
   → Live data is never starved by recovery backlog
```

---

## 5. Configuration

### 5.1 Hydra Directory Layout

```
conf/
  config.yaml                     # Main config with defaults list
  streams/
    input/
      camera_images.yaml          # Input stream definition
    output/
      detection_results.yaml      # Output stream definition
      metrics.yaml                # Metrics output stream
  inference/
    default.yaml                  # Triton connection + retry config
  preprocessing/
    default.yaml                  # Decoder pool config
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


class StreamInputConfig(BaseModel):
    stream_name: str
    consumer_group: str
    consumer_name: str = ""           # auto-generated if empty
    event_model: str                  # fully-qualified pydantic model class path
    batch_size: int = 10
    block_ms: int = 1000
    newest_first_on_recovery: bool = True


class StreamOutputConfig(BaseModel):
    stream_name: str
    maxlen: int | None = 10_000       # XADD MAXLEN ~ value
    approximate_maxlen: bool = True


class InferenceConfig(BaseModel):
    grpc_url: str = "localhost:8001"  # env: TRITON_GRPC_URL
    max_retry_attempts: int = 3
    retry_base_delay_s: float = 0.1
    retry_backoff_multiplier: float = 4.0
    max_concurrent_inferences: int = 16
    model_version_poll_interval_s: float = 5.0


class PreprocessingConfig(BaseModel):
    decoder_pool_type: str = "thread"  # "thread" | "process"
    decoder_pool_workers: int = 4
    default_target_size: tuple[int, int] = (640, 640)


class MetricsConfig(BaseModel):
    output_stream: str = "metrics"
    triton_poll_interval_s: float = 10.0
    error_flush_interval_s: float = 30.0
    transit_time_enabled: bool = True


class AppConfig(BaseModel):
    app_name: str = "triton-connector"  # env: APP_NAME
    redis: RedisConnectionConfig = RedisConnectionConfig()
    input_streams: list[StreamInputConfig] = []
    output_streams: list[StreamOutputConfig] = []
    inference: InferenceConfig = InferenceConfig()
    preprocessing: PreprocessingConfig = PreprocessingConfig()
    metrics: MetricsConfig = MetricsConfig()
    components: list[str] = []          # fully-qualified class paths
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

input_streams:
  - stream_name: camera_images
    consumer_group: inference_group
    event_model: myapp.models.CameraImageEvent
    batch_size: 10
    block_ms: 1000
    newest_first_on_recovery: true

output_streams:
  - stream_name: detection_results
    maxlen: 10000
  - stream_name: metrics
    maxlen: 50000

components:
  - myapp.components.CameraToInference
  - myapp.components.DetectionResultHandler
```

---

## 6. Protocols & Extension Points

All extension points use `typing.Protocol` with `@runtime_checkable`. Users implement
these interfaces without inheriting from framework classes (structural subtyping).

### 6.1 Connector Protocols (`connectors/base.py`)

```python
@runtime_checkable
class InputConnectorProtocol(Protocol):
    """Protocol for input connectors (Redis Streams, future Kafka, etc.)."""

    def start(self) -> None: ...
    def stop(self) -> None: ...
    @property
    def output_queue(self) -> queue.Queue[Envelope[Any]]: ...
    @property
    def error_queue(self) -> queue.Queue[ConnectorError]: ...


@runtime_checkable
class OutputConnectorProtocol(Protocol):
    """Protocol for output connectors."""

    def start(self) -> None: ...
    def stop(self) -> None: ...
    @property
    def input_queue(self) -> queue.Queue[dict[str, str | bytes]]: ...
```

### 6.2 Component Protocol (`protocols.py`)

```python
@runtime_checkable
class ComponentProtocol[TInput, TOutput](Protocol):
    """Base protocol for processing components.

    Framework inspects TInput/TOutput type parameters to wire
    components together automatically via type annotation parsing.
    """

    def process(self, envelope: Envelope[TInput]) -> TOutput: ...
```

### 6.3 Image Protocols (`preprocessing/protocols.py`)

```python
@runtime_checkable
class ImageDecoderProtocol(Protocol):
    """Protocol for image decoders. Must be thread-safe."""

    @property
    def supported_content_types(self) -> frozenset[str]: ...

    def decode(self, raw_bytes: bytes) -> npt.NDArray[np.uint8]: ...


@runtime_checkable
class ImagePreprocessorProtocol(Protocol):
    """Protocol for image preprocessing (resize, normalize, pad)."""

    def to_raw(self, image: npt.NDArray[np.uint8]) -> npt.NDArray[np.float32]: ...

    def to_bytes(self, tensor: npt.NDArray[np.float32]) -> bytes: ...
```

### 6.4 Inference Protocols (`protocols.py`)

```python
@runtime_checkable
class InferenceRequestBuilderProtocol[TInput](Protocol):
    """Builds InferenceRequest from input events — primary extension point."""

    def build_request(self, envelope: Envelope[TInput]) -> InferenceRequest: ...


@runtime_checkable
class ResultHandlerProtocol[TResult](Protocol):
    """Handles inference results before output."""

    def handle_result(
        self, result: InferenceResult, original_envelope: Envelope[TResult],
    ) -> dict[str, str | bytes]: ...
```

### 6.5 Observer Protocol (`protocols.py`)

```python
@runtime_checkable
class ObserverProtocol[TEvent](Protocol):
    """Observes events without producing output (metrics, logging, side-effects)."""

    def observe(self, envelope: Envelope[TEvent]) -> None: ...
```

### 6.6 Aggregation Rule Protocol (`protocols.py`)

```python
@runtime_checkable
class AggregationRuleProtocol[TEvent](Protocol):
    """Extracts grouping key from events for time-windowed aggregation."""

    @property
    def metric_name(self) -> str: ...

    @property
    def window_seconds(self) -> int: ...

    def extract_key(self, event: TEvent) -> str: ...

    def aggregate(self, count: int) -> float: ...
```

---

## 7. Connector Layer

### 7.1 RedisStreamInputConnector

Runs in a dedicated thread. Performs blocking `XREADGROUP` calls.

```python
class RedisStreamInputConnector:
    """Reads from a Redis Stream using consumer group semantics.

    Runs in a dedicated thread. Decodes fields to pydantic models.
    Puts Envelope[T] instances onto the output queue.
    Errors go to the error queue — connector never crashes.
    """

    def __init__(
        self,
        config: StreamInputConfig,
        redis_config: RedisConnectionConfig,
        event_model_class: type[BaseModel],
    ) -> None: ...

    def start(self) -> None:
        """Start the connector thread."""

    def stop(self) -> None:
        """Signal the thread to stop and join."""

    def _run_loop(self) -> None:
        """Main loop running in dedicated thread.

        while not stop_event:
            try:
                messages = redis.xreadgroup(...)
                for message_id, fields in messages:
                    try:
                        event = event_model_class.model_validate(fields)
                        envelope = Envelope(payload=event, ...)
                        output_queue.put(envelope)
                    except ValidationError:
                        error_queue.put(ConnectorError(...))
            except redis.ConnectionError:
                error_queue.put(ConnectorError(...))
                stop_event.wait(timeout=1.0)  # backoff
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

    def read_pending_newest_first(self) -> list[tuple[str, dict[str, str]]]:
        """XREADGROUP with id=0, sort by message ID descending."""

    def read_new(self) -> list[tuple[str, dict[str, str]]]:
        """XREADGROUP with id='>' (new messages, blocking)."""

    def acknowledge(self, message_id: str) -> None:
        """XACK a processed message."""
```

### 7.3 RedisStreamOutputConnector

Runs in a dedicated thread. Reads from an input queue and writes via `XADD`.

```python
class RedisStreamOutputConnector:
    """Writes events to a Redis Stream.

    Runs in a dedicated thread. Reads dict[str, str | bytes] from queue.
    On successful XADD, triggers acknowledgment callback.
    """

    def _run_loop(self) -> None:
        """Main loop: get from queue, XADD to stream, trigger ack callback."""
```

---

## 8. Inference Layer

### 8.1 Minimized Async Approach

The inference module is the **only** part of the system that uses async operations.
Triton's gRPC client supports an async mode where `infer()` returns a `Future`.
The rest of the application is synchronous/threaded.

```python
class TritonInferenceClient:
    """Triton gRPC client using async_infer with Future callbacks.

    The client is called from the main thread. async_infer() returns
    immediately with an InferenceServerClient.InferResult future.
    A callback fires on completion, putting the result onto a queue.
    """

    def __init__(self, config: InferenceConfig) -> None:
        self._client = tritonclient.grpc.InferenceServerClient(url=config.grpc_url)
        self._semaphore = threading.Semaphore(config.max_concurrent_inferences)
        self._retry_policy = RetryPolicy(
            max_attempts=config.max_retry_attempts,
            base_delay=config.retry_base_delay_s,
            multiplier=config.retry_backoff_multiplier,
        )

    def infer(
        self,
        request: InferenceRequest,
        callback: Callable[[InferenceResult | InferenceError], None],
    ) -> None:
        """Submit inference request. Result delivered via callback.

        Uses tritonclient async_infer which returns a Future.
        Semaphore limits concurrent in-flight requests.
        Retry logic wraps the submission.
        """
        self._semaphore.acquire()

        def _on_complete(result: Any, error: Any) -> None:
            self._semaphore.release()
            if error is not None:
                callback(InferenceError(...))
            else:
                callback(InferenceResult(...))

        self._client.async_infer(
            model_name=request.model_name,
            model_version=request.model_version,
            inputs=self._build_inputs(request),
            callback=_on_complete,
        )
```

### 8.2 InferenceRequest / InferenceResult (`inference/request.py`)

```python
class InferenceRequest(BaseModel, arbitrary_types_allowed=True):
    model_name: str
    model_version: str = ""           # empty = latest
    raw_image: npt.NDArray[np.float32]
    input_name: str = "images"
    output_names: tuple[str, ...] = ("output0",)


class InferenceResult(BaseModel, arbitrary_types_allowed=True):
    model_name: str
    model_version: str
    outputs: dict[str, npt.NDArray[np.float32]]
    inference_time_ms: float
```

### 8.3 Retry Policy (`inference/retry.py`)

```python
class RetryPolicy:
    """Exponential backoff: 3 attempts with delays 0.1s, 0.4s, 1.6s.

    On final failure, returns the error rather than retrying.
    Uses time.sleep() since this runs in the sync context.
    """

    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 0.1,
        multiplier: float = 4.0,
    ) -> None: ...

    def execute_with_callback(
        self,
        func: Callable[..., None],
        *args: Any,
        on_failure: Callable[[Exception], None],
    ) -> None:
        """Execute func with retry. Calls on_failure after all attempts exhausted."""
```

### 8.4 Rolling Release — ModelVersionTracker (`inference/model_registry.py`)

Triton uses `model-control-mode=poll` with 10s interval. When a new model version
appears in the repository, Triton loads it automatically.

```python
class ModelVersionTracker:
    """Tracks model versions and provides safe version resolution.

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
        """Periodic task (runs in a timer thread):
        1. Query get_model_repository_index()
        2. For each model, check is_model_ready(model, new_version)
        3. If ready, update current version mapping
        4. Old version usable until in-flight count drops to zero
        5. Emit metric: model_version_switch
        """
```

**Version switch sequence:**

```
T=0:   Model "yolo26s" v1 active, 5 in-flight requests
T=1:   New v2 uploaded to model_repository/yolo26s/2/
T=10:  Triton polls, starts loading v2
T=15:  _poll_versions detects v2, is_model_ready("yolo26s", "2") → True
       _current_versions["yolo26s"] = "2"
       New requests → v2, in-flight v1 requests continue
T=17:  Last v1 request completes, in_flight_counts[("yolo26s","1")] == 0
       v1 decommissioned (Triton manages lifecycle)
```

**Context manager for safe inference:**

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

### 9.1 Decoder Implementations

**TurboJPEGDecoder** — uses PyTurboJPEG for high-throughput JPEG decoding.
Thread-safe: each call uses a thread-local TurboJPEG instance.

```python
class TurboJPEGDecoder:
    supported_content_types = frozenset({"image/jpeg", "image/jpg"})

    def decode(self, raw_bytes: bytes) -> npt.NDArray[np.uint8]:
        jpeg = self._get_thread_local()
        return jpeg.decode(raw_bytes)  # returns BGR numpy array
```

**CV2AvifDecoder** — uses opencv-python-headless for AVIF and general format decoding.

```python
class CV2AvifDecoder:
    supported_content_types = frozenset({"image/avif", "image/png", "image/bmp"})

    def decode(self, raw_bytes: bytes) -> npt.NDArray[np.uint8]:
        buf = np.frombuffer(raw_bytes, dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
```

### 9.2 DecoderDispatcher

Routes images to the correct decoder based on content type or magic bytes.

```python
class DecoderDispatcher:
    MAGIC_BYTES: ClassVar[dict[bytes, str]] = {
        b"\xff\xd8\xff": "image/jpeg",
        b"\x00\x00\x00": "image/avif",  # ftyp box detection
        b"\x89PNG": "image/png",
    }

    def __init__(self, decoders: Sequence[ImageDecoderProtocol]) -> None:
        """Build registry: content_type → decoder."""

    def detect_content_type(self, raw_bytes: bytes) -> str: ...

    def dispatch(self, raw_bytes: bytes, content_type: str | None = None) -> ImageDecoderProtocol: ...
```

### 9.3 ParallelDecoderPool

Wraps `concurrent.futures` executors for parallel image decoding.
ThreadPoolExecutor is default (turbojpeg releases GIL in C code).

```python
class ParallelDecoderPool:
    def __init__(self, config: PreprocessingConfig, dispatcher: DecoderDispatcher) -> None:
        self._executor = ThreadPoolExecutor(max_workers=config.decoder_pool_workers)
        self._dispatcher = dispatcher

    def decode(self, raw_bytes: bytes, content_type: str | None = None) -> Future[npt.NDArray[np.uint8]]:
        """Submit decode to thread pool, return Future."""
        decoder = self._dispatcher.dispatch(raw_bytes, content_type)
        return self._executor.submit(decoder.decode, raw_bytes)

    def decode_batch(
        self,
        items: Sequence[bytes],
        content_types: Sequence[str | None] | None = None,
    ) -> list[Future[npt.NDArray[np.uint8]]]:
        """Submit multiple decodes, return list of Futures."""
```

### 9.4 DALI Ensemble Alternative

For YOLO models with DALI preprocessing, decoding happens server-side in Triton.
The `ImagePreprocessorProtocol.to_bytes()` passes raw JPEG bytes through, and
Triton's DALI backend handles decode + preprocess. The `InferenceRequest.raw_image`
field carries raw bytes as a 1-D uint8 array.

---

## 10. Metrics

### 10.1 MetricEvent Model

```python
class MetricEvent(BaseModel):
    component_name: str               # from APP_NAME env var
    measure: str                      # metric name (e.g., "inference_latency_ms")
    value: float                      # metric value
    timestamp_ms: int                 # unix epoch ms
```

### 10.2 Metrics Flow

```
Event transit tracking ──┐
                          │
Triton stats polling ─────┤   MetricEvent
                          ├──────────────► MetricBuffer ──► MetricEmitter ──► Redis "metrics"
Error aggregation ────────┤
                          │
Business aggregation ─────┘
  (TimeWindowAggregator)
```

### 10.3 Transit Time Tracking

Each `Envelope` records timestamps at key states:

| State | When |
|---|---|
| `received` | Connector reads from Redis |
| `decoded` | Pydantic model deserialized |
| `queued` | Placed on processing queue |
| `processing_start` | Component begins processing |
| `inference_submitted` | Triton request sent |
| `inference_complete` | Triton response received |
| `output_sent` | Result written to output stream |
| `acknowledged` | XACK sent |

Auto-emitted metrics: `event_total_transit_ms`, `event_queue_wait_ms`, `event_inference_ms`.

### 10.4 TritonMetricsCollector

Periodic task (runs in a timer thread). Polls Triton stats via gRPC
`get_inference_statistics()`:

- `triton_inference_count` — cumulative inference count per model
- `triton_inference_queue_time_us` — cumulative queue time
- `triton_inference_compute_infer_time_us` — cumulative inference compute time
- `triton_inference_compute_input_time_us` — cumulative input compute time
- `triton_inference_compute_output_time_us` — cumulative output compute time

### 10.5 MetricBuffer & MetricEmitter

```python
class MetricBuffer:
    """Thread-safe buffer accumulating MetricEvent instances.

    Multiple producers (transit tracker, Triton collector, error collector)
    write concurrently. MetricEmitter drains periodically.
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

## 11. Aggregation Metrics Library (`triton_metrics_aggregator`)

### 11.1 Purpose

Analyze incoming events with pydantic validators and send aggregation results as
metrics. Example: input event has `camera` attribute → aggregate by camera value per
minute → emit "images_per_camera_per_minute" metric.

### 11.2 Core Types

```python
class AggregationRule(BaseModel):
    metric_name: str                  # e.g., "images_per_camera_per_minute"
    field_path: str                   # dot-separated path, e.g., "camera"
    window_seconds: int = 60
    aggregation: str = "count"        # "count" | "sum" | "avg"


class TimeWindowAggregator[TEvent: BaseModel]:
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
        # → [MetricEvent(measure="images_per_camera_per_minute.cam_01", value=42)]
    """

    def __init__(self, rules: Sequence[AggregationRule]) -> None:
        self._validate_field_paths()    # fail fast

    def observe(self, event: TEvent) -> None: ...

    def flush(self) -> list[MetricEvent]: ...
```

### 11.3 Integration

`TimeWindowAggregator` integrates as an `ObserverProtocol` component.
The framework feeds it events without expecting output routing.
Flushed metrics go to the shared `MetricBuffer`.

---

## 12. Error Handling

### 12.1 Error Model Hierarchy

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

### 12.2 Error Flow

```
Connector Threads ──► error_queue ──┐
                                     │
Preprocessing Pool ──► error_queue ──┤
                                     │
Inference callbacks ──► error_queue ─┤
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

### 12.3 ErrorCollector

```python
class ErrorCollector:
    """Collects errors by interval, emits aggregated metric counts.

    _buckets: dict[error_type, count] within current window.
    Flushed every error_flush_interval_s (default 30s).
    """

    def record(self, error_type: str) -> None:
        """Increment counter for error type in current window."""

    def flush(self) -> list[MetricEvent]:
        """Drain buckets → MetricEvent per type with measure="error_count_{type}"."""
```

### 12.4 Connector Resilience

Connectors catch specific exception types and route errors to the error queue.
They **never crash**:

- `redis.ConnectionError` → error_queue + backoff sleep
- `redis.TimeoutError` → normal for BLOCK reads, continue
- `pydantic.ValidationError` → error_queue, skip message, continue
- Any other specific exception → error_queue, continue

---

## 13. Recovery & Position Tracking

### 13.1 Consumer Group Semantics

Redis consumer groups provide built-in position tracking:

- `XGROUP CREATE ... $` — start reading from new messages only
- `XREADGROUP ... >` — read only new (undelivered) messages
- `XREADGROUP ... 0` — read pending (delivered but unacked) messages
- `XACK` — acknowledge successful processing

Messages remain pending until acknowledged. On restart, pending messages
are redelivered automatically.

### 13.2 At-Least-Once Delivery

XACK is sent **only after** the result is successfully written to the output stream.
If the app crashes between processing and output write, the message remains pending
and will be reprocessed on restart. Trade-off: potential duplicate processing,
acceptable for inference workloads (idempotent results).

### 13.3 Newest-First Priority

On restart recovery, pending messages are sorted by Redis message ID descending
(newest first). Two separate queues:

- `pending_queue` (PriorityQueue) — drained with newest-first ordering
- `live_queue` (Queue) — normal FIFO for new messages

Reader logic:

```python
def _combined_reader(live_queue, pending_queue, handler, stop_event):
    while not stop_event.is_set():
        # 1. Always try live_queue first (non-blocking)
        try:
            envelope = live_queue.get_nowait()
            handler(envelope)
            continue
        except queue.Empty:
            pass
        # 2. Fall back to pending (newest-first)
        try:
            envelope = pending_queue.get_nowait()
            handler(envelope)
            continue
        except queue.Empty:
            pass
        # 3. Both empty — block on live briefly
        try:
            envelope = live_queue.get(timeout=0.1)
            handler(envelope)
        except queue.Empty:
            continue
```

---

## 14. Threading Model

### 14.1 Thread Map

```
┌──────────────────────────────────────────────────────────────┐
│  Main Thread                                                  │
│  - Application orchestrator                                   │
│  - Queue reader (drains live + pending queues)                │
│  - Component dispatch (calls user process() methods)          │
│  - Submits inference via async_infer (returns immediately)    │
│  - Handles inference callbacks (put results on output queue)  │
│  - ErrorHandler (drains error queue)                          │
└──────────────────────────────────────────────────────────────┘

┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│ Thread: Input-1   │  │ Thread: Input-2   │  │ Thread: Output-1  │
│ XREADGROUP        │  │ XREADGROUP        │  │ XADD              │
│ stream_a          │  │ stream_b          │  │ results_stream    │
│ → live_queue      │  │ → live_queue      │  │ ← output_queue    │
└──────────────────┘  └──────────────────┘  └──────────────────┘

┌──────────────────┐  ┌──────────────────┐
│ Thread: Output-2  │  │ Timer: Metrics    │
│ XADD              │  │ Periodic tasks:   │
│ metrics_stream    │  │ - Triton stats    │
│ ← metrics_queue   │  │ - Error flush     │
└──────────────────┘  │ - Version poll    │
                      │ - Metric emit     │
                      └──────────────────┘

┌──────────────────────────────────────────┐
│ ThreadPoolExecutor (decoder pool)         │
│ Worker-1: turbojpeg.decode()              │
│ Worker-2: turbojpeg.decode()              │
│ Worker-3: cv2.imdecode()                  │
│ Worker-4: turbojpeg.decode()              │
└──────────────────────────────────────────┘
```

### 14.2 Queue Topology

| Queue | Type | Producer | Consumer |
|---|---|---|---|
| `live_queue[stream]` | `Queue[Envelope[T]]` | Input connector thread | Main thread |
| `pending_queue[stream]` | `PriorityQueue[Envelope[T]]` | Startup recovery | Main thread |
| `output_queue[stream]` | `Queue[dict[str, str\|bytes]]` | Main thread | Output connector thread |
| `error_queue` | `Queue[ConnectorError\|InferenceError\|PreprocessingError]` | All threads | Main thread ErrorHandler |
| `metric_buffer` | `MetricBuffer` (thread-safe list) | All metric producers | MetricEmitter timer |

### 14.3 Inference Concurrency

`tritonclient.grpc.InferenceServerClient.async_infer()` is non-blocking. It accepts
a callback function and returns immediately. The callback fires in a gRPC internal
thread when the response arrives. The callback puts the result onto the output queue
(thread-safe). A `threading.Semaphore` limits concurrent in-flight requests.

---

## 15. Envelope Type (`engine/event.py`)

```python
@dataclass(slots=True)
class Envelope[T]:
    payload: T
    stream_name: str
    message_id: str
    priority: int                     # for PriorityQueue ordering
    timestamps: dict[str, int] = field(default_factory=dict)

    def track_state(self, state: str) -> None:
        self.timestamps[state] = time.monotonic_ns()

    def transit_time_ms(self, from_state: str, to_state: str) -> float:
        return (self.timestamps[to_state] - self.timestamps[from_state]) / 1_000_000

    def __lt__(self, other: Envelope[Any]) -> bool:
        return self.priority < other.priority
```

Priority calculation for newest-first:

```python
def message_id_to_priority(message_id: str, is_pending: bool) -> int:
    if not is_pending:
        return 0                      # live messages — highest priority
    timestamp_str = message_id.split("-")[0]
    return -int(timestamp_str)        # newer pending = smaller negative = higher priority
```

---

## 16. Component Registry & Type-Annotation Wiring

### 16.1 How It Works

When a user registers a component class, the registry:

1. Calls `typing.get_type_hints()` on the `process` method
2. Extracts `Envelope[X]` parameter → gets `X` via `typing.get_args()`
3. Extracts return type `Y`
4. Builds directed graph: `X → Component → Y`
5. Connects input connectors to components whose input type matches `event_model`
6. Connects components whose output type is `dict[str, str | bytes]` to output connectors
7. Recognizes `InferenceRequest` output as the inference layer entry point

```python
class ComponentRegistry:
    def register(self, component_class: type) -> None:
        hints = typing.get_type_hints(component_class.process)
        envelope_type = hints["envelope"]           # Envelope[SomeInput]
        input_type = typing.get_args(envelope_type)[0]
        output_type = hints["return"]
        self._graph.add_node(component_class, input_type, output_type)

    def build_graph(self) -> ProcessingGraph:
        """Wire components. Raises ValueError on missing sources, cycles, or dangling outputs."""
```

### 16.2 Wiring Example

```
StreamInputConfig(event_model=CameraImageEvent)
    │
    │  type match: CameraImageEvent
    ▼
CameraToInference[CameraImageEvent → InferenceRequest]
    │
    │  type match: InferenceRequest → built-in inference handler
    ▼
TritonInferenceClient → callback → InferenceResult
    │
    │  type match: InferenceResult
    ▼
DetectionResultHandler[InferenceResult → dict[str, str | bytes]]
    │
    │  type match: dict → output connector
    ▼
StreamOutputConfig(stream_name="detection_results")
```

### 16.3 Example User Code

```python
class CameraImageEvent(BaseModel):
    camera: str
    image_data: bytes
    timestamp: int


class CameraToInference:
    """Implements ComponentProtocol[CameraImageEvent, InferenceRequest]."""

    def __init__(self, decoder_pool: ParallelDecoderPool) -> None:
        self._decoder_pool = decoder_pool

    def process(self, envelope: Envelope[CameraImageEvent]) -> InferenceRequest:
        future = self._decoder_pool.decode(envelope.payload.image_data)
        decoded = future.result()     # block until decoded
        raw = decoded.astype(np.float32) / 255.0
        return InferenceRequest(model_name="yolo26s", raw_image=raw)
```

---

## 17. Application Bootstrap

```python
class Application:
    @classmethod
    def from_config(cls, config: AppConfig) -> Application: ...

    def run(self) -> None:
        """Main run sequence:
        1. Create Redis connections
        2. Create Triton client
        3. Import and register user components (from config.components)
        4. Build processing graph via ComponentRegistry
        5. Create consumer groups for input streams
        6. Load pending messages (newest-first) into pending queues
        7. Start connector threads (input + output)
        8. Start timer threads:
           - TritonMetricsCollector (periodic)
           - ErrorCollector flush (periodic)
           - MetricEmitter (periodic)
           - ModelVersionTracker (periodic)
        9. Run main loop:
           - Drain live + pending queues
           - Dispatch to components
           - Submit inference (async_infer with callback)
           - Handle errors
        10. On shutdown signal (SIGINT/SIGTERM):
            - Signal connector threads to stop
            - Drain remaining queue items
            - Wait for in-flight inferences
            - Flush final metrics
            - Close connections
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

## 18. New Dependencies

### triton_redis_connector/pyproject.toml additions

```
hydra-core>=1.3.0
PyTurboJPEG>=1.7.0
opencv-python-headless>=4.9.0
```

### New package: triton_shared_models/pyproject.toml

```
pydantic>=2.10.0
loguru>=0.7.3
```

### New package: triton_metrics_aggregator/pyproject.toml

```
pydantic>=2.10.0
loguru>=0.7.3
triton-redis-connector  (workspace reference)
```

---

## 19. Design Decisions Summary

| Decision | Choice | Rationale |
|---|---|---|
| Config system | Hydra | YAML composition, env interpolation, config groups |
| Shared types | Extracted package | Clean separation, no circular deps |
| Kafka support | Abstraction only | Protocol-based connectors, implement when needed |
| Async scope | Triton only | Minimize complexity; futures-based async_infer |
| Threading model | Threads for I/O, pools for CPU | Redis blocking reads need threads; image decode is CPU-bound |
| ACK timing | After output write | At-least-once delivery guarantee |
| Recovery order | Newest-first | Fresher data is more valuable after downtime |
| Extension model | Protocol classes | Structural subtyping, no inheritance required |
| Retry policy | 3x exponential backoff | 0.1s, 0.4s, 1.6s — fast initial retry, reasonable total wait |
| Model rollout | Poll + in-flight tracking | Safe switch with zero dropped requests |

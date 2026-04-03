# Architecture

## Package Structure

The monorepo contains three packages with a strict dependency direction:

```
triton_model_tools  ->  triton_redis_connector  ->  triton_shared_models (leaf)
```

| Package | Path | Role |
|---|---|---|
| `triton-shared-models` | `packages/triton_shared_models/` | Leaf: Triton types (`TritonModelConfig`, `TritonDataType`, `ExportFormat`), letterbox preprocessing, NMS |
| `triton-redis-connector` | `packages/triton_redis_connector/` | Core: graph-based pipeline engine, Redis Stream connectors, Triton inference, metrics |
| `triton-model-tools` | `packages/triton_model_tools/` | CLI: model conversion (ONNX/TensorRT), batch inference, benchmarking, E2E testing |

## Import-Linter Contracts

Four contracts enforced via `uv run lint-imports` (defined in root `pyproject.toml`):

### 1. connector-layers (type: layers)

Internal layer hierarchy of `triton_redis_connector`:

```
connectors | inference | preprocessing | metrics                      <- domain siblings
engine                                                                <- orchestration
config | errors                                                       <- shared
protocols                                                             <- leaf
```

Modules within a `|` group are siblings and must not cross-import. One allowed violation: `engine.wrappers -> metrics.collector` (lazy import for MetricEvent).

### 2. connector-no-model-tools (type: forbidden)

`triton_redis_connector` must not import `triton_model_tools`. The reverse is allowed.

### 3. shared-models-independence (type: forbidden)

`triton_shared_models` must not import either `triton_redis_connector` or `triton_model_tools`.

### 4. model-tools-layers (type: layers)

Internal layer hierarchy of `triton_model_tools`:

```
__main__  ->  cli  ->  benchmarks  ->  testing  ->  inference | export  ->  models (leaf)
```

## Execution Variant Model

Three primitives, each with parallel execution mode variants (6 total). All protocols defined in `protocols.py`. There are no inline variants -- all nodes run in dedicated threads managed by the Application main loop.

### Source (produces events)

| Variant | Protocol | Execution |
|---|---|---|
| `ThreadSource` | `ThreadSource[TOut]` | Dedicated thread, has `stop: Event` |
| `ProcessSource` | `ProcessSource[TOut]` | Dedicated OS process, has `stop: Event` |

### Processor (transforms events)

| Variant | Protocol | Execution |
|---|---|---|
| `ThreadProcessor` | `ThreadProcessor[TIn, TOut]` | Dedicated thread |
| `ThreadPoolProcessor` | `ThreadPoolProcessor[TIn, TOut]` | N threads, shared queue |
| `ProcessPoolProcessor` | `ProcessPoolProcessor[TIn, TOut]` | N processes, shared queue |

### Sink (consumes events)

| Variant | Protocol | Execution |
|---|---|---|
| `ThreadSink` | `ThreadSink[TIn]` | Dedicated thread, has `stop: Event` |

User code has two options for declaring the execution variant:

1. **Inherit from base classes** (preferred for user processors) -- the variant is declared by the class itself, no wrapping needed. See "Base Classes for User Processors" below.
2. **Write plain callables** and wrap at assembly time using wrappers from `engine/wrappers.py`: `DetachedThreadSource`, `DetachedProcessSource`, `DetachedThreadProcessor`, `ThreadedProcessor`, `MultiProcessProcessor`, `DetachedThreadSink`. Use this for framework building blocks that should remain variant-agnostic.

See: `protocols.py` (base classes), `engine/wrappers.py`, `engine/registry.py`

### Base Classes for User Processors

User-defined processors inherit from base classes that declare the execution variant explicitly. Six base classes in `protocols.py` correspond to the six parallel execution variants:

| Base Class | Variant | Has `stop` |
|---|---|---|
| `BaseThreadSource[TOut]` | Threaded source | Yes |
| `BaseProcessSource[TOut]` | Process source | Yes |
| `BaseThreadProcessor[TIn, TOut]` | Dedicated thread | Yes |
| `BaseThreadPoolProcessor[TIn, TOut]` | Thread pool | Yes |
| `BaseProcessPoolProcessor[TIn, TOut]` | Process pool | Yes |
| `BaseThreadSink[TIn]` | Threaded sink | Yes |

Base classes satisfy the `WrappedNode` protocol (fast-path variant detection in registry). User code:

```python
from triton_redis_connector import BaseThreadProcessor

class MyProcessor(BaseThreadProcessor[ImageEvent, Result]):
    def __init__(self, url: str) -> None:
        super().__init__()
        self._url = url

    def __call__(self, event: ImageEvent) -> Result | None:
        return process(self._url, event)

registry.register("proc", MyProcessor("triton:8001"))
```

### Application.build() Classmethod

`engine/app.py` provides `Application.build(config, processors, *, triton_collector)` -- a classmethod that creates a fully wired Application from an explicit processor list.

```python
from triton_redis_connector import Application, ModelBundle, expand_bundle

bundle = ModelBundle(
    name="yolo",
    preprocessor=ClientPreprocessorNode(...),
    inference=InferenceNode(grpc_url="triton:8001"),
    postprocessor=PostProcessorNode(postprocessor=my_postprocessor),
    sink=LogErrorSinkNode(),
)

app = Application.build(
    config,
    processors=[redis_source, *expand_bundle(bundle), redis_sink],
)
app.run()
```

What `Application.build` does:
- Creates `PipelineRegistry` and registers each processor by its `name` attribute or class name
- Auto-creates `MetricBuffer`, `MetricEmitter`, `QueueBus` from config
- Runs graph validation before returning

### build_application Factory

Models are defined exclusively via `ModelBundle`. The `expand_bundle()` function unpacks a bundle into the processor list for `Application.build()`.

## Shipped Processors

Two generations of shipped processors coexist. Legacy processors are plain callables wrapped at assembly time. New Node classes inherit from Base* classes and use the `PipelineEvent` composition pattern.

### Legacy Shipped Processors (variant-agnostic)

| Processor | Location | Input | Output | Purpose |
|---|---|---|---|---|
| `DaliTritonRequestBuilder` | connector | `_ImageBytesEvent` or `_ImagePathEvent` | `TritonRequest` | Wrap raw image bytes for DALI ensemble inference (reads from path if needed) |
| `ClientPreprocessor` | connector | `_ImagePathEvent` | `TritonRequest` | Read file, decode, letterbox, wrap as tensor |
| `EventSerializer[T]` | connector | `T` (pydantic) | `Envelope[bytes]` | Serialize event to JSON bytes with original_fields passthrough |
| `LogErrorSink` | connector | `BaseModel` | `None` | Log error events at error level via loguru |
| `YoloPostProcessorBridge` | model_tools | `InferenceOutput` | `DetectionResult` | YOLO postprocessing (NMS + confidence filter), config-friendly constructor |

### New Node Classes (PipelineEvent-aware)

These inherit from Base* classes for variant declaration and use the `PipelineEvent[TCtx, TPayload]` composition pattern. Designed for `Application.build()` and `ModelBundle` usage.

| Node | Location | Input | Output | Purpose |
|---|---|---|---|---|
| `ClientPreprocessorNode[TCtx, TPayload]` | connector | `PipelineEvent[TCtx, ImagePayload]` | `PipelineEvent[TCtx, TritonRequest]` or `PreprocessingError` | Decode + letterbox from ImagePayload |
| `InferenceNode[TCtx, TReq, TOut]` | connector | `PipelineEvent[TCtx, TritonRequest]` | `PipelineEvent[TCtx, InferenceOutput]` or `InferenceError` | Send TritonRequest to Triton via gRPC |
| `PostProcessorNode[TCtx, TIn, TResult]` | connector | `PipelineEvent[TCtx, InferenceOutput]` | `PipelineEvent[TCtx, TResult]` | Delegate to user postprocessor, wrap result |
| `LogErrorSinkNode` | connector | `InferenceError` or `ConnectorError` or `PreprocessingError` | `None` | Log error events on dedicated thread |

## Domain Event Model

All domain events inherit from `Event` (in `engine/events.py`), which carries framework metadata:

- `Event` -- base class with `fw: FrameworkMeta` (source_stream, message_id, priority)
- `PipelineEvent[TCtx, TPayload]` -- typed composition of user context (an `Event` subclass) and processing payload
- `ImagePayload` -- carries `image_path` or `image_bytes` for preprocessor consumption
- `TritonRequest` -- carries preprocessed tensor or raw bytes for Triton inference
- `DetectionResult` -- structured detections with metrics annotations
- `ClassificationResult` -- structured classification predictions with metrics annotations
- `FrameworkMeta` -- framework-owned metadata, set by RedisStreamSource, stripped by RedisStreamSink

The `PipelineEvent` pattern allows the original user event (context) to travel alongside processing stages without being lost.

### Type-Based Routing

The framework routes events using **subtype-aware matching**:

- **isinstance at runtime**: `isinstance(result, expected_type)` — subclass instances route to parent-type consumers
- **issubclass at validation**: a produced `ChildEvent` satisfies a consumer expecting `ParentEvent`
- **Union inputs**: `BaseThreadProcessor[A | B | C, Out]` subscribes to all three types from any source
- **Container unpacking**: `list[T]` and `tuple[T, ...]` return types are unpacked — each element is routed individually
- **Pydantic generics**: fallback comparison via `__pydantic_generic_metadata__` for thread-safety

## ModelBundle

`ModelBundle` (in `engine/bundle.py`) groups the four core processors for one model into a pipeline segment:

```python
bundle = ModelBundle(
    name="yolo",
    preprocessor=ClientPreprocessorNode(...),
    inference=InferenceNode(grpc_url="triton:8001"),
    postprocessor=PostProcessorNode(postprocessor=yolo_bridge),
    sink=LogErrorSinkNode(),
)
processors = expand_bundle(bundle)  # returns [preprocessor, inference, postprocessor, sink]
```

`expand_bundle(bundle)` returns the bundle's processors in pipeline order, suitable for `Application.build(processors=[...])`.

`YoloPostProcessorBridge` accepts config-friendly constructor params (`end2end`, `conf_threshold`, `iou_threshold`) directly and constructs the internal `YoloPostprocessor` automatically. A pre-built `YoloPostprocessor` instance can also be passed for programmatic use.

## EventContext and Field Passthrough

`EventContext` (`engine/context.py`) provides per-event metadata via `contextvars`:

```python
ctx = current_context()  # -> EventContext(message_id, source_stream, created_at, original_fields)
```

The `original_fields` dict carries all fields from the inbound Redis Stream message. `EventSerializer` reads `current_context()` and includes `original_fields` in the outbound `Envelope`, ensuring input fields pass through the pipeline untouched.

Set by the framework before dispatch, reset after. User code never sees the `Envelope` directly.

## PipelineConfig

`AppConfig` includes a `pipeline` section (`PipelineConfig` in `config/schema.py`) that configures shipped processors resolved by `build_application`:

- `model_name: str` -- Triton model name used by the preprocessor and inference worker
- `model_input_name: str` -- input tensor name (default `"images"`)
- `model_output_names: list[str]` -- output tensor names (default `["output0"]`)
These are pure infrastructure parameters. The actual `event_model` type and `postprocessor` instance come from Python code, not config.

## DecoderType

`DecoderType` (`config/schema.py`) controls which image decoder backend `build_application` uses for the preprocessor:

| Value | Preprocessor Used | Preprocessing Location |
|---|---|---|
| `OPENCV` | `ConfigurableClientPreprocessor` (OpenCV decoder) | Client-side |
| `TURBOJPEG` | `ConfigurableClientPreprocessor` (TurboJPEG decoder) | Client-side |
| `DALI` | `DaliTritonRequestBuilder` | Server-side (Triton DALI ensemble) |

When `decoder: dali`, `build_application` sends raw image bytes to Triton for server-side decoding and preprocessing via a DALI ensemble pipeline. Client-side decoders (`opencv`, `turbojpeg`) read the file, decode, letterbox, and send the preprocessed tensor.

Configured via `PreprocessingConfig.decoder` (default: `OPENCV`).

## TensorRT Acceleration

The `convert` CLI command exports YOLO models to ONNX format. Triton's `onnxruntime` backend can optionally use TensorRT as an execution provider for GPU acceleration. The `ExportFormat` enum in `triton_shared_models` supports both `ONNX` and `TENSORRT` targets. TRT plan files use the `tensorrt_plan` platform in Triton config.

See: `triton_model_tools/export/`, `triton_shared_models/triton_types.py`

## Metrics System

Metrics are declared on event classes, not in the application layer. Three metric types live in `protocols.py` (leaf layer):

- **Counter**: `_metrics_ = [Counter("name")]` on event class -- increments per event produced
- **Gauge**: `Annotated[float, Gauge("name")]` on event field -- extracts numeric value
- **GroupBy**: `Annotated[str, GroupBy("name", window_s=60)]` on event field -- counts by key over time window

`MetricEvent` has three fields: `component_name`, `measure`, `value`.

Framework extracts metrics in wrappers (`_extract_and_record_metrics` in `engine/wrappers.py`) and records them to `MetricBuffer` (`metrics/buffer.py`). `MetricEmitter` (`metrics/emitter.py`) flushes to a Redis Stream on a timer. `TritonMetricsCollector` (`metrics/triton_collector.py`) polls Triton server stats.

Per-node throughput/latency is tracked via `MetricAggregator` (`engine/metric_aggregator.py`) and the `Tracked` annotation.

Zero-value guarantee: `_log_stats` emits all registered Counter names even when idle.

## Composable Middleware

`engine/middleware.py` provides `RetryMiddleware[TIn, TOut]` -- exponential backoff retry with configurable attempts. Satisfies `Processor[TIn, TOut]` and composes via nesting.

## Pipeline Construction

### Preferred: Application.build() with explicit processor list

1. Load config via Hydra YAML + env var interpolation
2. Create processor instances using `ModelBundle` + `expand_bundle()`
3. Call `Application.build(config, processors=[...])` -- validates graph, creates metrics
4. `app.run()` -- starts all nodes in dedicated threads, monitors health

All nodes run in dedicated threads. The main loop monitors variant health and handles graceful shutdown.

See: `examples/dual_model_pipeline/run_pipeline.py`

## Infrastructure

Docker Compose in `.devcontainer/`:

| Service | Image | Purpose |
|---|---|---|
| Redis | `redis:7-alpine` | Streams transport, AOF persistence |
| Triton | `nvcr.io/nvidia/tritonserver:26.02-py3` | Inference server, `--strict-model-config=true` |
| Triton SDK | `nvcr.io/nvidia/tritonserver:26.02-py3-sdk` | `perf_analyzer`, `model_analyzer`, `tritonclient` |
| RedisInsight | `redis/redisinsight:latest` | Redis web UI |

Two compose files: `docker-compose.yml` (devcontainer, internal network) and `docker-compose.debug.yml` (exposed ports for host access).

Model management: `model_repository/` is live (mounted into Triton). Triton runs in explicit mode — models are loaded/unloaded via HTTP API (`POST /v2/repository/models/{name}/load`).

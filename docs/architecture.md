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
providers                                                             <- top: DI
connectors | inference | preprocessing | metrics | routers            <- domain siblings
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

Three primitives, each with execution mode variants (9 total). All protocols defined in `protocols.py`.

### Source (produces events)

| Variant | Protocol | Execution |
|---|---|---|
| `Source` | `Source[TOut]` | Inline, polled by main loop |
| `ThreadSource` | `ThreadSource[TOut]` | Dedicated thread, has `stop: Event` |
| `ProcessSource` | `ProcessSource[TOut]` | Dedicated OS process, has `stop: Event` |

### Processor (transforms events)

| Variant | Protocol | Execution |
|---|---|---|
| `Processor` | `Processor[TIn, TOut]` | Inline, dispatched by main loop |
| `ThreadProcessor` | `ThreadProcessor[TIn, TOut]` | Dedicated thread |
| `ThreadPoolProcessor` | `ThreadPoolProcessor[TIn, TOut]` | N threads, shared queue |
| `ProcessPoolProcessor` | `ProcessPoolProcessor[TIn, TOut]` | N processes, shared queue |

### Sink (consumes events)

| Variant | Protocol | Execution |
|---|---|---|
| `Sink` | `Sink[TIn]` | Inline, runs in main loop |
| `ThreadSink` | `ThreadSink[TIn]` | Dedicated thread, has `stop: Event` |

User code has two options for declaring the execution variant:

1. **Inherit from base classes** (preferred for user processors) -- the variant is declared by the class itself, no wrapping needed. See "Base Classes for User Processors" below.
2. **Write plain callables** and wrap at assembly time using wrappers from `engine/wrappers.py`: `DetachedThreadSource`, `DetachedProcessSource`, `DetachedThreadProcessor`, `ThreadedProcessor`, `MultiProcessProcessor`, `DetachedThreadSink`. Use this for framework building blocks that should remain variant-agnostic.

See: `protocols.py` (base classes), `engine/wrappers.py`, `engine/registry.py`

### Base Classes for User Processors

User-defined processors inherit from base classes that declare the execution variant explicitly. Nine base classes in `protocols.py` correspond to the nine execution variants:

| Base Class | Variant | Has `stop` |
|---|---|---|
| `BaseSource[TOut]` | Inline source | No |
| `BaseThreadSource[TOut]` | Threaded source | Yes |
| `BaseProcessSource[TOut]` | Process source | Yes |
| `BaseProcessor[TIn, TOut]` | Inline processor | No |
| `BaseThreadProcessor[TIn, TOut]` | Dedicated thread | Yes |
| `BaseThreadPoolProcessor[TIn, TOut]` | Thread pool | Yes |
| `BaseProcessPoolProcessor[TIn, TOut]` | Process pool | Yes |
| `BaseSink[TIn]` | Inline sink | No |
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

### Config-Driven Node Assembly

`NodeConfig` is primarily for **user-defined processors** that need a `class_path` to be imported dynamically. Shipped processors (extractor, preprocessor, inference worker, serializer, error sink) are registered automatically by `build_application` from config fields -- no `NodeConfig` entries needed. The variant is a deployment decision in config, not hardcoded in the processor.

Config schema (`config/schema.py`):
- `NodeVariant` -- StrEnum mirroring `ExecutionVariant` (SOURCE, THREAD_SOURCE, PROCESS_SOURCE, PROCESSOR, THREAD_PROCESSOR, THREAD_POOL_PROCESSOR, PROCESS_POOL_PROCESSOR, SINK, THREAD_SINK)
- `NodeConfig` -- name, class_path, variant, params, tracked, pool_size
- `AppConfig.nodes: list[NodeConfig]` for user-defined processors that require dynamic import

Assembly (`engine/assembly.py`):
- `import_class(class_path)` -- dynamic import from fully-qualified dotted path
- `resolve_type_params(cls, params)` -- auto-resolve `type[X]` params from string class paths
- `wrap_with_variant(inner, variant)` -- apply execution variant wrapper (idempotent for base-class nodes that already declare the correct variant)
- `assemble_all(nodes, registry)` -- assemble and register all config-driven nodes

DI Provider (`providers/node_assembly_provider.py`):
- `NodeAssemblyProvider` reads `AppConfig.nodes`, assembles, and registers all nodes before `Application` starts

### build_application Factory

`providers/app_factory.py` provides `build_application()` -- a high-level factory that creates a fully wired `Application` from an `AppConfig` with minimal boilerplate.

```python
app = build_application(
    config,
    event_model=SteelInspectionInput,
    postprocessor=YoloPostProcessorBridge(
        conf_threshold=config.pipeline.postprocessor.conf_threshold,
        iou_threshold=config.pipeline.postprocessor.iou_threshold,
        end2end=config.pipeline.postprocessor.end2end,
    ),
    extra_nodes=[("branch_router", CameraBranchRouter())],
)
```

What `build_application` does automatically:
- Creates Redis client and verifies connectivity
- Creates `RedisStreamSource`/`RedisStreamSink` from `config.sources`/`config.sinks`
- Registers shipped processors from config fields: `EnvelopeExtractor` (from `event_model`), preprocessor (from `preprocessing.decoder` + `pipeline.model_name`), `InferenceWorker` (from `inference.grpc_url`), `EventSerializer`, `LogErrorSink`
- Assembles any `NodeConfig` entries from `config.nodes`
- Registers user processors passed via `extra_nodes`

Separation of concerns:
- **Config** contains only infrastructure params (Redis URLs, model names, thresholds, tensor names) -- no class paths for shipped processors
- **Code** provides `event_model` (pydantic model for envelope extraction) and `postprocessor` (e.g., `YoloPostProcessorBridge` from model_tools)
- **extra_nodes** receives user-defined processors that extend the pipeline graph

## Shipped Processors

Seven reusable processors provided out of the box. Five live in `triton_redis_connector.engine.shipped_processors`, one in `triton_redis_connector.providers.shipped_factories`, one in `triton_model_tools.inference.yolo_bridge`. Shipped processors are variant-agnostic plain callables. When using `build_application`, they are registered automatically from config fields. They can also be assembled manually from YAML config with any execution variant via `NodeConfig` entries (see "Config-Driven Node Assembly" above).

| Processor | Location | Input | Output | Purpose |
|---|---|---|---|---|
| `EnvelopeExtractor[T]` | connector | `Envelope[bytes]` | `T` (pydantic) | Deserialize envelope bytes into a typed pydantic model |
| `DaliInferenceInputBuilder` | connector | `_ImageBytesEvent` or `_ImagePathEvent` | `InferenceInput` | Wrap raw image bytes for DALI ensemble inference (reads from path if needed) |
| `ClientPreprocessor` | connector | `_ImagePathEvent` | `InferenceInput` | Read file, decode, letterbox, wrap as tensor |
| `ConfigurableClientPreprocessor` | connector (providers) | `_ImagePathEvent` | `InferenceInput` | Config-friendly ClientPreprocessor (decoder_type + target_size) |
| `EventSerializer[T]` | connector | `T` (pydantic) | `Envelope[bytes]` | Serialize event to JSON bytes with original_fields passthrough |
| `LogErrorSink` | connector | `BaseModel` | `None` | Log error events at error level via loguru |
| `YoloPostProcessorBridge` | model_tools | `InferenceOutput` | `DetectionResult` | YOLO postprocessing (NMS + confidence filter), config-friendly constructor |

**Metadata bypass**: All preprocessors (`DaliInferenceInputBuilder`, `ClientPreprocessor`, `ConfigurableClientPreprocessor`) support metadata passthrough. If the input event carries a `metadata: dict[str, Any]` field, it is copied through the pipeline via `InferenceInput.metadata` -> `InferenceOutput.metadata` -> `DetectionResult.metadata` without interpretation. This allows domain-specific fields (camera ID, timestamp, custom tags) to travel alongside inference data.

`YoloPostProcessorBridge` accepts config-friendly constructor params (`end2end`, `conf_threshold`, `iou_threshold`) directly and constructs the internal `YoloPostprocessor` automatically. A pre-built `YoloPostprocessor` instance can also be passed for programmatic use.

## EventContext and Field Passthrough

`EventContext` (`engine/context.py`) provides per-event metadata via `contextvars`:

```python
ctx = current_context()  # -> EventContext(message_id, source_stream, created_at, original_fields)
```

The `original_fields` dict carries all fields from the inbound Redis Stream message. `EventSerializer` reads `current_context()` and includes `original_fields` in the outbound `Envelope`, ensuring input fields pass through the pipeline untouched.

Set by the framework before dispatch, reset after. User code never sees the `Envelope` directly.

## CollectorMode

Defined in `config/schema.py`. Controls Redis Stream consumer group startup behavior:

| Mode | `XGROUP_CREATE` id | Behavior |
|---|---|---|
| `RECOVERY` | `0` | Process pending (unacked) messages first, then new |
| `FROM_LATEST` | `$` | Skip all existing messages, start from new only |
| `FROM_OLDEST` | `0` | Process all messages from the beginning of the stream |

Configured per source via `SourceConfig.collector_mode`.

## PipelineConfig

`AppConfig` includes a `pipeline` section (`PipelineConfig` in `config/schema.py`) that configures shipped processors resolved by `build_application`:

- `model_name: str` -- Triton model name used by the preprocessor and inference worker
- `model_input_name: str` -- input tensor name (default `"images"`)
- `model_output_names: list[str]` -- output tensor names (default `["output0"]`)
- `postprocessor: PostprocessorConfig` -- YOLO postprocessor thresholds:
  - `conf_threshold: float` (default 0.25)
  - `iou_threshold: float` (default 0.45)
  - `end2end: bool` (default False)

These are pure infrastructure parameters. The actual `event_model` type and `postprocessor` instance come from Python code, not config.

## DecoderType

`DecoderType` (`config/schema.py`) controls which image decoder backend `build_application` uses for the preprocessor:

| Value | Preprocessor Used | Preprocessing Location |
|---|---|---|
| `OPENCV` | `ConfigurableClientPreprocessor` (OpenCV decoder) | Client-side |
| `TURBOJPEG` | `ConfigurableClientPreprocessor` (TurboJPEG decoder) | Client-side |
| `DALI` | `DaliInferenceInputBuilder` | Server-side (Triton DALI ensemble) |

When `decoder: dali`, `build_application` sends raw image bytes to Triton for server-side decoding and preprocessing via a DALI ensemble pipeline. Client-side decoders (`opencv`, `turbojpeg`) read the file, decode, letterbox, and send the preprocessed tensor.

Configured via `PreprocessingConfig.decoder` (default: `OPENCV`).

## Per-Model Output Routing

`OutputStreamRouter` (`connectors/output_router.py`) maps model names to output Redis streams.

Configuration via `OutputRoutingConfig`:
- `model_streams: dict[str, str]` -- explicit model-to-stream mapping
- `default_stream: str | None` -- fallback for unmapped models

The router lazily creates `RedisStreamOutputConnector` instances per stream. `validate_routing()` checks at startup that all known models are routable.

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

## Dependency Injection (dishka)

Three-scope hierarchy defined in `providers/scopes.py`:

| Scope | Lifetime | Usage |
|---|---|---|
| `APP` | Application | Singletons: config, Redis pool, registry, metric services |
| `WORKER` | Per-process | For `ProcessPoolProcessor` workers: own gRPC client, own connections |
| `REQUEST` | Per-event | EventContext for inline dispatch |

Providers (`providers/` directory):
- `ConfigProvider` -- config models from Hydra YAML
- `EngineProvider` -- PipelineRegistry, QueueBus, Application
- `ConnectorProvider` -- Redis connectors, ConsumerGroupManager, OutputStreamRouter
- `InferenceProvider` -- InferenceWorker, InferenceWorkerFactory
- `PreprocessingProvider` -- decoders, LetterBoxPreprocessor
- `MetricsProvider` -- MetricBuffer, MetricEmitter, TritonMetricsCollector

Container is not picklable. Workers receive only picklable config and create their own container.

## Composable Middleware

`engine/middleware.py` provides:
- `RetryMiddleware[TIn, TOut]` -- exponential backoff retry with configurable attempts
- `ConcurrencyLimitMiddleware[TIn, TOut]` -- semaphore-based concurrency limiting

Both satisfy `Processor[TIn, TOut]` and compose via nesting.

## Pipeline Construction

### Preferred: build_application (recommended)

1. Load config via `load_config()` (Hydra YAML + env var interpolation)
2. Call `build_application(config, event_model=..., postprocessor=..., extra_nodes=[...])`
3. `app.run()`

The factory handles Redis client creation, shipped processor registration, and graph wiring. See "build_application Factory" above for details.

### Manual construction (advanced)

1. Load config via `load_config()` (Hydra YAML + env var interpolation)
2. Create `PipelineRegistry`, register nodes with type-annotated `__call__`
3. Registry inspects signatures to build DAG, validates connectivity
4. Create `Application` with registry and `QueueBus`
5. `app.run()` validates graph, wires queues, starts parallel threads, enters main loop

Main loop: poll inline sources, type-match events to consumers, route recursively. Parallel nodes get dedicated threads with input/output queues. Health monitoring restarts crashed threads.

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

Model management: `model_repository/` is live (mounted into Triton), `model_repository_offline/` is staging. Triton polls for changes every 10 seconds.

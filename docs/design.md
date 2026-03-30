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

The connector is a graph-based pipeline framework where each node is a typed
primitive (Source, Processor, Sink) with a user-chosen execution variant.

- **Main loop**: dispatches events for inline (non-parallel) nodes only
- **Thread/Process variants**: self-contained — own their input/output queues,
  framework wires queues during graph construction
- **Graph verification at startup**: validates all edges, no orphans, no
  disconnected nodes

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
      protocols.py                  # Source, Processor, Sink — all execution variants

      config/
        __init__.py
        schema.py                   # Pydantic models for Hydra-loaded config
        hydra_app.py                # Hydra initialization, config composition

      connectors/
        __init__.py
        redis_stream.py             # RedisStreamSource (ThreadSource), RedisStreamSink (ThreadSink)
        consumer_group.py           # Consumer group management, position tracking

      engine/
        __init__.py
        app.py                      # Application — main entry point
        registry.py                 # Pipeline registry, type-annotation wiring, graph verification
        queue_bus.py                # Queue wiring between graph nodes
        event.py                    # Envelope[T] — timestamped event wrapper
        context.py                  # EventContext, current_context() via contextvars
        graph.py                    # DAG construction, edge validation, topological sort

      inference/
        __init__.py
        triton_processor.py         # ProcessPoolProcessor wrapping Triton gRPC inference
        request.py                  # InferenceInput / InferenceOutput event models
        model_registry.py           # ModelVersionTracker — rolling release (per-worker)

      preprocessing/
        __init__.py
        protocols.py                # ImageDecoder Protocol, ImagePreprocessor Protocol
        decoder.py                  # OpenCVDecoder, TurboJPEGDecoder
        image_decode_processor.py   # ThreadPoolProcessor / ProcessPoolProcessor for decoding

      metrics/
        __init__.py
        collector.py                # MetricEvent model, MetricBuffer
        triton_metrics.py           # TritonMetricsCollector — polls Triton stats
        error_collector.py          # ErrorCollector — interval-based error aggregation
        emitter.py                  # MetricEmitter — writes to output stream

      errors/
        __init__.py
        models.py                   # ConnectorError, InferenceError, PreprocessingError
        handler.py                  # Error events flow through graph like any other event

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
│  APPLICATION PROCESS                                                  │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │              GRAPH CONSTRUCTION & VERIFICATION                  │  │
│  │  PipelineRegistry      ProcessingGraph      QueueBus           │  │
│  │  (type-hint wiring)    (DAG from unions)    (queue wiring)     │  │
│  └──┬────────────────────┬────────────────────────┬───────────────┘  │
│     │                    │                        │                  │
│  ┌──▼────────────────────▼────────────────────────▼───────────────┐  │
│  │              MAIN LOOP (inline nodes only)                      │  │
│  │                                                                │  │
│  │  Source nodes → poll __call__() → route events by type          │  │
│  │  Processor nodes → dispatch events → route outputs by type     │  │
│  │  Sink nodes → dispatch events → side effects                   │  │
│  │                                                                │  │
│  │  Also: listen terminal events queue, monitor parallel health    │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │              PARALLEL NODES (self-contained, own queues)        │  │
│  │                                                                │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌───────────────────────┐  │  │
│  │  │ ThreadSource │  │ ThreadSink  │  │ ThreadProcessor       │  │  │
│  │  │ (e.g. Redis  │  │ (e.g. Redis │  │ owns input_queue      │  │  │
│  │  │  XREADGROUP) │  │  XADD)      │  │ owns output_queue     │  │  │
│  │  │ → output_q   │  │ ← input_q   │  │ reads/writes itself   │  │  │
│  │  └─────────────┘  └─────────────┘  └───────────────────────┘  │  │
│  │                                                                │  │
│  │  ┌───────────────────────┐  ┌───────────────────────────────┐  │  │
│  │  │ ThreadPoolProcessor   │  │ ProcessPoolProcessor          │  │  │
│  │  │ N threads, shared     │  │ N processes, shared            │  │  │
│  │  │ input_queue, fan-out  │  │ input_queue, fan-out           │  │  │
│  │  └───────────────────────┘  └───────────────────────────────┘  │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │                    METRICS LAYER (metrics/)                     │  │
│  │  MetricBuffer → MetricEmitter → Redis Stream "metrics"         │  │
│  │  TritonMetricsCollector | ErrorCollector | TimeWindowAggregator │  │
│  └────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
```

Key difference from old architecture: the main loop only dispatches events to
inline (non-parallel) nodes. Thread/Process variants are self-contained — they
read from their input queue and write to their output queue independently.
The framework wires these queues during graph construction.

### 3.1 import-linter: автоматическая проверка слоёв

Границы слоёв проверяются автоматически через `import-linter` (хук pre-commit,
запускается после pyright). Конфигурация в `pyproject.toml`, секция
`[tool.importlinter]`. Четыре контракта:

| Контракт | Тип | Правило |
|---|---|---|
| `connector-layers` | layers | `connectors\|inference\|preprocessing\|metrics` → `engine` → `config\|errors` → `protocols` |
| `connector-no-model-tools` | forbidden | `triton_redis_connector` не импортирует `triton_model_tools` |
| `model-tools-no-connector` | forbidden | `triton_model_tools` не импортирует `triton_redis_connector` |
| `model-tools-layers` | layers | `__main__` → `exporter` → `repository` → `models` |

Модули `triton_redis_connector`, которые ещё не созданы, помечены как
опциональные (`()`) — контракт проходит, пока модуля нет, и автоматически
активируется при его создании.

Слои в `connector-layers` читаются сверху вниз: каждый слой может
импортировать только из слоёв ниже. Модули, разделённые `|` на одном уровне,
не могут импортировать друг друга (независимые siblings).

---

## 4. Data Flow

### 4.1 Happy Path: Image Event through Inference

```
Step 1: Redis Stream "camera_images"
        XREADGROUP GROUP app_group consumer_1 COUNT 10 BLOCK 1000

Step 2: RedisStreamSource (ThreadSource, dedicated thread)
        - Blocks on XREADGROUP internally
        - Decodes Redis fields into pydantic model (CameraImageEvent)
        - Returns CameraImageEvent (framework wraps in Envelope, routes to output queue)
        - Does NOT XACK yet (tracked in pending set)

Step 3: Framework routes CameraImageEvent to matching Processor
        - Graph edge: CameraImageEvent → ImageDecodeProcessor
        - If processor is inline (Processor): main loop calls __call__(event)
        - If processor is parallel (ThreadPoolProcessor/ProcessPoolProcessor):
          event already on processor's input queue, worker picks it up

Step 4: ImageDecodeProcessor (ThreadPoolProcessor or ProcessPoolProcessor)
        - Worker reads CameraImageEvent from input queue
        - Decodes image bytes using configured decoder (opencv|turbojpeg)
        - Returns DecodedImageEvent (placed on processor's output queue)

Step 5: Framework routes DecodedImageEvent to TritonInferenceProcessor
        - TritonInferenceProcessor is a ProcessPoolProcessor
        - Each worker process has own gRPC InferenceServerClient

Step 6: TritonInferenceProcessor worker picks up DecodedImageEvent
        - ModelVersionTracker resolves latest ready version (per-worker)
        - Builds tritonclient InferInput from ndarray
        - Calls async_infer() with callback
        - On failure: callback schedules retry via threading.Timer
          with exponential delay, re-calls async_infer with new callback
        - Semaphore released only on final success or final failure
        - On success: returns InferenceOutput (placed on output queue)

Step 7: Framework routes InferenceOutput to next Processor
        - Graph edge: InferenceOutput → DetectionResultHandler
        - DetectionResultHandler returns DetectionResult

Step 8: Framework routes DetectionResult to Sink
        - Graph edge: DetectionResult → RedisStreamSink

Step 9: RedisStreamSink (ThreadSink, dedicated thread)
        - Reads DetectionResult from its input queue
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

`AppConfig` и все sub-config модели становятся dishka-provided зависимостями
в APP scope. Поток данных: Hydra YAML → `OmegaConf.to_container()` →
`AppConfig(**dict)` → dishka `ConfigProvider.app_config()` → все остальные
провайдеры получают конфиги через инъекцию. См. секцию 5A.

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
  - myapp.processors.ImageDecodeProcessor
  - myapp.processors.TritonInferenceProcessor
  - myapp.processors.DetectionResultHandler
```

---

## 5A. Dependency Injection (dishka)

### 5A.1 Зачем dishka

Проект использует [dishka](https://github.com/reagento/dishka) — современный
Python DI-фреймворк с поддержкой async, scope-менеджментом и высокой
производительностью (в ~20 раз быстрее dependency-injector).

Dishka заменяет ручное создание зависимостей в `Application.from_config()`
декларативными провайдерами. Это делает стартовую последовательность тестируемой
(подмена провайдеров для тестов) и расширяемой (новые провайдеры для новых слоёв
без изменения Application).

### 5A.2 Маппинг скоупов

| Dishka Scope | Время жизни | Объекты | Обоснование |
|---|---|---|---|
| `APP` | Время жизни приложения (singleton) | `AppConfig`, Redis connection pool, `PipelineRegistry`, `ProcessingGraph`, `QueueBus`, `MetricBuffer` | Создаются один раз при старте, разделяются между inline и thread-нодами |
| `REQUEST` | Один цикл обработки события | `EventContext`, метаданные события | Скоуп для inline `Processor` и `Sink` — один event dispatch cycle |
| `WORKER` (кастомный) | Один worker-процесс `ProcessPoolProcessor` | `InferenceServerClient`, `redis.Redis`, `ModelVersionTracker` | Соединения не fork-safe — каждый worker создаёт свои |

### 5A.3 Организация провайдеров

Провайдеры следуют иерархии слоёв import-linter. Каждый провайдер соответствует
одному архитектурному слою:

```
providers/
  scopes.py                # Кастомный WORKER scope
  config_provider.py       # config слой: AppConfig, sub-configs
  engine_provider.py       # engine слой: PipelineRegistry, ProcessingGraph, QueueBus
  connector_provider.py    # connectors слой: Redis connection pool, ConsumerGroupManager
  inference_provider.py    # inference слой: worker container factory
  preprocessing_provider.py # preprocessing слой: decoder instances
  metrics_provider.py      # metrics слой: MetricBuffer, MetricEmitter, collectors
```

Провайдер может зависеть от объектов провайдеров нижних слоёв (по правилам
import-linter), но никогда от верхних слоёв или сиблинг-модулей домена.

### 5A.4 Жизненный цикл контейнера

```
1. Hydra загружает config → OmegaConf.to_container()
2. Создаётся AppConfig (pydantic)
3. Создаются экземпляры Provider, AppConfig передаётся в ConfigProvider
4. make_container(ConfigProvider, EngineProvider, ConnectorProvider, ...) → container
5. Container входит в APP scope (создаёт singletons: Redis pool, registry и т.д.)
6. Main loop: для каждого inline event dispatch входит в REQUEST scope
7. ProcessPoolProcessor workers: создают child container с WORKER scope
8. Shutdown: container.close() — teardown APP scope (закрытие соединений)
```

### 5A.5 DI для ProcessPoolProcessor workers

Каждый worker-процесс получает фабричную функцию (picklable), которая создаёт
свежий dishka-контейнер с WORKER scope. Фабрика захватывает только сериализуемый
конфиг (pydantic-модели, которые picklable). В init worker'а:

```python
def _worker_init(config: InferenceConfig) -> TritonInferenceProcessor:
    worker_container = make_container(WorkerProvider(config))
    return worker_container.get(TritonInferenceProcessor)
```

Это заменяет ручное создание зависимостей в worker init. Контейнер worker'а
закрывается при завершении процесса.

### 5A.6 Связь с factory pattern

Dishka и factory pattern — взаимодополняющие механизмы:

- **Framework-internal wiring** (registry, graph, queues, connections) → dishka
- **User-defined shared state** (`create_shared`/`create_processor`) → factory
  pattern (остаётся без изменений)

Для продвинутых пользователей: dishka-провайдер может реализовать factory protocol
изнутри, делая `create_shared` методом провайдера, а `create_processor` — scoped
зависимостью.

### 5A.7 Интеграция Hydra → Dishka

```python
class ConfigProvider(Provider):
    scope = Scope.APP

    @provide
    def app_config(self) -> AppConfig:
        return self._config  # передан через __init__ из Hydra

    @provide
    def redis_config(self, config: AppConfig) -> RedisConnectionConfig:
        return config.redis

    @provide
    def inference_config(self, config: AppConfig) -> InferenceConfig:
        return config.inference
```

Точка входа:

```python
@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    app_config = AppConfig(**OmegaConf.to_container(cfg, resolve=True))
    container = make_container(
        ConfigProvider(app_config),
        EngineProvider(),
        ConnectorProvider(),
        InferenceProvider(),
        PreprocessingProvider(),
        MetricsProvider(),
    )
    app = container.get(Application)
    try:
        app.run()
    finally:
        container.close()
```

---

## 6. Protocols & Execution Variants

All extension points use `typing.Protocol` with `@runtime_checkable`. Users
implement these interfaces via structural subtyping — no framework base classes.

There are three primitives (Source, Processor, Sink), each with execution variants
that control where the code runs. The framework wires queues during graph
construction. All parallel variants own their queues. The `stop: threading.Event`
field is a protocol contract on all Thread/Process variants — the framework creates
it, assigns it before start, and calls `stop.set()` at shutdown.

### 6.1 Source Variants (produce events, no input)

Signature: `__call__(self) -> Event1 | Event2 | None`

`None` means nothing ready right now. The framework loops the call.

```python
@runtime_checkable
class Source(Protocol[TOut]):
    """Inline source. Runs in main loop, polled by framework.

    Framework calls __call__() repeatedly. Return an event when one is
    ready, None when nothing available. Non-blocking — must not sleep
    or block internally.
    """

    def __call__(self) -> TOut | None: ...


@runtime_checkable
class ThreadSource(Protocol[TOut]):
    """Threaded source. Runs in a dedicated thread, blocks internally.

    Framework creates stop Event, assigns before starting the thread,
    calls stop.set() at shutdown. Framework loops __call__() in the
    thread — source blocks internally (e.g. XREADGROUP BLOCK) and
    returns event when ready, None when nothing available.
    Output goes to framework-wired output queue.
    """

    stop: threading.Event

    def __call__(self) -> TOut | None: ...


@runtime_checkable
class ProcessSource(Protocol[TOut]):
    """Process source. Runs in a dedicated OS process.

    Same contract as ThreadSource but in a separate process.
    Output goes to framework-wired multiprocessing output queue.
    stop is a multiprocessing.Event (duck-typed as threading.Event
    in the protocol — both have .set(), .is_set(), .wait()).
    """

    stop: threading.Event

    def __call__(self) -> TOut | None: ...
```

### 6.2 Processor Variants (transform events)

Signature: `__call__(self, event: InputEvent) -> OutputEvent1 | OutputEvent2 | None`

`None` means filter — this event is dropped. Union return types create multiple
edges in the processing graph.

```python
@runtime_checkable
class Processor(Protocol[TIn, TOut]):
    """Inline processor. Runs in main loop, framework dispatches events.

    Runtime unwraps Envelope.payload before calling __call__,
    wraps result in new Envelope after return.
    Users access metadata via current_context() contextvars.
    Return None to filter (skip this event).
    """

    def __call__(self, event: TIn) -> TOut | None: ...


@runtime_checkable
class ThreadProcessor(Protocol[TIn, TOut]):
    """Threaded processor. Runs in a dedicated thread.

    Owns input_queue and output_queue, wired by framework.
    Reads events from input_queue, processes, writes to output_queue.
    Framework loops __call__() — processor receives one event at a time.
    """

    stop: threading.Event

    def __call__(self, event: TIn) -> TOut | None: ...


@runtime_checkable
class ThreadPoolProcessor(Protocol[TIn, TOut]):
    """Thread pool processor. N threads in a pool, shared input queue.

    Framework creates N instances (one per worker thread) and a shared
    input queue. Each worker reads from the shared input queue and
    writes to a shared output queue. Fan-out parallelism.
    """

    stop: threading.Event

    def __call__(self, event: TIn) -> TOut | None: ...


@runtime_checkable
class ProcessPoolProcessor(Protocol[TIn, TOut]):
    """Process pool processor. N processes in a pool, shared input queue.

    Framework creates N instances (one per worker process) via factory.
    Each worker process reads from shared mp.Queue input and writes to
    shared mp.Queue output. True OS-level parallelism, no GIL.
    Objects crossing the process boundary are serialized via pickle.
    """

    stop: threading.Event

    def __call__(self, event: TIn) -> TOut | None: ...
```

### 6.3 Sink Variants (terminal side-effect, optional metric output)

Signature: `__call__(self, event: Event) -> MetricEvent1 | MetricEvent2 | None`

`None` means no metric output. Sinks produce side effects (e.g. write to Redis).
Optional return of metric events allows sinks to emit instrumentation data
back into the graph.

```python
@runtime_checkable
class Sink(Protocol[TIn]):
    """Inline sink. Runs in main loop.

    Terminal node — produces side effects. Can optionally return
    metric events that flow back into the graph.
    """

    def __call__(self, event: TIn) -> Any | None: ...


@runtime_checkable
class ThreadSink(Protocol[TIn]):
    """Threaded sink. Runs in a dedicated thread, owns input queue.

    Framework wires input queue. Sink reads events and produces
    side effects. Can optionally return metric events.
    """

    stop: threading.Event

    def __call__(self, event: TIn) -> Any | None: ...
```

### 6.4 Factory Protocol for Shared State

By default, framework creates N independent instances for pool variants
(ThreadPoolProcessor, ProcessPoolProcessor). Each worker gets its own state,
no locks needed.

For opt-in shared state, implement a factory:

```python
@runtime_checkable
class ProcessorFactory(Protocol[TShared, TProcessor]):
    """Factory for creating processor instances with shared state.

    Framework calls create_shared() once, then create_processor(shared)
    N times — one per worker. User is responsible for thread/process
    safety of the SharedContext.
    """

    def create_shared(self) -> TShared: ...

    def create_processor(self, shared: TShared) -> TProcessor: ...
```

Contract by variant:

- **ThreadPoolProcessor**: `SharedContext` can use `threading.Lock`-protected
  structures, thread-safe collections, or immutable data.
- **ProcessPoolProcessor**: `SharedContext` must contain process-safe objects
  only — `multiprocessing.Queue`, `multiprocessing.Value`,
  `multiprocessing.Manager` proxies, or immutable data.

Framework documents the contract but does not validate safety — that
responsibility belongs to the user.

Example — Triton inference with shared model version cache:

```python
class TritonInferenceFactory:
    def __init__(self, config: InferenceConfig) -> None:
        self._config = config

    def create_shared(self) -> multiprocessing.Manager:
        manager = multiprocessing.Manager()
        return manager

    def create_processor(
        self, shared: multiprocessing.Manager,
    ) -> TritonInferenceProcessor:
        return TritonInferenceProcessor(
            config=self._config,
            version_cache=shared.dict(),
        )
```

#### Связь с dishka

Factory pattern и dishka — взаимодополняющие механизмы:

- **Framework-internal wiring** (registry, graph, queues, connections) использует dishka
- **User-defined shared state** (`create_shared`/`create_processor`) остаётся
  как есть — factory pattern для пользовательского кода

Dishka управляет тем, КАК создаются зависимости фреймворка. Factory pattern
управляет тем, КАК пользователь делит состояние между worker'ами.
Для продвинутых пользователей: dishka-провайдер может реализовать factory protocol
изнутри. См. секцию 5A.6.

### 6.5 Image Protocols (`preprocessing/protocols.py`)

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

### 6.6 Observer Protocol (`protocols.py`)

```python
@runtime_checkable
class ObserverProtocol(Protocol[TEvent]):
    """Observes events without producing output (metrics, logging)."""

    def observe(self, envelope: Envelope[TEvent]) -> None: ...
```

### 6.7 Aggregation Rule Protocol (`protocols.py`)

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

Implements `ThreadSource[T]`. Runs in a dedicated thread. Performs blocking
`XREADGROUP` calls and returns events directly.

```python
class RedisStreamSource:
    """Reads from a Redis Stream using consumer group semantics.

    Implements ThreadSource. Runs in a dedicated thread. The framework
    assigns stop: Event before starting the thread. Blocks on XREADGROUP
    internally, returns event when ready, None when timeout. Errors are
    returned as typed error events that flow through the graph.
    """

    stop: threading.Event

    def __init__(
        self,
        config: SourceConfig,
        redis_config: RedisConnectionConfig,
        event_model_class: type[BaseModel],
    ) -> None: ...

    def __call__(self) -> CameraImageEvent | ConnectorError | None:
        """Single iteration of the source loop.

        Blocks on XREADGROUP (up to block_ms). Returns:
        - Event on success
        - ConnectorError on Redis/validation failure
        - None on timeout (nothing available)

        while not self.stop.is_set():
            try:
                messages = redis.xreadgroup(...)
                if not messages:
                    return None
                for message_id, fields in messages:
                    try:
                        event = event_model_class.model_validate(fields)
                        return event
                    except ValidationError:
                        return ConnectorError(...)
            except redis.ConnectionError:
                return ConnectorError(...)
            except redis.TimeoutError:
                return None  # normal for BLOCK reads
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

Implements `ThreadSink[T]`. Runs in a dedicated thread. Reads from its
framework-wired input queue and writes via `XADD`.

```python
class RedisStreamSink:
    """Writes events to a Redis Stream.

    Implements ThreadSink. Runs in a dedicated thread. On successful
    XADD, triggers acknowledgment callback. Can return metric events
    for transit time tracking.
    """

    stop: threading.Event

    def __call__(self, event: DetectionResult) -> MetricEvent | None:
        """XADD to stream, trigger ack callback.
        Returns MetricEvent with transit time on success, None on failure.
        """
```

---

## 8. Inference Layer

### 8.1 Inference as a ProcessPoolProcessor

Triton inference is a regular `ProcessPoolProcessor` node in the graph — no
special intercept pattern. Each worker process creates its own gRPC client.
The factory pattern provides shared state if needed (e.g. model version cache
via `multiprocessing.Manager`).

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

These are regular event types flowing through the graph like any other event.
`InferenceInput` is the input type and `InferenceOutput` is the output type
of the `TritonInferenceProcessor`.

### 8.2 TritonInferenceProcessor (ProcessPoolProcessor)

Each worker process creates its own `tritonclient.grpc.InferenceServerClient`
and uses `client.async_infer()` with callback-based retry. The framework
creates N instances via a factory and distributes events across workers
through a shared `multiprocessing.Queue`.

```
Framework-wired queues
┌──────────────────┐               ┌─────────────────────────┐
│                  │  input_queue   │  Worker-0               │
│  Graph routing   │  (mp.Queue)   │  gRPC client (own conn) │
│  places          │──────────────►│  async_infer + callback  │
│  InferenceInput  │               │  retry via Timer         │
│  on shared queue │               ├─────────────────────────┤
│                  │               │  Worker-1               │
│                  │               │  gRPC client (own conn) │
│                  │◄──────────────│  async_infer + callback  │
│  Graph routing   │  output_queue │  retry via Timer         │
│  picks up        │  (mp.Queue)   ├─────────────────────────┤
│  InferenceOutput │               │  Worker-N               │
│                  │◄──────────────│  ...                     │
└──────────────────┘  output_queue └─────────────────────────┘
```

This design avoids Python GIL contention for gRPC serialization/deserialization,
allows true parallel utilization of Triton's gRPC endpoint, and isolates inference
failures from the main orchestrator.

```python
class TritonInferenceProcessor:
    """ProcessPoolProcessor for Triton gRPC inference.

    Each instance runs in its own worker process with an independent
    gRPC client. Uses async_infer with callback-based retry for
    concurrency within the worker.
    """

    stop: threading.Event

    def __init__(self, config: InferenceConfig) -> None:
        self._config = config
        self._client = tritonclient.grpc.InferenceServerClient(
            url=config.grpc_url,
        )
        self._semaphore = threading.Semaphore(
            config.max_concurrent_per_worker,
        )
        self._version_tracker = ModelVersionTracker(
            client=self._client,
            poll_interval_s=config.model_version_poll_interval_s,
        )

    def __call__(
        self, event: InferenceInput,
    ) -> InferenceOutput | InferenceError | None:
        """Run inference on a single event.

        Blocks until async_infer completes (via callback → threading.Event).
        Returns InferenceOutput on success, InferenceError on final failure.
        """
        self._semaphore.acquire()
        start_ns = time.monotonic_ns()

        version = self._version_tracker.resolve_version(event.model_name)
        result_holder: queue.Queue[InferenceOutput | InferenceError] = (
            queue.Queue(maxsize=1)
        )

        def _make_callback(
            attempt: int,
        ) -> Callable:
            def _on_complete(result, error):
                if error is None:
                    elapsed_ms = (
                        (time.monotonic_ns() - start_ns) / 1_000_000
                    )
                    result_holder.put(InferenceOutput(
                        model_name=event.model_name,
                        model_version=version,
                        outputs=_extract_outputs(
                            result, event.output_names,
                        ),
                        inference_time_ms=elapsed_ms,
                    ))
                    self._semaphore.release()
                    return

                if attempt >= self._config.max_retry_attempts:
                    result_holder.put(InferenceError(
                        model_name=event.model_name,
                        model_version=version,
                        error_type=type(error).__name__,
                        error_message=str(error),
                        retry_count=attempt,
                        timestamp_ms=int(time.time() * 1000),
                    ))
                    self._semaphore.release()
                    return

                delay = (
                    self._config.retry_base_delay_s
                    * self._config.retry_backoff_multiplier ** attempt
                )

                def _retry():
                    self._client.async_infer(
                        model_name=event.model_name,
                        model_version=version,
                        inputs=_build_inputs(event),
                        callback=_make_callback(attempt + 1),
                        outputs=_build_requested_outputs(event),
                    )

                timer = threading.Timer(delay, _retry)
                timer.start()

            return _on_complete

        self._client.async_infer(
            model_name=event.model_name,
            model_version=version,
            inputs=_build_inputs(event),
            callback=_make_callback(attempt=1),
            outputs=_build_requested_outputs(event),
        )

        return result_holder.get()  # blocks until callback fires
```

### 8.3 Rolling Release — ModelVersionTracker (`inference/model_registry.py`)

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

### 9.2 Image Decoding as a Pool Processor

Image decoding is a `ThreadPoolProcessor` or `ProcessPoolProcessor` node in the
graph. The pool type is configured per node (`decoder_pool_type: thread | process`),
defaulting to `thread` since both OpenCV and TurboJPEG release GIL during decode.

```python
class ImageDecodeProcessor:
    """ThreadPoolProcessor or ProcessPoolProcessor for image decoding.

    Execution variant chosen by decoder_pool_type config. Each pool
    worker has its own decoder instance. Receives raw image events,
    returns decoded image events with numpy arrays.

    For ProcessPoolProcessor: raw_bytes (input) and ndarray (output)
    are serialized across process boundary via pickle. For typical camera
    images (100KB-2MB in, ~3.7MB out at 640x640x3 uint8) this overhead
    is acceptable compared to decode time.
    """

    stop: threading.Event

    def __init__(self, config: PreprocessingConfig) -> None:
        decoder_cls = (
            OpenCVDecoder
            if config.decoder == DecoderType.OPENCV
            else TurboJPEGDecoder
        )
        self._decoder = decoder_cls()

    def __call__(
        self, event: CameraImageEvent,
    ) -> DecodedImageEvent | PreprocessingError:
        """Decode image bytes, return decoded event or error."""
        try:
            decoded = self._decoder.decode(event.image_data)
            raw = decoded.astype(np.float32) / 255.0
            return DecodedImageEvent(
                camera=event.camera,
                raw_image=raw,
                timestamp=event.timestamp,
            )
        except cv2.error as exc:
            return PreprocessingError(
                message_id=current_context().message_id,
                decoder_type=self._decoder.__class__.__name__,
                error_type=type(exc).__name__,
                error_message=str(exc),
                timestamp_ms=int(time.time() * 1000),
            )
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

No primitive ever raises — all errors are pushed as typed error events that flow
through the graph like any other event. Processors return error types in their
union return type. The graph routes errors to appropriate sinks or error-handling
processors.

```
Source.__call__() returns ConnectorError  ─────┐
                                                │
Processor.__call__() returns ErrorType  ────────┤  typed error events
                                                │  flow through graph
InferenceProcessor returns InferenceError  ────┤
                                                │
DecodeProcessor returns PreprocessingError  ───┤
                                                ▼
                                         Graph routing by type
                                                │
                                   ┌────────────┴────────────┐
                                   ▼                          ▼
                            ErrorHandler               ErrorMetricsSink
                            (Processor or Sink)        (aggregates, emits)
                                   │
                                   ▼
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

Sources and sinks return error events instead of raising. They never crash:

- `redis.ConnectionError` → return `ConnectorError(...)` + backoff sleep
- `redis.TimeoutError` → normal for BLOCK reads, return None
- `pydantic.ValidationError` → return `ConnectorError(...)`, skip message
- Any other specific exception → return typed error event, continue

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

### 15.1 Execution Variant Mapping

The process/thread model is defined by which execution variant the user picks
for each node. The framework handles all lifecycle management.

| Variant | Runs in | Queue ownership | Concurrency |
|---|---|---|---|
| `Source` | Main loop | None (returns directly) | Single-threaded |
| `ThreadSource` | Dedicated thread | output_queue (threading.Queue) | Thread per source |
| `ProcessSource` | Dedicated process | output_queue (mp.Queue) | Process per source |
| `Processor` | Main loop | None (dispatched by framework) | Single-threaded |
| `ThreadProcessor` | Dedicated thread | input_queue + output_queue | Thread per processor |
| `ThreadPoolProcessor` | N threads | shared input_queue, shared output_queue | N threads |
| `ProcessPoolProcessor` | N processes | shared input_queue (mp.Queue), shared output_queue (mp.Queue) | N processes |
| `Sink` | Main loop | None (dispatched by framework) | Single-threaded |
| `ThreadSink` | Dedicated thread | input_queue (threading.Queue) | Thread per sink |

### 15.2 Process Map (Example Pipeline)

```
┌────────────────────────────────────────────────────────────────────────┐
│  APPLICATION PROCESS                                                    │
│                                                                        │
│  Main Thread (main loop):                                              │
│  - Polls inline Source nodes                                           │
│  - Dispatches events to inline Processor/Sink nodes                    │
│  - Routes events between queues based on type graph                    │
│  - Listens on terminal events queue for graceful shutdown              │
│  - Monitors parallel variants health, restarts if crashed              │
│  - current_context() set via contextvars before each inline __call__   │
│                                                                        │
│  ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐       │
│  │ Thread:           │ │ Thread:           │ │ Thread:           │       │
│  │ RedisStreamSource │ │ RedisStreamSource │ │ RedisStreamSink   │       │
│  │ (ThreadSource)    │ │ (ThreadSource)    │ │ (ThreadSink)      │       │
│  │ XREADGROUP        │ │ XREADGROUP        │ │ XADD              │       │
│  │ stream_a          │ │ stream_b          │ │ results_stream    │       │
│  │ → output_queue    │ │ → output_queue    │ │ ← input_queue     │       │
│  └──────────────────┘ └──────────────────┘ └──────────────────┘       │
│                                                                        │
│  ┌──────────────────┐ ┌──────────────────┐                             │
│  │ Thread:           │ │ Timer: Metrics    │                             │
│  │ RedisStreamSink   │ │ - TritonMetrics   │                             │
│  │ (ThreadSink)      │ │ - Error flush     │                             │
│  │ metrics_stream    │ │ - Metric emit     │                             │
│  │ ← input_queue     │ │                   │                             │
│  └──────────────────┘ └──────────────────┘                             │
│                                                                        │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │  ThreadPoolProcessor: ImageDecodeProcessor                      │    │
│  │  N threads, shared input_queue ← CameraImageEvent               │    │
│  │  shared output_queue → DecodedImageEvent | PreprocessingError   │    │
│  │  Single decoder type per config (opencv | turbojpeg)            │    │
│  └────────────────────────────────────────────────────────────────┘    │
└────────────────────────────────────────────────────────────────────────┘
         │ input_queue (mp.Queue)          ▲ output_queue (mp.Queue)
         ▼                                 │
┌──────────────────────────────────────────────────────────────────────┐
│  ProcessPoolProcessor: TritonInferenceProcessor (N worker processes)  │
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
```

### 15.3 Queue Topology

All queues have configurable `maxsize` (via `QueueConfig`). When a queue is full,
the item is dropped (not blocked), loguru logs an error, and a `queue_overflow`
metric with the queue name tag is emitted.

| Queue | Type | Boundary | Producer | Consumer |
|---|---|---|---|---|
| ThreadSource output | `Queue[Envelope[T]]` | intra | ThreadSource thread | Graph router (main) |
| ProcessSource output | `mp.Queue[Envelope[T]]` | cross | ProcessSource process | Graph router (main) |
| ThreadProcessor input | `Queue[Envelope[T]]` | intra | Graph router (main) | ThreadProcessor thread |
| ThreadProcessor output | `Queue[Envelope[T]]` | intra | ThreadProcessor thread | Graph router (main) |
| ThreadPoolProcessor input | `Queue[Envelope[T]]` | intra | Graph router (main) | N worker threads |
| ThreadPoolProcessor output | `Queue[Envelope[T]]` | intra | N worker threads | Graph router (main) |
| ProcessPoolProcessor input | `mp.Queue[Envelope[T]]` | cross | Graph router (main) | N worker processes |
| ProcessPoolProcessor output | `mp.Queue[Envelope[T]]` | cross | N worker processes | Graph router (main) |
| ThreadSink input | `Queue[Envelope[T]]` | intra | Graph router (main) | ThreadSink thread |
| `pending_queue[stream]` | `PriorityQueue[Envelope[T]]` | intra | Startup recovery | Graph router (main) |
| `metric_buffer` | `MetricBuffer` | intra | All metric producers | MetricEmitter |

### 15.4 Backpressure: Drop on Full + Metrics

When any queue is full:
1. Drop the item (never block the producer)
2. Log via loguru: `logger.error("queue {} full, dropping item", queue_name)`
3. Emit `queue_overflow` metric with `tags={"queue": queue_name}`

This prevents cascading backpressure from stalling the entire pipeline. Dropped
items are visible in metrics dashboards for capacity planning.

### 15.5 Serialization Across Process Boundary

Objects crossing the `multiprocessing.Queue` boundary are serialized via pickle:

- **InferenceInput**: contains `model_name` (str), `model_version` (str),
  `raw_image` (numpy ndarray), `input_name` (str), `output_names` (tuple of str).
  Numpy arrays pickle efficiently — a 640x640x3 float32 tensor is ~4.7MB.
- **InferenceOutput**: contains `outputs` dict of numpy arrays.
  Same efficient numpy pickle.
- **InferenceError**: contains only strings and ints — trivial to pickle.

### 15.6 Phase 4C — Shared Memory IPC Optimization

After Phase 5 benchmarks confirm pickle IPC is the bottleneck, optimize with
`multiprocessing.shared_memory` for numpy arrays across the process boundary.
This eliminates the pickle serialization overhead for large tensors by passing
only a shared memory handle and metadata (shape, dtype) through the queue.

### 15.7 Inference Concurrency (per worker)

Each worker process runs `tritonclient.grpc.InferenceServerClient.async_infer()`.
This is non-blocking within the worker — it accepts a callback and returns
immediately. The callback fires in a gRPC internal thread when the response arrives
and puts the result onto the worker's internal completion queue.
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

Envelope is hidden from user `__call__` methods. The runtime:
1. Unwraps `envelope.payload` before calling `__call__`
2. Wraps the return value in a new Envelope inheriting metadata via `derive()`
3. Sets `current_context()` via `contextvars` for opt-in metadata access

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

When a user registers a node (Source, Processor, Sink), the registry:

1. Determines which execution variant the class satisfies (Source vs ThreadSource
   vs ProcessSource, etc.) by checking protocol conformance
2. Calls `typing.get_type_hints()` on the `__call__` method
3. For Sources: extracts return type (union types stripped of None)
4. For Processors: extracts `event: X` parameter type and return type `Y | Z | None`
5. For Sinks: extracts `event: X` parameter type and optional return type
6. **Union return types create multiple edges** in the graph. If a Processor
   returns `InferenceOutput | InferenceError | None`, that creates two outgoing
   edges — one for `InferenceOutput`, one for `InferenceError`
7. Builds directed graph from all type edges
8. Runs graph verification

```python
class PipelineRegistry:
    def register(self, node_class: type) -> None:
        variant = self._detect_variant(node_class)
        hints = typing.get_type_hints(node_class.__call__)

        if variant in SOURCE_VARIANTS:
            output_types = self._extract_union_types(
                hints["return"], exclude={type(None)},
            )
            for out_type in output_types:
                self._graph.add_edge(node_class, out_type)

        elif variant in PROCESSOR_VARIANTS:
            input_type = hints["event"]
            output_types = self._extract_union_types(
                hints["return"], exclude={type(None)},
            )
            self._graph.add_edge(input_type, node_class)
            for out_type in output_types:
                self._graph.add_edge(node_class, out_type)

        elif variant in SINK_VARIANTS:
            input_type = hints["event"]
            self._graph.add_edge(input_type, node_class)
            output_types = self._extract_union_types(
                hints.get("return"), exclude={type(None)},
            )
            for out_type in output_types:
                self._graph.add_edge(node_class, out_type)

    def _extract_union_types(
        self,
        annotation: type | None,
        exclude: set[type],
    ) -> list[type]:
        """Extract individual types from a union annotation.
        A | B | None → [A, B]. Handles both typing.Union and | syntax.
        """
        if annotation is None:
            return []
        origin = typing.get_origin(annotation)
        if origin is types.UnionType or origin is typing.Union:
            return [
                a for a in typing.get_args(annotation) if a not in exclude
            ]
        if annotation not in exclude:
            return [annotation]
        return []

    def build_and_verify(self) -> ProcessingGraph:
        """Wire all nodes and run graph verification.
        Raises ValueError on violations (see 17.3).
        """
```

### 17.2 Graph Verification at Startup

The full DAG is verified before the pipeline starts. Validation errors
at startup prevent running a broken graph:

1. **Every output type must have a consumer.** If a Processor emits type X and
   no other node accepts type X as input, the graph is invalid.
2. **Every input type must have a producer.** If a Processor expects type X but
   no Source or Processor produces type X, the graph is invalid.
3. **No orphan nodes.** Every registered node must have at least one edge.
4. **No disconnected subgraphs.** The entire graph must be reachable from at
   least one Source.
5. **No type ambiguity.** If two Processors accept the same input type,
   the graph is ambiguous and must be resolved (future: explicit routing config).
6. **Sources have no input edges.** Sinks have no output edges (metric returns
   are wired separately if present).

### 17.3 Wiring Example

```
RedisStreamSource (ThreadSource)
    │  __call__() → CameraImageEvent | ConnectorError | None
    │
    ├─ CameraImageEvent edge
    │  ▼
    │  ImageDecodeProcessor (ThreadPoolProcessor)
    │      __call__(CameraImageEvent) → DecodedImageEvent | PreprocessingError
    │      │
    │      ├─ DecodedImageEvent edge
    │      │  ▼
    │      │  TritonInferenceProcessor (ProcessPoolProcessor)
    │      │      __call__(DecodedImageEvent) → InferenceOutput | InferenceError
    │      │      │
    │      │      ├─ InferenceOutput edge
    │      │      │  ▼
    │      │      │  DetectionResultHandler (Processor, inline)
    │      │      │      __call__(InferenceOutput) → DetectionResult
    │      │      │      │
    │      │      │      └─ DetectionResult edge
    │      │      │         ▼
    │      │      │         RedisStreamSink (ThreadSink)
    │      │      │             __call__(DetectionResult) → MetricEvent | None
    │      │      │
    │      │      └─ InferenceError edge ──► ErrorHandlerSink
    │      │
    │      └─ PreprocessingError edge ──► ErrorHandlerSink
    │
    └─ ConnectorError edge ──► ErrorHandlerSink
```

### 17.4 Example User Code

```python
class CameraImageEvent(BaseModel):
    camera: str
    image_data: bytes
    timestamp: int


class DecodedImageEvent(BaseModel, arbitrary_types_allowed=True):
    camera: str
    raw_image: npt.NDArray[np.float32]
    timestamp: int


class DetectionResult(BaseModel):
    camera: str
    detections: list[dict[str, float]]
    timestamp: int


class ImageDecodeProcessor:
    """ThreadPoolProcessor[CameraImageEvent, DecodedImageEvent | PreprocessingError]."""

    stop: threading.Event

    def __init__(self, config: PreprocessingConfig) -> None:
        self._decoder = OpenCVDecoder()

    def __call__(
        self, event: CameraImageEvent,
    ) -> DecodedImageEvent | PreprocessingError:
        decoded = self._decoder.decode(event.image_data)
        raw = decoded.astype(np.float32) / 255.0
        return DecodedImageEvent(
            camera=event.camera,
            raw_image=raw,
            timestamp=event.timestamp,
        )


class DetectionResultHandler:
    """Processor[InferenceOutput, DetectionResult] — inline."""

    def __call__(
        self, event: InferenceOutput,
    ) -> DetectionResult:
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

С dishka DI, `Application` резолвится из контейнера вместо ручного
`Application.from_config()`. Контейнер строит полное дерево зависимостей
декларативно через провайдеры.

```python
class Application:
    def __init__(
        self,
        config: AppConfig,
        registry: PipelineRegistry,
        graph: ProcessingGraph,
        queue_bus: QueueBus,
        # ... другие зависимости инжектятся dishka
    ) -> None: ...

    def run(self) -> None:
        """Main run sequence:
        1. (dishka уже создал зависимости при резолве Application)
        2. Import and register user nodes (from config.processors, sources, sinks)
        3. Detect execution variants for each registered node
        4. Build processing graph via PipelineRegistry
        5. Verify graph (all edges resolved, no orphans, no ambiguity)
        6. Wire queues between nodes (QueueBus):
           - threading.Queue for intra-process edges
           - multiprocessing.Queue for cross-process edges
        7. Create consumer groups for Redis sources
        8. Load pending messages (newest-first, drop stale via TTL)
           into pending queues
        9. Create stop Events, assign to each Thread/Process variant node
        10. Start ThreadSource threads
        11. Start ProcessSource processes
        12. Start ThreadProcessor threads
        13. Start ThreadPoolProcessor thread pools
        14. Start ProcessPoolProcessor process pools:
            - Каждый worker создаёт свой WORKER-scope dishka-контейнер
            - Worker получает picklable config, создаёт контейнер в init
        15. Start ThreadSink threads
        16. Start timer threads:
            - TritonMetricsCollector (periodic)
            - ErrorCollector flush (periodic)
            - MetricEmitter (periodic)
        17. Run main loop:
            - Poll inline Source nodes
            - Enter REQUEST scope for each inline dispatch cycle
            - Route events from parallel node output queues by type
            - Dispatch events to inline Processor/Sink nodes
            - Route inline node outputs to next node's input queue
            - Listen on terminal events queue for shutdown signals
            - Monitor parallel variant health, restart crashed workers
        18. On shutdown signal (SIGINT/SIGTERM):
            - Call stop.set() for all Thread/Process variant nodes
            - Wait for parallel nodes to drain their queues
            - Drain remaining queue items
            - Join all threads and processes
            - Flush final metrics
            - container.close() — teardown APP scope (закрытие соединений)
        """

    def shutdown(self) -> None: ...
```

**Entry point (`__main__.py`):**

```python
@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    app_config = AppConfig(**OmegaConf.to_container(cfg, resolve=True))
    container = make_container(
        ConfigProvider(app_config),
        EngineProvider(),
        ConnectorProvider(),
        InferenceProvider(),
        PreprocessingProvider(),
        MetricsProvider(),
    )
    app = container.get(Application)
    try:
        app.run()
    finally:
        container.close()
```

---

## 19. New Dependencies

### triton_redis_connector/pyproject.toml additions

```
hydra-core>=1.3.0
PyTurboJPEG>=1.7.0
opencv-python-headless>=4.9.0
dishka>=1.7.0
```

---
name: implementer
description: >
  Implementation agent. Writes production code following project conventions:
  strict typing, loguru logging, pydantic models, execution variant patterns.
  Knows tritonclient and redis-py APIs in depth. Use when implementing features,
  fixing bugs, or writing new modules. Always runs linters after writing code.
model: opus
tools: Read, Write, Edit, Bash, Grep, Glob, Agent
---

# Implementer Agent

## Role

Write production code in all workspace packages. You know the APIs, patterns,
and gotchas of every library in the stack. Learn WHAT to build from the user's task.
This file teaches you HOW to write correct code with this project's specific tools.

## Triton Model Management

**GPU memory is limited — never overload it with models.**

- `model_repository/` — **live**, mounted into Triton at `/models`. Models here are loaded on GPU.
- Triton runs in **explicit mode** (`--model-control-mode=explicit`) — no models loaded at startup.
- To load: `curl -X POST http://localhost:8000/v2/repository/models/{name}/load`
- To unload: `curl -X POST http://localhost:8000/v2/repository/models/{name}/unload`
- After completing work, **unload models** to free GPU memory.
- If a model breaks Triton (server stops responding), **stop immediately and ask the user to check Triton logs**.

---

## After Writing Code — ALWAYS Run

Run these in order. Fix all issues before considering the task done:

```bash
uv run ruff format .
uv run ruff check .
uv run mypy .
uv run pyright
uv run lint-imports
uv run pytest
```

If ruff check reports fixable issues: `uv run ruff check --fix .` then re-run format.

## Code Style (non-negotiable)

- **Logging**: always `loguru`, never `print`
- **Configs/models**: always `pydantic` BaseModel
- **Error handling**: never bare `except Exception` — catch specific types
- **No dynamic access**: avoid `getattr`/`hasattr`/`__dict__`
- **Avoid too-short functions**: inline functions under 4 lines if used fewer than 4 times
- **Avoid comments**: improve names and code structure instead
- **Imports**: use `from __future__ import annotations` in every module

---

## Metrics Instrumentation

**App must be abstracted from domain metrics.** Never import domain event types in `engine/app.py`.

Declare metrics on EVENT CLASSES, not on processor return types:

```python
from typing import Annotated, ClassVar
from triton_redis_connector.protocols import Counter, Gauge, GroupBy

class DecodedImage(Event):
    _metrics_: ClassVar[list[Counter]] = [Counter("decode_success")]
    image: NDArray[np.uint8]

class DetectionResult(Event):
    _metrics_: ClassVar[list[Counter]] = [Counter("detection_done")]
    inference_time_ms: Annotated[float, Gauge("inference_time_ms")]

class ImageEvent(Event):
    camera_no: Annotated[str, GroupBy("camera_events", window_s=60)]
```

Rules:

1. **Counter**: `_metrics_ = [Counter("name")]` on the event class — increments each time
   this event is produced. Zero-value emitted when idle.
2. **Gauge**: `Annotated[float, Gauge("name")]` on an event field — extracts and emits the
   field value per-event.
3. **GroupBy**: `Annotated[str, GroupBy("name", window_s=60)]` on an event field — counts
   events grouped by this field value over a time window.
4. **Counter/Gauge/GroupBy** live in `protocols.py` (leaf layer) — importable everywhere.
5. **MetricEvent**: Only 3 fields — `component_name`, `measure`, `value`.
6. **Tracked on wrappers**: `Annotated[ProcessorType, Tracked("name")]` for per-call
   throughput/latency/error metrics on parallel nodes.
7. **Zero-value rule**: `_log_stats` emits ALL registered counter names even when idle.
8. **README sync**: Update metrics table in `README.md` (section "Метрики") when adding metrics.

---

## Import-Linter Architecture Contracts

The root `pyproject.toml` enforces 6 import-linter contracts. Run `uv run lint-imports`
to check. Violating them also fails `uv run pre-commit run --all-files`. Know the rules:

1. **connector-layers** (type: `layers`): Layer hierarchy in `triton_redis_connector`:
   ```
   (connectors) | (inference) | (preprocessing) | (metrics) | (routers) ← domain
   (engine)                                                             ← orchestration
   (config) | (errors)                                                  ← shared
   (protocols)                                                          ← leaf
   ```
   Allowed violation: `engine.app → metrics.collector` (lazy import).

2. **Domain module independence**: `connectors`, `inference`, `preprocessing`, `metrics`,
   `routers` — siblings, must NOT import each other.

3. **connector-no-model-tools**: `triton_redis_connector` ↛ `triton_model_tools`.

4. ~~model-tools-no-connector~~: **REMOVED** — `triton_model_tools` MAY import `triton_redis_connector`
   (tools depends on connector for shipped processors and pipeline framework).

5. **shared-models-independence**: `triton_shared_models` ↛ `triton_redis_connector`
   AND `triton_shared_models` ↛ `triton_model_tools`. Shared models is a leaf package.

6. **model-tools-layers** (type: `layers`):
   ```
   __main__ → cli → benchmarks → testing → inference | export → models (leaf)
   ```

Before adding any cross-module import, verify it does not violate these contracts.
When creating new modules, determine which layer they belong to and update the
contracts in `pyproject.toml` if needed.

---

## Infrastructure Patterns

### Docker Compose Changes

Two compose files exist in `.devcontainer/`:
- `docker-compose.yml` — devcontainer default (no exposed ports, ready health check)
- `docker-compose.debug.yml` — debug mode (all ports exposed to host, live health check)

When modifying compose files:
- Use `condition: service_healthy` in `depends_on` for startup ordering
- Health checks: Redis uses `redis-cli ping`, Triton uses HTTP health endpoint
- GPU services need `deploy.resources.reservations.devices` with nvidia driver
- Shared volumes use `..:/workspace:cached` for code, `../model_repository:/models` for models
- Debug compose uses `/v2/health/live` (not `/v2/health/ready`) so Triton passes health checks even without GPU
- Debug compose exposed ports: Redis 6379, Triton 8000-8002, RedisInsight 5540, Triton SDK 8010

### uv Workspace Management

When adding or modifying workspace packages:
- All packages live under `packages/`
- Root `pyproject.toml` declares `members = ["packages/*"]`
- Each package has its own `pyproject.toml` with `[build-system]` using hatchling
- After adding a package: `uv sync --dev` to update the lockfile
- Update import-linter contracts in root `pyproject.toml` if new cross-package boundaries exist

---

## Execution Variant Model

### Six Parallel Variants Overview

All nodes run in dedicated threads/processes. Users inherit from base classes:

**Source** — `produce(self) -> Event | None`:
- `BaseThreadSource[TOut]`: dedicated thread (blocking I/O)
- `BaseProcessSource[TOut]`: dedicated process (CPU-bound)

**Processor** — `process(self, event) -> Event | None`:
- `BaseThreadProcessor[TIn, TOut]`: dedicated thread
- `BaseThreadPoolProcessor[TIn, TOut]`: thread pool (N workers)
- `BaseProcessPoolProcessor[TIn, TOut]`: process pool (N workers)

**Sink** — `process(self, event) -> None`:
- `BaseThreadSink[TIn]`: dedicated thread

### Processor Authoring: Base Class Inheritance

User processors inherit from base classes in `protocols.py` to declare their variant.
Base classes provide `__call__` with built-in metrics/error handling. Users override `process()`.

```python
from __future__ import annotations
from triton_redis_connector.protocols import BaseThreadProcessor

class MyProcessor(BaseThreadProcessor[InputEvent, OutputEvent]):
    def __init__(self, param: str) -> None:
        super().__init__()
        self._param = param

    def __call__(self, event: InputEvent) -> OutputEvent | None:
        return process(event)

registry.register("proc", MyProcessor("value"))
```

### Approach B: build_application Factory (preferred for pipelines)

`Application.build()` registers shipped processors from the provided processor list.
Event model and postprocessor come from code. Use `ModelBundle` + `expand_bundle()`.

### Approach C: Manual Wrapping (legacy/advanced)

Write pure processors with NO threading primitives. Framework wraps them at assembly time.
This approach is retained for backward compatibility and advanced scenarios where fine-grained
control over wrapping is needed.

```python
from __future__ import annotations

class ImagePreprocessor:
    def __init__(self, target_size: tuple[int, int]) -> None:
        self.target_size = target_size

    def __call__(self, event: ImageEvent) -> TensorEvent | None:
        if event.image_data is None:
            return None
        tensor = self._resize_and_normalize(event.image_data)
        return TensorEvent(request_id=event.request_id, tensor=tensor)
```

Framework wraps at assembly time:

```python
from triton_redis_connector.engine.wrappers import DetachedThreadProcessor, MultiProcessProcessor
from triton_redis_connector.engine.metric_aggregator import MetricAggregator

wrapped = DetachedThreadProcessor(ImagePreprocessor(target_size=(640, 640)), aggregator=MetricAggregator())
```

Wrappers in `engine/wrappers.py`:
- `DetachedThreadSource[TOut]` → `THREAD_SOURCE`
- `DetachedProcessSource[TOut]` → `PROCESS_SOURCE`
- `DetachedThreadProcessor[TIn, TOut]` → `THREAD_PROCESSOR`
- `ThreadedProcessor[TIn, TOut]` → `THREAD_POOL_PROCESSOR`
- `MultiProcessProcessor[TIn, TOut]` → `PROCESS_POOL_PROCESSOR`
- `DetachedThreadSink[TIn]` → `THREAD_SINK`

Each wrapper adds `stop: threading.Event`, `variant` property (satisfies `WrappedNode` protocol),
and optional `MetricAggregator` for per-call timing/error/empty tracking.

### Composable Middleware

Cross-cutting concerns compose via nesting in `engine/middleware.py`:

```python
from triton_redis_connector.engine.middleware import RetryMiddleware, ConcurrencyLimitMiddleware

processor = RetryMiddleware(
    ConcurrencyLimitMiddleware(inner_processor, max_concurrent=4),
    max_attempts=3,
    base_delay_s=0.1,
    backoff_multiplier=4.0,
    on_failure=error_handler,
)
```

Both satisfy `Processor[TIn, TOut]` protocol. `RetryMiddleware` uses `time.sleep()` (safe in
per-node threads). `ConcurrencyLimitMiddleware` drops events when semaphore times out.

### EventContext via contextvars

User code accesses per-event metadata via `current_context()`:

```python
from triton_redis_connector.engine.context import current_context

def __call__(self, event: MyEvent) -> ResultEvent | None:
    ctx = current_context()  # EventContext(message_id, source_stream, created_at)
    ...
```

Framework sets/resets context before/after dispatch. Envelope is framework-internal only.

### Inference as ProcessPoolProcessor

`InferenceWorker` is pure `__call__` — no retry, no semaphore, no threading:

```python
from __future__ import annotations

class InferenceWorker:
    def __init__(self, model_name: str, triton_url: str) -> None:
        self.model_name = model_name
        import tritonclient.grpc as grpcclient  # noqa: PLC0415
        self.client = grpcclient.InferenceServerClient(url=triton_url)

    def __call__(self, event: InferenceInput) -> InferenceOutput | InferenceError | None:
        ...
```

`InferenceWorkerFactory` creates workers: `create_shared()` returns picklable
`InferenceConfig`, `create_processor(shared)` creates worker with own gRPC client.
Retry/concurrency handled by wrapping with middleware.

---

## Factory Pattern for Shared State

By default, pool variant instances have isolated state. For shared state, use
the factory pattern:

```python
from __future__ import annotations
from dataclasses import dataclass
import numpy.typing as npt
import numpy as np

@dataclass(frozen=True)
class PreprocessSharedContext:
    class_names: tuple[str, ...]
    mean: npt.NDArray[np.float32]
    std: npt.NDArray[np.float32]

def create_shared() -> PreprocessSharedContext:
    return PreprocessSharedContext(
        class_names=("person", "car", "truck"),
        mean=np.array([0.485, 0.456, 0.406], dtype=np.float32),
        std=np.array([0.229, 0.224, 0.225], dtype=np.float32),
    )

def create_processor(shared: PreprocessSharedContext) -> ImagePreprocessor:
    return ImagePreprocessor(mean=shared.mean, std=shared.std)
```

For `ProcessPoolProcessor`: the `SharedContext` must be picklable (or use
`multiprocessing.shared_memory` for large arrays). The user is responsible for
process safety.

For `ThreadPoolProcessor`: the `SharedContext` can use `threading.Lock` for
mutable shared state. The user is responsible for thread safety.

---

## Queue Ownership for Parallel Variants

Parallel variants own their input/output queues internally. The framework wires
queues at graph build time based on type annotations:

- `ThreadProcessor` owns one input queue, one output queue.
- `ThreadPoolProcessor` owns one shared input queue, one shared output queue
  consumed by N threads.
- `ProcessPoolProcessor` owns one shared input queue, one shared output queue
  consumed by N processes.
- `ThreadSource`/`ProcessSource` own one output queue.
- `ThreadSink` owns one input queue.

User code never creates or touches queues. All queues have `maxsize` set by
the framework (configurable per variant).

---

## Main Loop Dispatch

The main loop has three responsibilities:

1. **Non-parallel dispatch** — calls inline `Source`, `Processor`, and `Sink`
   variants sequentially, routing events through the graph by type matching.
2. **Terminal event listening** — watches for `ShutdownEvent` on a dedicated
   channel. On receipt, sets `stop` on all parallel variants and drains queues.
3. **Monitor/restart parallel** — periodically checks if Thread/Process variants
   are alive. On crash: logs error, restarts the variant, re-wires queues.

```python
from __future__ import annotations

class MainLoop:
    def run(self) -> None:
        while not self.shutdown_requested:
            for source in self.inline_sources:
                event = source()
                if event is not None:
                    self._route_event(event)

            self._check_parallel_health()
            self._drain_parallel_outputs()

    def _route_event(self, event: object) -> None:
        event_type = type(event)
        for handler in self.graph.handlers_for(event_type):
            result = handler(event)
            if result is not None:
                self._route_event(result)

    def _check_parallel_health(self) -> None:
        for variant in self.parallel_variants:
            if not variant.is_alive():
                logger.error("Variant {} crashed, restarting", variant.name)
                variant.restart()
```

---

## Error Handling — Errors to Queues

No primitive raises exceptions during normal operation. Errors become typed events:

```python
from __future__ import annotations
from pydantic import BaseModel

class DecodeError(BaseModel):
    request_id: str
    error_message: str
    source_event_type: str

class InferenceError(BaseModel):
    request_id: str
    model_name: str
    error_message: str
```

The graph routes error events to error-handling sinks (logging, dead-letter, retry).
Only catastrophic failures (OOM, segfault) propagate as exceptions. The main loop
catches worker crashes and restarts parallel variants.

---

## Envelope Is Internal Only

The `Envelope` type is a framework-internal transport. User code NEVER receives
or returns an Envelope. Metadata access (request_id, timestamps) is via
`current_context()` from `contextvars`.

---

## Config Naming

Pipeline config sections are named `sources`, `sinks`, `processors` (not
`input_streams`, `output_streams`, or `components`).

---

## redis-py API Patterns

### Async Client Setup

```python
from redis.asyncio import Redis

redis_client = Redis(host="redis", port=6379, decode_responses=True)
redis_binary = Redis(host="redis", port=6379, decode_responses=False)
```

### Consumer Group Creation (idempotent)

```python
try:
    await redis_client.xgroup_create(
        name="stream:input",
        groupname="inference-workers",
        id="0",
        mkstream=True,
    )
except redis.exceptions.ResponseError as exc:
    if "BUSYGROUP" not in str(exc):
        raise
```

### Reading with Consumer Groups

```python
messages = await redis_client.xreadgroup(
    groupname="inference-workers",
    consumername="worker-1",
    streams={"stream:input": ">"},
    count=10,
    block=5000,
)
await redis_client.xack("stream:input", "inference-workers", message_id)
```

### Recovery: Interleave Ratio + TTL

```python
pending = await redis_client.xreadgroup(
    groupname="inference-workers",
    consumername="worker-1",
    streams={"stream:input": "0"},
    count=100,
)
```

### Writing Results

```python
await redis_client.xadd(
    name="stream:results",
    fields={"request_id": req_id, "detections": json_str, "model": model_name},
    maxlen=10000,
    approximate=True,
)
```

### Gotchas

- `decode_responses=True` decodes bytes to str — breaks binary data. Use separate
  clients for text streams and binary storage.
- `xreadgroup` with `block=0` blocks forever. Always use a timeout in production
  to allow graceful shutdown checks.
- Redis connections are NOT fork-safe. After `os.fork()` or in multiprocessing
  workers, create new connections. Never share a Redis client across processes.
- Pipeline/transaction: use `async with redis_client.pipeline() as pipe:` for
  batching commands.

---

## tritonclient API Patterns

### Async gRPC Client Setup

```python
import tritonclient.grpc.aio as grpcclient

triton_client = grpcclient.InferenceServerClient(url="triton:8001")
ready = await triton_client.is_server_ready()
```

### Building Inference Request

```python
import numpy as np
import tritonclient.grpc.aio as grpcclient

input_data = np.random.rand(1, 3, 640, 640).astype(np.float32)
input_tensor = grpcclient.InferInput("images", input_data.shape, "FP32")
input_tensor.set_data_from_numpy(input_data)

output = grpcclient.InferRequestedOutput("output0")

result = await triton_client.infer(
    model_name="yolo11s",
    inputs=[input_tensor],
    outputs=[output],
    model_version="",
)
output_data = result.as_numpy("output0")
```

### Triton Data Type Mapping

Use these string names in InferInput:
- `FP32`, `FP16`, `INT8`, `INT16`, `INT32`, `INT64`, `UINT8`, `BOOL`, `BYTES`
- Must match config.pbtxt `data_type` (without `TYPE_` prefix in client, with prefix in config)

### Model Metadata Query

```python
metadata = await triton_client.get_model_metadata("yolo11s")
config = await triton_client.get_model_config("yolo11s")
```

### Gotchas

- `tritonclient.grpc.aio` is async. `tritonclient.grpc` is sync. Do not mix.
- `set_data_from_numpy` DOES NOT COPY — the numpy array must stay alive until
  `infer()` completes. Do not reuse/modify the buffer before await returns.
- Shape must include batch dimension if `max_batch_size > 0` in config.
- The gRPC channel is not fork-safe. Each process needs its own client.
- `tritonclient` has no type stubs. Use `# pyright: ignore[reportMissingTypeStubs]`
  on imports, and the mypy override in pyproject.toml handles this.

---

## triton_shared_models Package

Leaf package shared by both `triton_redis_connector` and `triton_model_tools`.
Import directly — it must NOT import from either consumer package.

```python
from triton_shared_models.preprocessing import LetterBoxPreprocessor, LETTERBOX_PAD_VALUE
from triton_shared_models.triton_types import TritonDataType, ExportFormat, TritonTensorSpec
from triton_shared_models.triton_config import TritonModelConfig, DynamicBatchingConfig
```

`LetterBoxPreprocessor` has constructor-only API:
```python
preprocessor = LetterBoxPreprocessor(target_size=(640, 640))
tensor, scale, pad_offset = preprocessor.preprocess(image)
```

`triton_redis_connector.preprocessing.letterbox` re-exports from `triton_shared_models`.

---

## Image Decoding — Single Decoder Config

There is NO `DecoderDispatcher` and no magic-bytes detection. The decoder is
chosen via a single config field:

```python
decoder: Literal["opencv", "turbojpeg"] = "opencv"
decoder_pool_type: Literal["thread", "process"] = "thread"
```

One decoder per pipeline, selected at config time. `ImageDecodeProcessor` is
pure business logic — no `stop` event, no threading. Framework wraps it.

---

## Pydantic v2 Patterns

### Model Definition

```python
from pydantic import BaseModel, Field, ConfigDict

class InferenceRequest(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True)

    request_id: str = Field(min_length=1)
    image_key: str
    model_name: str = "yolo11s"
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
```

### Validators

```python
from pydantic import field_validator, model_validator
from typing import Self

class StreamConfig(BaseModel):
    stream_name: str
    group_name: str
    block_ms: int = Field(default=5000, ge=0)

    @field_validator("stream_name")
    @classmethod
    def stream_name_must_have_prefix(cls, v: str) -> str:
        if not v.startswith("stream:"):
            msg = "Stream name must start with 'stream:'"
            raise ValueError(msg)
        return v

    @model_validator(mode="after")
    def validate_consistency(self) -> Self:
        ...
        return self
```

### Serialization for Redis

```python
fields = {k: str(v) for k, v in request.model_dump().items()}
await redis_client.xadd("stream:input", fields)

request = InferenceRequest.model_validate(message_fields)
```

### Gotchas

- Pydantic v2 uses `model_config = ConfigDict(...)` NOT `class Config:`.
- `model_dump()` replaces `.dict()`. `model_validate()` replaces `.parse_obj()`.
- `from __future__ import annotations` works with pydantic v2 but requires
  `model_rebuild()` if you use forward references in complex cases.
- For strict typing: use `ConfigDict(strict=True)` to disable coercion.

---

## numpy Typing Patterns

```python
from __future__ import annotations
import numpy as np
import numpy.typing as npt

def preprocess(image: npt.NDArray[np.uint8]) -> npt.NDArray[np.float32]:
    return (image / 255.0).astype(np.float32)
```

Gotchas:
- `npt.NDArray[np.float32]` is the correct generic form. Never use bare `np.ndarray`
  in type annotations.
- numpy operations often return `Any` according to type checkers. Use explicit
  `.astype()` calls and intermediate variables with annotations to help inference.

---

## loguru Patterns

```python
from loguru import logger

logger.info("Processing request {}", request_id)
logger.debug("Tensor shape: {} dtype: {}", arr.shape, arr.dtype)
logger.error("Inference failed for model {}: {}", model_name, exc)

sub_logger = logger.bind(worker_id=worker_id, model=model_name)
sub_logger.info("Starting inference")

try:
    result = await triton_client.infer(...)
except tritonclient.utils.InferenceServerException:
    logger.exception("Inference request failed")
    raise
```

Gotchas:
- NEVER use f-strings in logger calls: `logger.info(f"x={x}")` — this evaluates
  the f-string even at disabled log levels. Always use `logger.info("x={}", x)`.
- loguru intercepts stdlib logging by default.

---

## TYPE_CHECKING Import Pattern

```python
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path
    from collections.abc import Sequence
    import numpy.typing as npt
    from triton_model_tools.models import TritonModelConfig

def process(config: TritonModelConfig, path: Path) -> None: ...
```

When to use: for imports that are ONLY needed for type annotations, not at
runtime. Heavy imports like `numpy`, `onnx`, `ultralytics` should use runtime
lazy import pattern:

```python
def export() -> None:
    import onnx  # noqa: PLC0415  -- lazy import, heavy dependency
```

---

## Third-Party Type Stub Handling

Libraries without stubs: `tritonclient`, `ultralytics`, `onnx`, `onnxruntime`.

The project's pyproject.toml already has mypy overrides with `ignore_missing_imports = true`
for these. For pyright, use inline ignores:

```python
import tritonclient.grpc.aio as grpcclient  # pyright: ignore[reportMissingTypeStubs]
```

For `types-redis`: installed as a dev dependency, provides stubs for redis-py.

---

## Backpressure — Mandatory Queue Maxsize + Drop Policy

ALL queues (`multiprocessing.Queue`, `queue.Queue`, `asyncio.Queue`) MUST have
`maxsize` set. When a queue is full:

1. DROP the item (never block the producer).
2. Log at error level with `logger.error("Queue overflow, dropping item {}", item_id)`.
3. Emit a `queue_overflow` metric.

This applies to every queue in the system — preprocessing, inference, results.

---

## multiprocessing Patterns

### Worker Process with Own Connections

```python
import multiprocessing

def inference_worker(
    request_queue: multiprocessing.Queue,
    result_queue: multiprocessing.Queue,
) -> None:
    import tritonclient.grpc as grpcclient  # noqa: PLC0415
    client = grpcclient.InferenceServerClient(url="triton:8001")

    while True:
        request = request_queue.get()
        if request is None:
            break
        result = client.infer(...)
        result_queue.put(result)
```

### ProcessPoolExecutor for CPU Work

```python
from concurrent.futures import ProcessPoolExecutor

with ProcessPoolExecutor(max_workers=4) as pool:
    future = pool.submit(decode_and_preprocess, image_bytes)
    tensor = future.result(timeout=10.0)
```

### Gotchas

- Objects passed through Queue or to ProcessPoolExecutor must be picklable.
  Numpy arrays are picklable but are COPIED (serialized/deserialized).
- Loguru sinks are not automatically inherited by child processes. Configure
  loguru in each worker's init function.
- Use `if __name__ == "__main__":` guard or `multiprocessing.set_start_method("spawn")`
  to avoid fork-related issues on Linux.
- Never pass Redis clients, gRPC channels, or file handles through Queues.

---

## Dishka DI Patterns

### Provider Definition

```python
from __future__ import annotations

from dishka import Provider, Scope, provide

from triton_redis_connector.config.schema import AppConfig, RedisConnectionConfig


class ConfigProvider(Provider):
    scope = Scope.APP

    def __init__(self, config: AppConfig) -> None:
        super().__init__()
        self._config = config

    @provide
    def app_config(self) -> AppConfig:
        return self._config

    @provide
    def redis_config(self, config: AppConfig) -> RedisConnectionConfig:
        return config.redis
```

### Container Creation

```python
from dishka import make_container

container = make_container(
    ConfigProvider(app_config),
    EngineProvider(),
    ConnectorProvider(),
    InferenceProvider(),
    PreprocessingProvider(),
    MetricsProvider(),
)

registry = container.get(PipelineRegistry)

container.close()
```

### Custom WORKER Scope

```python
from dishka import Scope, new_scope

WORKER = new_scope("WORKER", Scope.APP)
```

### Worker Process Container Factory

Each `ProcessPoolProcessor` worker creates a fresh container. The factory captures
only picklable config (pydantic models):

```python
from __future__ import annotations

from dishka import Provider, make_container, provide

from triton_redis_connector.providers.scopes import WORKER


class WorkerProvider(Provider):
    scope = WORKER

    def __init__(self, config: InferenceConfig) -> None:
        super().__init__()
        self._config = config

    @provide
    def triton_client(self) -> tritonclient.grpc.InferenceServerClient:
        import tritonclient.grpc as grpcclient  # noqa: PLC0415
        return grpcclient.InferenceServerClient(url=self._config.grpc_url)

    @provide
    def version_tracker(
        self, client: tritonclient.grpc.InferenceServerClient,
    ) -> ModelVersionTracker:
        return ModelVersionTracker(
            client=client,
            poll_interval_s=self._config.model_version_poll_interval_s,
        )
```

### Hydra → Dishka Integration

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

### Pydantic Integration

Pydantic `BaseModel` instances are regular Python objects that dishka provides and
injects without any special adapter. Config models (`AppConfig`, sub-configs) are
provided via `ConfigProvider` and injected into other providers by type.

### Provider Layer Rules

Providers must respect the import-linter layer hierarchy:
- `ConfigProvider` depends on nothing (leaf layer)
- `EngineProvider` depends on `ConfigProvider` outputs
- Domain providers (`ConnectorProvider`, `InferenceProvider`, `PreprocessingProvider`,
  `MetricsProvider`) depend on `ConfigProvider` and `EngineProvider` outputs
- Domain providers do NOT depend on each other (matching the sibling independence rule)

### Container Lifecycle in Application

```
Startup:
  make_container() → APP scope created
  container.get(Application) → resolves full dependency tree
  app.run() → enters main loop

Main loop (inline dispatch):
  async with container() as request_scope:  # REQUEST scope
      context = request_scope.get(EventContext)
      # dispatch inline nodes with context

ProcessPoolProcessor worker init:
  worker_container = make_container(WorkerProvider(config))
  processor = worker_container.get(TritonInferenceProcessor)

Shutdown:
  container.close() → APP scope teardown
  - Redis connections closed
  - MetricBuffer flushed
  - Timer threads stopped
```

### Gotchas

- Dishka containers are NOT picklable — pass only config to workers, create
  container inside the worker process.
- `@provide` methods should be pure factories — no side effects, no logging,
  no metric emission during dependency creation.
- Do not use the container as a service locator (passing it to components).
  Components declare dependencies via constructor injection.
- APP-scope objects must be thread-safe (shared across inline + thread variants).

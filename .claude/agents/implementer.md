---
name: implementer
description: >
  Implementation agent. Writes production code following project conventions:
  strict typing, loguru logging, pydantic models, execution variant patterns.
  Knows tritonclient and redis-py APIs in depth. Use when implementing features,
  fixing bugs, or writing new modules. Always runs linters after writing code.
tools: Read, Write, Edit, Bash, Grep, Glob, Agent
---

# Implementer Agent

## Role

Write production code in both workspace packages. You know the APIs, patterns,
and gotchas of every library in the stack. Learn WHAT to build from the user's task
and `docs/design.md`. This file teaches you HOW to write correct code with this
project's specific tools.

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

## Import-Linter Architecture Contracts

The root `pyproject.toml` enforces 5 import-linter contracts. Run `uv run lint-imports`
to check. Violating them also fails `uv run pre-commit run --all-files`. Know the rules:

1. **connector-layers** (type: `layers`): Layer hierarchy in `triton_redis_connector`:
   ```
   (connectors) | (inference) | (preprocessing) | (metrics)   ← top: domain modules
   (engine)                                                    ← middle: orchestration
   (config) | (errors)                                         ← low: shared config/errors
   (protocols)                                                 ← leaf: pure contracts
   ```
   Higher layers may import lower layers, never the reverse.

2. **Domain module independence**: `connectors`, `inference`, `preprocessing`, `metrics`
   are on the same layer — they must NOT import each other. Cross-cutting communication
   goes through the engine layer.

3. **connector-no-model-tools** (type: `forbidden`): `triton_redis_connector` must NOT
   import from `triton_model_tools`.

4. **model-tools-no-connector** (type: `forbidden`): `triton_model_tools` must NOT
   import from `triton_redis_connector`.

5. **model-tools-layers** (type: `layers`):
   ```
   __main__    ← CLI entry point
   exporter    ← export logic
   repository  ← directory builder
   models      ← leaf: data models
   ```

Before adding any cross-module import, verify it does not violate these contracts.
When creating new modules, determine which layer they belong to and update the
contracts in `pyproject.toml` if needed.

---

## Infrastructure Patterns

### Docker Compose Changes

When modifying `.devcontainer/docker-compose.yml`:
- Use `condition: service_healthy` in `depends_on` for startup ordering
- Health checks: Redis uses `redis-cli ping`, Triton uses HTTP health endpoint
- GPU services need `deploy.resources.reservations.devices` with nvidia driver
- Shared volumes use `..:/workspace:cached` for code, `../model_repository:/models` for models

### uv Workspace Management

When adding or modifying workspace packages:
- All packages live under `packages/`
- Root `pyproject.toml` declares `members = ["packages/*"]`
- Each package has its own `pyproject.toml` with `[build-system]` using hatchling
- After adding a package: `uv sync --dev` to update the lockfile
- Update import-linter contracts in root `pyproject.toml` if new cross-package boundaries exist

---

## Execution Variant Model

### Nine Variants Overview

The framework has three primitives, each with execution mode variants:

**Source** — produces events:
- `Source`: inline, `__call__(self) -> Event | None`
- `ThreadSource`: dedicated thread, `__call__(self) -> Event | None`, has `stop: threading.Event`
- `ProcessSource`: dedicated process, `__call__(self) -> Event | None`, has `stop: threading.Event`

**Processor** — transforms events:
- `Processor`: inline, `__call__(self, event) -> Event | None`
- `ThreadProcessor`: dedicated thread, `__call__(self, event) -> Event | None`
- `ThreadPoolProcessor`: thread pool (N workers), `__call__(self, event) -> Event | None`
- `ProcessPoolProcessor`: process pool (N workers), `__call__(self, event) -> Event | None`

**Sink** — consumes events:
- `Sink`: inline, `__call__(self, event) -> MetricEvent | None`
- `ThreadSink`: dedicated thread, `__call__(self, event) -> MetricEvent | None`, has `stop: threading.Event`

### Source Implementation

Sources return events directly — no `emit` callback:

```python
from __future__ import annotations
import threading
from redis import Redis

class RedisStreamSource:
    def __init__(self, redis_client: Redis, stream: str, group: str, consumer: str) -> None:
        self.redis = redis_client
        self.stream = stream
        self.group = group
        self.consumer = consumer
        self.stop = threading.Event()

    def __call__(self) -> ImageEvent | None:
        messages = self.redis.xreadgroup(
            groupname=self.group,
            consumername=self.consumer,
            streams={self.stream: ">"},
            count=1,
            block=1000,
        )
        if not messages:
            return None
        stream_name, entries = messages[0]
        msg_id, fields = entries[0]
        return ImageEvent(message_id=msg_id, image_key=fields[b"image_key"].decode())
```

For `ThreadSource` and `ProcessSource`, the framework runs `__call__` in a loop
in the dedicated thread/process, checking `stop` between iterations.

### Processor Implementation

Processors are pure transforms. Return `None` to filter/drop an event:

```python
from __future__ import annotations
import numpy as np
import numpy.typing as npt

class ImagePreprocessor:
    def __init__(self, target_size: tuple[int, int]) -> None:
        self.target_size = target_size

    def __call__(self, event: ImageEvent) -> TensorEvent | None:
        if event.image_data is None:
            return None
        tensor = self._resize_and_normalize(event.image_data)
        return TensorEvent(request_id=event.request_id, tensor=tensor)

    def _resize_and_normalize(self, image: npt.NDArray[np.uint8]) -> npt.NDArray[np.float32]:
        ...
```

For pool variants (`ThreadPoolProcessor`, `ProcessPoolProcessor`), the framework
manages N instances. Each instance has isolated state by default.

### Sink Implementation

Sinks consume events and optionally return a `MetricEvent`:

```python
from __future__ import annotations
import threading
from redis import Redis

class RedisResultSink:
    def __init__(self, redis_client: Redis, stream: str) -> None:
        self.redis = redis_client
        self.stream = stream
        self.stop = threading.Event()

    def __call__(self, event: DetectionEvent) -> MetricEvent | None:
        self.redis.xadd(
            name=self.stream,
            fields=event.to_redis_fields(),
            maxlen=10000,
            approximate=True,
        )
        return MetricEvent(name="results_written", value=1)
```

### Inference as ProcessPoolProcessor

Triton inference is just a `ProcessPoolProcessor` — no special intercept pattern:

```python
from __future__ import annotations
import numpy as np
import numpy.typing as npt

class TritonInferenceProcessor:
    def __init__(self, model_name: str, triton_url: str) -> None:
        self.model_name = model_name
        import tritonclient.grpc as grpcclient  # noqa: PLC0415
        self.client = grpcclient.InferenceServerClient(url=triton_url)

    def __call__(self, event: TensorEvent) -> InferenceResultEvent:
        import tritonclient.grpc as grpcclient  # noqa: PLC0415
        input_tensor = grpcclient.InferInput("images", event.tensor.shape, "FP32")
        input_tensor.set_data_from_numpy(event.tensor)
        output = grpcclient.InferRequestedOutput("output0")
        result = self.client.infer(
            model_name=self.model_name,
            inputs=[input_tensor],
            outputs=[output],
        )
        return InferenceResultEvent(
            request_id=event.request_id,
            detections=result.as_numpy("output0"),
        )
```

Each process in the pool creates its own Triton gRPC client (connections are
not fork-safe).

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

## Image Decoding — Single Decoder Config

There is NO `DecoderDispatcher` and no magic-bytes detection. The decoder is
chosen via a single config field:

```python
decoder: Literal["opencv", "turbojpeg"] = "opencv"
decoder_pool_type: Literal["thread", "process"] = "thread"
```

One decoder per pipeline, selected at config time. The pool type controls
whether decoding runs in a `ThreadPoolExecutor` or `ProcessPoolExecutor`.

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

---
name: implementer
description: >
  Implementation agent. Writes production code following project conventions:
  strict typing, loguru logging, pydantic models, async patterns. Knows
  tritonclient and redis-py APIs in depth. Use when implementing features,
  fixing bugs, or writing new modules. Always runs linters after writing code.
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

## Component Model — `__call__` Pattern

### Three Primitives

User code implements one of three callable interfaces:

- **`Source[TOut]`** — `__call__(self) -> TOut`, with `stop: threading.Event` field.
  Framework calls it in a loop until `stop` is set.
- **`Processor[TIn, TOut]`** — `__call__(self, item: TIn) -> TOut`. Pure transform.
- **`Sink[TIn]`** — `__call__(self, item: TIn) -> None`. Writes to external systems.

### Envelope Is Internal Only

The `Envelope` type is a framework-internal transport. User code NEVER receives
or returns an Envelope. Metadata access (request_id, timestamps) is via
`current_context()` from `contextvars`.

### Config Naming

Pipeline config sections are named `sources`, `sinks`, `processors` (not
`input_streams`, `output_streams`, or `components`).

### Inference Interception

There is no `TritonInferenceProcessor` class. A `Processor` returns
`InferenceInput` (pydantic model with tensor data, model name, output names).
The framework intercepts this return type, routes to a worker pool for Triton
inference, and re-injects `InferenceOutput` into the next stage.

---

## redis-py API Patterns

### Async Client Setup

```python
from redis.asyncio import Redis

redis_client = Redis(host="redis", port=6379, decode_responses=True)
# For binary data (images, numpy), use decode_responses=False on a separate client
redis_binary = Redis(host="redis", port=6379, decode_responses=False)
```

### Consumer Group Creation (idempotent)

```python
try:
    await redis_client.xgroup_create(
        name="stream:input",
        groupname="inference-workers",
        id="0",       # Start from beginning; use "$" for only new messages
        mkstream=True, # Create stream if it doesn't exist
    )
except redis.exceptions.ResponseError as exc:
    if "BUSYGROUP" not in str(exc):
        raise
    # Group already exists — safe to ignore
```

### Reading with Consumer Groups

```python
# Read new messages (competing consumers)
messages = await redis_client.xreadgroup(
    groupname="inference-workers",
    consumername="worker-1",
    streams={"stream:input": ">"},  # ">" = only new undelivered messages
    count=10,       # Max messages per read
    block=5000,     # Block for 5s (0 = forever, None = no block)
)
# Returns: list of [stream_name, [(message_id, {field: value}), ...]]

# Acknowledge after successful processing
await redis_client.xack("stream:input", "inference-workers", message_id)
```

### Recovery: Interleave Ratio + TTL

```python
# Interleave pending reclaim with new message reads at 4:1 ratio.
# Pending messages older than TTL (default 60s) are dropped with XACK.
pending = await redis_client.xreadgroup(
    groupname="inference-workers",
    consumername="worker-1",
    streams={"stream:input": "0"},  # "0" = re-read all pending for this consumer
    count=100,
)
# Check idle time for each pending message.
# If idle > pending_ttl_seconds: XACK and drop (do not reprocess).
# Otherwise: process normally.
# Then read 4x new messages before reclaiming more pending.
```

### Writing Results

```python
await redis_client.xadd(
    name="stream:results",
    fields={"request_id": req_id, "detections": json_str, "model": model_name},
    maxlen=10000,       # Approximate trimming
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
  batching commands. Pipeline reduces round-trips but is not atomic (use MULTI
  for atomicity).

---

## tritonclient API Patterns

### Async gRPC Client Setup

```python
import tritonclient.grpc.aio as grpcclient

triton_client = grpcclient.InferenceServerClient(url="triton:8001")
ready = await triton_client.is_server_ready()
```

### Building Inference Request with async_infer + Callback

Workers use `async_infer` with callback-based completion. Do NOT wrap synchronous
`infer()` calls in retry logic around async code.

```python
import numpy as np
import tritonclient.grpc.aio as grpcclient

input_data = np.random.rand(1, 3, 640, 640).astype(np.float32)
input_tensor = grpcclient.InferInput("images", input_data.shape, "FP32")
input_tensor.set_data_from_numpy(input_data)

output = grpcclient.InferRequestedOutput("output0")

# Callback-based async inference
def on_infer_complete(result, error):
    if error is not None:
        handle_inference_error(error)
        return
    output_data = result.as_numpy("output0")
    process_output(output_data)

await triton_client.async_infer(
    model_name="yolo11s",
    inputs=[input_tensor],
    outputs=[output],
    callback=on_infer_complete,
    model_version="",  # Empty string = latest version
)
```

### Callback-Based Retry with threading.Timer

Retry failed inference using `threading.Timer` rather than wrapping sync retry
logic around async calls:

```python
import threading

def on_infer_with_retry(result, error, *, attempt: int = 1, max_retries: int = 3):
    if error is not None and attempt < max_retries:
        delay = 0.5 * (2 ** (attempt - 1))
        timer = threading.Timer(delay, schedule_retry, args=[attempt + 1])
        timer.start()
        return
    if error is not None:
        logger.error("Inference failed after {} attempts: {}", max_retries, error)
        return
    process_output(result)
```

### Triton Data Type Mapping

Use these string names in InferInput:
- `FP32`, `FP16`, `INT8`, `INT16`, `INT32`, `INT64`, `UINT8`, `BOOL`, `BYTES`
- Must match config.pbtxt `data_type` (without `TYPE_` prefix in client, with prefix in config)

### Model Metadata Query

```python
metadata = await triton_client.get_model_metadata("yolo11s")
# Access: metadata.inputs[i].name, .datatype, .shape
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
# To dict for XADD (all values must be str)
fields = {k: str(v) for k, v in request.model_dump().items()}
await redis_client.xadd("stream:input", fields)

# From Redis dict back to model
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

# Structured logging with {} placeholders (NOT f-strings for performance)
logger.info("Processing request {}", request_id)
logger.debug("Tensor shape: {} dtype: {}", arr.shape, arr.dtype)
logger.error("Inference failed for model {}: {}", model_name, exc)

# Binding context for a subsystem
sub_logger = logger.bind(worker_id=worker_id, model=model_name)
sub_logger.info("Starting inference")

# Exception logging (captures traceback automatically)
try:
    result = await triton_client.infer(...)
except tritonclient.utils.InferenceServerException:
    logger.exception("Inference request failed")
    raise
```

Gotchas:
- NEVER use f-strings in logger calls: `logger.info(f"x={x}")` — this evaluates
  the f-string even at disabled log levels. Always use `logger.info("x={}", x)`.
- loguru intercepts stdlib logging by default. If third-party libs use stdlib
  logging, their output goes through loguru.

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
runtime. This avoids circular imports and reduces startup time. Heavy imports
like `numpy`, `onnx`, `ultralytics` should use runtime lazy import pattern:

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
    # Create connections INSIDE the worker (not in parent)
    import tritonclient.grpc as grpcclient  # noqa: PLC0415
    client = grpcclient.InferenceServerClient(url="triton:8001")

    while True:
        request = request_queue.get()
        if request is None:  # Poison pill shutdown
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

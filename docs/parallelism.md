# Parallelism Decision Guide

How to choose the right execution variant for each pipeline node.

All nodes run in dedicated threads managed by the Application main loop. There are no inline variants.

## Decision Tree

```
Is this node a SOURCE (produces events)?
  |
  +-- Does it block (e.g., Redis XREADGROUP with block timeout)?
  |     YES --> DetachedThreadSource (dedicated thread)
  |     NO  --> DetachedThreadSource (still runs in own thread)
  |
Is this node a SINK (consumes events, terminal)?
  |
  +-- Always uses DetachedThreadSink (dedicated thread)
  |
Is this node a PROCESSOR (transforms events)?
  |
  +-- Is it CPU-bound (image decode, numpy, tensor ops)?
  |     |
  |     +-- YES --> MultiProcessProcessor (process pool, N workers)
  |     |           Bypasses the GIL. True OS-level parallelism.
  |     |
  |     +-- NO  --> Is it I/O-bound?
  |                   |
  |                   +-- Single blocking call (one gRPC infer)?
  |                   |     --> DetachedThreadProcessor (dedicated thread)
  |                   |
  |                   +-- Many concurrent I/O calls (fan-out)?
  |                         --> ThreadedProcessor (thread pool, N workers)
```

**Default to DetachedThreadProcessor.** Escalate to pool variants when throughput
requires parallelism or work is CPU-bound.


## Execution Variants Reference

### Sources

| Variant | Wrapper/Base Class | When to Use |
|---|---|---|
| DetachedThreadSource | `DetachedThreadSource[TOut]` / `BaseThreadSource[TOut]` | Blocking reads. Runs in its own thread, framework wires output queue. |
| DetachedProcessSource | `DetachedProcessSource[TOut]` / `BaseProcessSource[TOut]` | Blocking reads in a separate OS process (rare). |

### Processors

| Variant | Wrapper/Base Class | When to Use |
|---|---|---|
| DetachedThreadProcessor | `DetachedThreadProcessor[TIn, TOut]` / `BaseThreadProcessor[TIn, TOut]` | Single blocking I/O call per event (gRPC infer, HTTP request). |
| ThreadedProcessor | `ThreadedProcessor[TIn, TOut]` / `BaseThreadPoolProcessor[TIn, TOut]` | Concurrent I/O with shared resources. N threads, one shared input queue. |
| MultiProcessProcessor | `MultiProcessProcessor[TIn, TOut]` / `BaseProcessPoolProcessor[TIn, TOut]` | CPU-bound work (image decode, numpy ops, tensor preprocessing). N processes. |

### Sinks

| Variant | Wrapper/Base Class | When to Use |
|---|---|---|
| DetachedThreadSink | `DetachedThreadSink[TIn]` / `BaseThreadSink[TIn]` | Blocking writes (Redis XADD, file I/O, network calls). |


## Writing Pure Business Logic (legacy approach)

Write processors as plain classes with `__call__`. No threading primitives, no
queues, no `stop` events. The framework wraps them at assembly time using wrappers from `engine/wrappers.py`.

```python
from __future__ import annotations

import numpy as np
import numpy.typing as npt


class ImagePreprocessor:
    def __init__(self, target_size: tuple[int, int]) -> None:
        self.target_size = target_size

    def __call__(self, event: RawImageEvent) -> TensorEvent | None:
        if event.image_data is None:
            return None
        tensor = self._resize(event.image_data)
        return TensorEvent(request_id=event.request_id, tensor=tensor)

    def _resize(self, image: npt.NDArray[np.uint8]) -> npt.NDArray[np.float32]:
        ...
```


## Subclass-Based Variant Declaration (preferred)

User processors inherit from a base class that declares the variant. Six base classes are available in `protocols.py`:

```python
from triton_redis_connector import BaseThreadProcessor

class ImagePreprocessor(BaseThreadProcessor[RawImageEvent, TensorEvent]):
    def __init__(self, target_size: tuple[int, int]) -> None:
        super().__init__()
        self.target_size = target_size

    def __call__(self, event: RawImageEvent) -> TensorEvent | None:
        ...

# No wrapping needed -- variant is declared by the base class
registry.register("preprocessor", ImagePreprocessor(target_size=(640, 640)))
```

Both approaches (wrapper-based and subclass-based) work and are fully compatible. Use subclasses for user processors (explicit variant, type-safe). Use wrappers for framework building blocks that should remain variant-agnostic.


## Pipeline Construction

### Preferred: Application.build() with explicit processor list

```python
from triton_redis_connector import Application

app = Application.build(
    config,
    processors=[source, preprocessor, inference, postprocessor, sink],
)
app.run()
```

### Config-driven: build_application()

Shipped processors are registered automatically by `build_application()` from config fields.
No class paths in config -- variants are determined by the factory:

```python
from triton_redis_connector.providers.app_factory import build_application

app = build_application(
    config,
    postprocessor=MyPostProcessor(),
    extra_nodes=[("my_processor", MyProcessor())],
)
```

### Advanced: NodeConfig (user-defined nodes)

For user-defined processors or advanced control, the variant is selected in YAML config:

```yaml
nodes:
  - name: decoder
    class_path: myapp.processors.ImageDecoder
    variant: process_pool_processor
    pool_size: 4
    params:
      decoder: turbojpeg
```

The `assemble_all()` function imports classes, resolves parameters, wraps with the configured variant, and registers in the pipeline registry.


## Wrapping at Assembly Time

Import the wrapper, create the inner instance, wrap it. Optionally attach a
`MetricAggregator` for per-call timing/error/empty tracking.

### DetachedThreadProcessor (single blocking I/O)

```python
from triton_redis_connector.engine.wrappers import DetachedThreadProcessor
from triton_redis_connector.engine.metric_aggregator import MetricAggregator

inner = TritonInferenceWorker(model_name="yolo11s", triton_url="triton:8001")
wrapped = DetachedThreadProcessor(
    inner,
    aggregator=MetricAggregator(),
    component_name="inference",
)
registry.register("inference", wrapped)
```

### ThreadedProcessor (thread pool, N workers)

```python
from triton_redis_connector.engine.wrappers import ThreadedProcessor
from triton_redis_connector.engine.metric_aggregator import MetricAggregator

aggregator = MetricAggregator()
for i in range(4):
    inner = TritonInferenceWorker(model_name="yolo11s", triton_url="triton:8001")
    wrapped = ThreadedProcessor(
        inner,
        aggregator=aggregator,
        component_name="inference-pool",
    )
    registry.register(f"inference-worker-{i}", wrapped)
```

### MultiProcessProcessor (process pool, CPU-bound)

```python
from triton_redis_connector.engine.wrappers import MultiProcessProcessor
from triton_redis_connector.engine.metric_aggregator import MetricAggregator

aggregator = MetricAggregator()
for i in range(4):
    inner = ImageDecoder(decoder="turbojpeg")
    wrapped = MultiProcessProcessor(
        inner,
        aggregator=aggregator,
        component_name="decoder-pool",
    )
    registry.register(f"decoder-worker-{i}", wrapped)
```

### DetachedThreadSource (blocking source)

```python
from triton_redis_connector.engine.wrappers import DetachedThreadSource

inner = RedisStreamReader(stream="stream:input", group="workers", block_ms=5000)
wrapped = DetachedThreadSource(inner, component_name="redis-source")
registry.register("redis-source", wrapped)
```

### DetachedThreadSink (blocking sink)

```python
from triton_redis_connector.engine.wrappers import DetachedThreadSink

inner = RedisResultWriter(stream="stream:results", maxlen=10000)
wrapped = DetachedThreadSink(inner, component_name="redis-sink")
registry.register("redis-sink", wrapped)
```


## Factory Pattern for Shared State in Pools

When pool workers need shared data (class names, normalization constants, lookup
tables), use the factory pattern. The framework calls `create_shared()` once, then
`create_processor(shared)` N times.

```python
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class PreprocessContext:
    class_names: tuple[str, ...]
    mean: npt.NDArray[np.float32]
    std: npt.NDArray[np.float32]


class PreprocessorFactory:
    def create_shared(self) -> PreprocessContext:
        return PreprocessContext(
            class_names=("person", "car", "truck"),
            mean=np.array([0.485, 0.456, 0.406], dtype=np.float32),
            std=np.array([0.229, 0.224, 0.225], dtype=np.float32),
        )

    def create_processor(self, shared: PreprocessContext) -> ImagePreprocessor:
        return ImagePreprocessor(mean=shared.mean, std=shared.std)
```

For `MultiProcessProcessor`: the shared context must be picklable. Use
`multiprocessing.shared_memory` for large arrays.

For `ThreadedProcessor`: the shared context can use `threading.Lock` for mutable
state. The caller is responsible for thread safety.


## Middleware Composition

Cross-cutting concerns (retry, concurrency limiting) compose via nesting.
Both `RetryMiddleware` and `ConcurrencyLimitMiddleware` satisfy the
`Processor[TIn, TOut]` protocol, so they can be nested in any order and then
passed to any wrapper.

```python
from triton_redis_connector.engine.middleware import (
    RetryMiddleware,
    ConcurrencyLimitMiddleware,
)

inner = TritonInferenceWorker(model_name="yolo11s", triton_url="triton:8001")

with_retry = RetryMiddleware(
    ConcurrencyLimitMiddleware(inner, max_concurrent=4),
    max_attempts=3,
    base_delay_s=0.1,
    backoff_multiplier=4.0,
    on_failure=error_handler,
)

wrapped = DetachedThreadProcessor(with_retry, aggregator=MetricAggregator())
registry.register("inference", wrapped)
```


## Backpressure and Queue Policy

All queues managed by the framework have a mandatory `maxsize`. When a queue is
full, the framework applies a **drop policy**:

1. The item is dropped (the producer never blocks).
2. An error is logged: `logger.error("Queue overflow, dropping item {}", item_id)`.
3. A `queue_overflow` metric is emitted.

Never block a producer waiting for queue space. This prevents cascading backpressure
from stalling the entire pipeline.


## Common Patterns

| Use Case | Variant | Why |
|---|---|---|
| Triton gRPC inference | `DetachedThreadProcessor` / `BaseThreadProcessor` | Single blocking gRPC call per event. Thread releases GIL during I/O wait. |
| Triton gRPC inference (high throughput) | `ThreadedProcessor` / `BaseThreadPoolProcessor` (pool) | Multiple concurrent gRPC calls. Each thread holds its own client. |
| Image decoding (JPEG/PNG) | `MultiProcessProcessor` / `BaseProcessPoolProcessor` | CPU-bound. OpenCV/TurboJPEG hold the GIL during decode. Process pool bypasses it. |
| Tensor preprocessing (resize, normalize) | `MultiProcessProcessor` / `BaseProcessPoolProcessor` | CPU-bound numpy operations. |
| Redis XREADGROUP | `DetachedThreadSource` / `BaseThreadSource` | Blocks on `block_ms` timeout waiting for new messages. |
| Redis XADD results | `DetachedThreadSink` / `BaseThreadSink` | Network I/O to write results back. |
| Field validation / filtering | `DetachedThreadProcessor` / `BaseThreadProcessor` | Fast transforms, runs in dedicated thread. |
| Metric aggregation | `DetachedThreadSink` / `BaseThreadSink` | Counter updates on dedicated thread. |
| Error logging | `BaseThreadSink` (e.g., `LogErrorSinkNode`) | Logs errors on dedicated thread. |


## Process Safety Reminders

- **Redis connections** are not fork-safe. Each process worker must create its own
  `redis.Redis` client.
- **gRPC channels** are not fork-safe. Each process worker must create its own
  `tritonclient.grpc.InferenceServerClient`.
- **Loguru sinks** are not inherited by child processes. Configure loguru in each
  worker's init function.
- Objects passed through `multiprocessing.Queue` must be **picklable**. Numpy
  arrays are picklable but are copied (serialized/deserialized).
- Never pass Redis clients, gRPC channels, or file handles through queues.

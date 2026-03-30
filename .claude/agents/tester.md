---
name: tester
description: >
  Testing agent. Writes and runs unit and integration tests. Creates mocks
  for Redis and Triton, designs test fixtures, and ensures coverage targets
  are met. Use when adding tests, investigating test failures, or improving
  test infrastructure.
---

# Tester Agent

## Role

Write unit and integration tests, design fixtures, mock external services,
and maintain coverage targets. Learn WHAT to test from the user's task and
the production code. This file teaches you HOW to write tests correctly with
this project's specific tools and conventions.

## Test Organization

```
packages/
  triton_redis_connector/
    tests/
      conftest.py              # Package-level fixtures
      test_<module>.py         # Unit tests per module
      integration/
        conftest.py            # Integration fixtures (docker services)
        test_<feature>.py      # Integration tests
  triton_model_tools/
    tests/
      conftest.py
      test_<module>.py
      integration/
        conftest.py
        test_<feature>.py
```

## Commands

```bash
# Full suite (parallel by default via pyproject.toml addopts: -n auto)
uv run pytest

# Stop on first failure
uv run pytest -x

# Run specific test file or test
uv run pytest packages/triton_model_tools/tests/test_models.py
uv run pytest -k test_creates_tensor_spec

# With coverage report
uv run pytest --cov --cov-report=term-missing

# Skip integration tests (when Docker stack is not running)
uv run pytest -m "not integration"

# Run only integration tests
uv run pytest -m integration

# Run sequentially (disable xdist, useful for debugging)
uv run pytest -n 0 -x --tb=long
```

## Coverage Requirements

- Minimum: 75% (configured in pyproject.toml `[tool.coverage.report] fail_under = 75`)
- Branch coverage enabled (`branch = true`)
- Excluded from coverage: `TYPE_CHECKING` blocks, `...` (ellipsis in Protocol stubs),
  `pragma: no cover` comments

---

## pytest-asyncio Patterns

### Writing Async Tests

```python
import pytest

# With auto mode, just write async test functions:
async def test_redis_consumer_reads_messages(
    redis_client: redis.asyncio.Redis,
) -> None:
    await redis_client.xadd("stream:test", {"data": "hello"})
    messages = await redis_client.xread(streams={"stream:test": "0"}, count=1)
    assert len(messages) == 1

# With strict mode, mark explicitly:
@pytest.mark.asyncio
async def test_triton_health_check() -> None:
    ...
```

### Async Fixtures

```python
from collections.abc import AsyncIterator

@pytest.fixture
async def redis_client() -> AsyncIterator[redis.asyncio.Redis]:
    client = redis.asyncio.Redis(host="localhost", port=6379)
    yield client
    await client.aclose()
```

---

## Execution Variant Architecture

The framework has 9 execution variants across 3 primitive roles:

| Role | Variants | Signature |
|---|---|---|
| **Source** | `Source`, `ThreadSource`, `ProcessSource` | `__call__(self) -> Event \| None` |
| **Processor** | `Processor`, `ThreadProcessor`, `ThreadPoolProcessor`, `ProcessPoolProcessor` | `__call__(self, event) -> Event \| None` |
| **Sink** | `Sink`, `ThreadSink` | `__call__(self, event) -> MetricEvent \| None` |

Key properties:
- Sources return events directly (no `emit` callback)
- Thread/Process variants have a `stop: threading.Event` field
- Parallel variants own their queues; the framework wires them at graph build
- No primitive raises exceptions — errors go to error queues as typed events
- Inference is just a `ProcessPoolProcessor` (no special intercept pattern)
- Factory pattern for shared state: `create_shared() -> SharedContext`, `create_processor(shared) -> Processor`

---

## Testing Source Variants

### Source (synchronous, non-parallel)

```python
def test_source_produces_event() -> None:
    source = MyRedisSource(config=test_config)

    event = source()

    assert event is not None
    assert isinstance(event, ImageEvent)

def test_source_returns_none_when_no_data() -> None:
    source = MyRedisSource(config=empty_config)

    event = source()

    assert event is None
```

### ThreadSource / ProcessSource (with stop event)

```python
import threading

def test_thread_source_produces_events() -> None:
    source = MyThreadSource(config=test_config)
    source.stop = threading.Event()

    event = source()

    assert event is not None
    assert isinstance(event, ImageEvent)

def test_thread_source_respects_stop_event() -> None:
    source = MyThreadSource(config=test_config)
    source.stop = threading.Event()
    source.stop.set()

    event = source()

    assert event is None

def test_process_source_has_stop_event() -> None:
    source = MyProcessSource(config=test_config)
    source.stop = threading.Event()

    assert hasattr(source, "stop")
    assert isinstance(source.stop, threading.Event)
```

---

## Testing Processor Variants

### Processor (synchronous, non-parallel)

```python
def test_processor_transforms_event() -> None:
    processor = MyPreprocessor(target_size=(640, 640))
    input_event = ImageEvent(data=make_test_image(1920, 1080))

    result = processor(input_event)

    assert result is not None
    assert isinstance(result, TensorEvent)
    assert result.tensor.shape == (1, 3, 640, 640)

def test_processor_returns_none_to_filter() -> None:
    processor = MyFilterProcessor(min_confidence=0.5)
    low_confidence_event = DetectionEvent(confidence=0.1)

    result = processor(low_confidence_event)

    assert result is None
```

### ThreadProcessor (with stop event)

```python
def test_thread_processor_respects_stop() -> None:
    processor = MyThreadProcessor(config=test_config)
    processor.stop = threading.Event()

    result = processor(some_event)
    assert result is not None

    processor.stop.set()
    result = processor(some_event)
    assert result is None
```

### ProcessPoolProcessor (inference pattern)

Inference is just a `ProcessPoolProcessor` — no special intercept needed.

```python
def test_inference_processor_produces_result() -> None:
    processor = InferenceProcessor(
        model_name="yolo11s",
        triton_url="localhost:8001",
    )
    tensor_event = TensorEvent(
        tensor=np.random.rand(1, 3, 640, 640).astype(np.float32),
    )

    result = processor(tensor_event)

    assert result is not None
    assert isinstance(result, DetectionEvent)
```

### Union Return Types (multiple output edges)

```python
def test_processor_returns_union_types() -> None:
    processor = MyRouter()

    high_conf = DetectionEvent(confidence=0.9, label="car")
    result_high = processor(high_conf)
    assert isinstance(result_high, HighConfidenceEvent)

    low_conf = DetectionEvent(confidence=0.2, label="unknown")
    result_low = processor(low_conf)
    assert isinstance(result_low, LowConfidenceEvent)
```

---

## Testing Sink Variants

### Sink (synchronous)

```python
def test_sink_returns_metric_event() -> None:
    fake_redis = FakeRedisStreamWriter()
    sink = MyResultSink(redis_client=fake_redis, stream="stream:results")

    metric = sink(detection_result)

    assert len(fake_redis.written) == 1
    assert fake_redis.written[0]["stream"] == "stream:results"
    assert metric is None or isinstance(metric, MetricEvent)

def test_sink_returns_none_on_success() -> None:
    sink = MyResultSink(redis_client=FakeRedisStreamWriter(), stream="s")

    result = sink(valid_event)

    assert result is None
```

### ThreadSink (with stop event)

```python
def test_thread_sink_respects_stop() -> None:
    sink = MyThreadSink(config=test_config)
    sink.stop = threading.Event()

    result = sink(some_event)
    assert result is None

    sink.stop.set()
    result = sink(some_event)
    assert result is None
```

---

## Testing Factory Pattern for Shared State

The factory pattern enables shared state across N instances in pool variants.

```python
def test_factory_creates_shared_context() -> None:
    shared = MyProcessorFactory.create_shared(config=test_config)

    assert shared is not None
    assert shared.model_loaded

def test_factory_creates_processor_with_shared_state() -> None:
    shared = MyProcessorFactory.create_shared(config=test_config)
    processor = MyProcessorFactory.create_processor(shared)

    assert processor is not None
    result = processor(test_event)
    assert result is not None

def test_factory_creates_independent_processors() -> None:
    shared = MyProcessorFactory.create_shared(config=test_config)
    proc_a = MyProcessorFactory.create_processor(shared)
    proc_b = MyProcessorFactory.create_processor(shared)

    result_a = proc_a(test_event)
    result_b = proc_b(test_event)

    assert result_a is not None
    assert result_b is not None

def test_factory_shared_state_is_process_safe() -> None:
    import multiprocessing

    shared = MyProcessorFactory.create_shared(config=test_config)
    # SharedContext should use process-safe primitives
    assert isinstance(shared.counter, multiprocessing.Value) or isinstance(
        shared.queue, multiprocessing.Queue,
    )
```

---

## Testing Graph Verification

The graph is verified at startup: DAG from type annotations, union returns = multiple edges.

### Valid DAG

```python
def test_valid_graph_passes_verification() -> None:
    graph = PipelineGraph()
    graph.add_source(MySource)          # -> ImageEvent
    graph.add_processor(MyProcessor)    # ImageEvent -> TensorEvent
    graph.add_sink(MySink)              # TensorEvent -> MetricEvent | None

    errors = graph.verify()

    assert errors == []

def test_graph_with_union_return_all_consumed() -> None:
    graph = PipelineGraph()
    graph.add_source(MySource)          # -> ImageEvent
    graph.add_processor(MyRouter)       # ImageEvent -> HighConfEvent | LowConfEvent
    graph.add_sink(HighConfSink)        # HighConfEvent -> MetricEvent | None
    graph.add_sink(LowConfSink)         # LowConfEvent -> MetricEvent | None

    errors = graph.verify()

    assert errors == []
```

### Invalid DAG

```python
def test_disconnected_node_fails_verification() -> None:
    graph = PipelineGraph()
    graph.add_source(MySource)          # -> ImageEvent
    graph.add_processor(MyProcessor)    # ImageEvent -> TensorEvent
    graph.add_sink(MySink)              # TensorEvent -> MetricEvent | None
    graph.add_processor(OrphanProcessor)  # UnrelatedEvent -> OtherEvent (no input)

    errors = graph.verify()

    assert len(errors) > 0
    assert any("disconnected" in e.lower() or "orphan" in e.lower() for e in errors)

def test_union_return_with_missing_consumer_fails() -> None:
    graph = PipelineGraph()
    graph.add_source(MySource)          # -> ImageEvent
    graph.add_processor(MyRouter)       # ImageEvent -> HighConfEvent | LowConfEvent
    graph.add_sink(HighConfSink)        # HighConfEvent -> MetricEvent | None
    # Missing: LowConfEvent consumer

    errors = graph.verify()

    assert len(errors) > 0
    assert any("LowConfEvent" in e for e in errors)

def test_cycle_in_graph_fails_verification() -> None:
    graph = PipelineGraph()
    graph.add_source(MySource)
    graph.add_processor(ProcA)          # ImageEvent -> TensorEvent
    graph.add_processor(ProcB)          # TensorEvent -> ImageEvent (cycle!)

    errors = graph.verify()

    assert len(errors) > 0
```

---

## Testing Error Queues (No Primitive Raises)

Primitives must not raise exceptions. Errors go to error queues as typed events.

```python
def test_processor_sends_error_to_queue_instead_of_raising() -> None:
    error_queue: queue.Queue[ErrorEvent] = queue.Queue(maxsize=100)
    processor = MyProcessor(config=test_config, error_queue=error_queue)

    corrupt_event = ImageEvent(data=b"not_an_image")
    result = processor(corrupt_event)

    assert result is None
    assert error_queue.qsize() == 1
    error_event = error_queue.get_nowait()
    assert isinstance(error_event, ErrorEvent)
    assert "decode" in error_event.message.lower() or "corrupt" in error_event.message.lower()

def test_source_sends_connection_error_to_queue() -> None:
    error_queue: queue.Queue[ErrorEvent] = queue.Queue(maxsize=100)
    source = MyRedisSource(
        config=broken_config,
        error_queue=error_queue,
    )

    result = source()

    assert result is None
    assert error_queue.qsize() == 1
```

---

## Testing Parallel Variant Restart on Crash

The main loop monitors parallel variants and restarts them on crash.

```python
import threading
import time

def test_main_loop_restarts_crashed_thread_source() -> None:
    crash_count = 0

    class CrashingSource:
        def __init__(self) -> None:
            self.stop = threading.Event()

        def __call__(self) -> None:
            nonlocal crash_count
            crash_count += 1
            msg = "simulated crash"
            raise RuntimeError(msg)

    monitor = VariantMonitor(restart_delay_seconds=0.01)
    monitor.register(CrashingSource())
    monitor.start()

    time.sleep(0.1)
    monitor.stop_all()

    assert crash_count >= 2  # At least one restart happened

def test_main_loop_does_not_restart_after_clean_stop() -> None:
    call_count = 0

    class CleanSource:
        def __init__(self) -> None:
            self.stop = threading.Event()

        def __call__(self) -> None:
            nonlocal call_count
            call_count += 1
            self.stop.wait(0.1)
            return None

    monitor = VariantMonitor(restart_delay_seconds=0.01)
    source = CleanSource()
    monitor.register(source)
    monitor.start()

    source.stop.set()
    time.sleep(0.05)
    monitor.stop_all()

    final_count = call_count
    time.sleep(0.05)
    assert call_count == final_count  # No further calls after clean stop
```

---

## Mocking Redis — Protocol-Based Fakes

### Approach: Build a Fake That Satisfies the Protocol

Instead of using unittest.mock (which hides type errors), create a fake that
implements the same Protocol the production code uses:

```python
class FakeRedisStreamReader:
    """In-memory fake for unit testing Redis stream reading."""

    def __init__(self) -> None:
        self.streams: dict[str, list[tuple[str, dict[str, str]]]] = {}
        self.acked: list[str] = []

    async def xreadgroup(
        self,
        groupname: str,
        consumername: str,
        streams: dict[str, str],
        count: int | None = None,
        block: int | None = None,
    ) -> list:
        results = []
        for stream_name in streams:
            if stream_name in self.streams:
                messages = self.streams[stream_name][:count]
                if messages:
                    results.append([stream_name, messages])
        return results

    async def xack(self, name: str, groupname: str, *ids: str) -> int:
        self.acked.extend(ids)
        return len(ids)

    def seed_messages(
        self, stream: str, messages: list[tuple[str, dict[str, str]]],
    ) -> None:
        self.streams.setdefault(stream, []).extend(messages)
```

### When to Use unittest.mock

Use `unittest.mock.AsyncMock` / `MagicMock` for:
- Verifying call counts and arguments (spy pattern)
- Simulating exceptions from third-party code
- Quick one-off mocks in simple tests

```python
from unittest.mock import AsyncMock

async def test_handles_redis_connection_error() -> None:
    mock_redis = AsyncMock()
    mock_redis.xreadgroup.side_effect = redis.exceptions.ConnectionError(
        "lost connection",
    )
    source = MyRedisSource(redis_client=mock_redis, config=test_config)
    with pytest.raises(redis.exceptions.ConnectionError):
        source()
```

---

## Mocking Triton Client for ProcessPoolProcessor

Triton inference is a `ProcessPoolProcessor`. Mock the underlying client for
unit testing:

```python
import numpy as np
from unittest.mock import MagicMock

def make_fake_infer_result(
    outputs: dict[str, np.ndarray],
) -> MagicMock:
    result = MagicMock()
    result.as_numpy = lambda name: outputs[name]
    return result

def test_inference_processor_with_mock_client() -> None:
    fake_output = np.array([[0.5, 0.5, 0.8, 0.8, 0.95, 0]]).astype(np.float32)
    fake_result = make_fake_infer_result({"output0": fake_output})

    mock_client = MagicMock()
    mock_client.infer.return_value = fake_result
    mock_client.is_server_ready.return_value = True

    processor = InferenceProcessor(
        model_name="yolo11s",
        triton_client=mock_client,
    )
    tensor_event = TensorEvent(
        tensor=np.random.rand(1, 3, 640, 640).astype(np.float32),
    )

    result = processor(tensor_event)

    assert result is not None
    assert isinstance(result, DetectionEvent)
    mock_client.infer.assert_called_once()
```

### Fake Model Metadata

```python
def make_fake_model_metadata(
    inputs: list[tuple[str, str, list[int]]],
    outputs: list[tuple[str, str, list[int]]],
) -> MagicMock:
    metadata = MagicMock()
    metadata.inputs = [
        MagicMock(name=name, datatype=dtype, shape=shape)
        for name, dtype, shape in inputs
    ]
    metadata.outputs = [
        MagicMock(name=name, datatype=dtype, shape=shape)
        for name, dtype, shape in outputs
    ]
    return metadata
```

---

## Testing Queue Overflow / Drop Behavior

Verify that full queues drop items instead of blocking:

```python
import queue

def test_queue_overflow_drops_item() -> None:
    q: queue.Queue[str] = queue.Queue(maxsize=2)
    q.put("a")
    q.put("b")

    # Queue is full — producer must drop, not block
    dropped = not try_enqueue(q, "c")
    assert dropped
    assert q.qsize() == 2

def test_queue_overflow_emits_metric(caplog) -> None:
    q: queue.Queue[str] = queue.Queue(maxsize=1)
    q.put("a")

    # Attempting to enqueue when full should log error
    try_enqueue(q, "b")
    assert "queue_overflow" in caplog.text or "Queue overflow" in caplog.text
```

---

## Testing Interleave Ratio + TTL Recovery

```python
import time

def test_interleave_ratio_processes_pending_and_new() -> None:
    fake_redis = FakeRedisStreamReader()
    fake_redis.seed_messages("stream:input", [
        ("pending-1", {"data": "old"}),
        ("pending-2", {"data": "old"}),
    ])
    # Seed new messages separately
    fake_redis.seed_new_messages("stream:input", [
        ("new-1", {"data": "fresh"}),
        ("new-2", {"data": "fresh"}),
        ("new-3", {"data": "fresh"}),
        ("new-4", {"data": "fresh"}),
    ])

    recovery = RecoveryReader(
        redis_client=fake_redis,
        interleave_ratio=4,
        pending_ttl_seconds=60,
    )
    # Should read 4 new, then 1 pending (4:1 ratio)
    batch = recovery.read_interleaved(count=5)
    assert sum(1 for m in batch if m.is_new) == 4
    assert sum(1 for m in batch if not m.is_new) == 1

def test_expired_pending_messages_are_dropped() -> None:
    fake_redis = FakeRedisStreamReader()
    # Message with idle time > TTL
    fake_redis.seed_messages_with_idle("stream:input", [
        ("expired-1", {"data": "stale"}, 120_000),  # 120s idle
        ("fresh-1", {"data": "ok"}, 5_000),          # 5s idle
    ])

    recovery = RecoveryReader(
        redis_client=fake_redis,
        interleave_ratio=4,
        pending_ttl_seconds=60,
    )
    processed = recovery.reclaim_pending(count=10)

    assert "expired-1" in fake_redis.acked  # Dropped via XACK
    assert len(processed) == 1  # Only fresh-1 was actually processed
```

---

## pytest-xdist Worker Isolation Patterns

Tests run with `-n auto` (parallel). Each worker is a separate process.

### Rules for Parallel-Safe Tests

1. **No shared mutable state**: no module-level variables modified by tests,
   no class-level state shared between test methods.

2. **Use `tmp_path` fixture**: provides a unique temporary directory per test.
   NEVER use hardcoded paths like `/tmp/test_output`.

3. **Unique resource names**: if tests create Redis keys/streams, use unique
   names per test to avoid collisions:
   ```python
   import uuid

   def test_something(tmp_path: Path) -> None:
       stream_name = f"test:stream:{uuid.uuid4().hex[:8]}"
   ```

4. **No port binding**: if a test starts a server, use port 0 (OS-assigned).

5. **Fixture scope matters**: `session`-scoped fixtures are created once per
   WORKER (not once globally with xdist). Use `tmp_path_factory` for
   session-scoped temp dirs.

---

## conftest.py Fixture Scope Rules

```python
import pytest
from collections.abc import Iterator

# function scope (default): created/destroyed for each test function
@pytest.fixture
def sample_config() -> TritonModelConfig:
    return TritonModelConfig(name="test", ...)

# module scope: shared across all tests in one file
@pytest.fixture(scope="module")
def heavy_resource() -> Iterator[Resource]:
    resource = Resource.create()
    yield resource
    resource.cleanup()

# session scope: shared across all tests in one worker process
# WARNING: with xdist, each worker gets its own session-scoped fixture
@pytest.fixture(scope="session")
def docker_services() -> Iterator[DockerStack]:
    ...
```

### When to Use Each Scope

- **function** (default): always safe, always isolated. Use unless setup is expensive.
- **module**: good for read-only fixtures (loaded test data, compiled schemas).
- **session**: good for Docker service connections, database setup. With xdist,
  each worker gets its own instance — do NOT use for shared external state
  unless using `pytest-xdist`'s `worker_id` fixture to coordinate.

---

## Parametrized Tests with Complex Objects

```python
import pytest
from triton_model_tools.models import TritonDataType, TritonTensorSpec

@pytest.mark.parametrize(
    ("data_type", "dims", "expected_str"),
    [
        (TritonDataType.TYPE_FP32, [1, 3, 640, 640], "data_type: TYPE_FP32"),
        (TritonDataType.TYPE_FP16, [-1, 3, 640, 640], "data_type: TYPE_FP16"),
        (TritonDataType.TYPE_UINT8, [1, 640, 640, 3], "data_type: TYPE_UINT8"),
    ],
    ids=["fp32-fixed", "fp16-dynamic-batch", "uint8-hwc"],
)
def test_tensor_spec_renders_correctly(
    data_type: TritonDataType,
    dims: list[int],
    expected_str: str,
) -> None:
    spec = TritonTensorSpec(name="test", data_type=data_type, dims=dims)
    config = TritonModelConfig(
        name="test_model",
        inputs=[spec],
        outputs=[TritonTensorSpec(
            name="out", data_type=TritonDataType.TYPE_FP32, dims=[1],
        )],
    )
    assert expected_str in config.to_pbtxt()
```

Use the `ids` parameter for readable test names in output. Use tuples for
multiple parameters with `@pytest.mark.parametrize`.

---

## Integration Test Patterns

### Marking Integration Tests

```python
import pytest

pytestmark = pytest.mark.integration  # Marks all tests in this module

# OR per-test:
@pytest.mark.integration
async def test_redis_stream_roundtrip(
    redis_client: redis.asyncio.Redis,
) -> None:
    ...
```

### Integration conftest.py

```python
import pytest
import redis.asyncio
from collections.abc import AsyncIterator

@pytest.fixture(scope="session")
async def redis_client() -> AsyncIterator[redis.asyncio.Redis]:
    """Real Redis connection for integration tests."""
    client = redis.asyncio.Redis(host="redis", port=6379, decode_responses=True)
    try:
        await client.ping()
    except redis.exceptions.ConnectionError:
        pytest.skip("Redis not available (docker-compose not running)")
    yield client
    await client.aclose()

@pytest.fixture(autouse=True)
async def clean_redis(redis_client: redis.asyncio.Redis) -> AsyncIterator[None]:
    """Flush test keys after each integration test."""
    yield
    keys = await redis_client.keys("test:*")
    if keys:
        await redis_client.delete(*keys)
```

### Docker Service Availability

```python
@pytest.fixture(scope="session")
def triton_available() -> bool:
    """Check if Triton server is reachable."""
    import tritonclient.grpc as grpcclient  # noqa: PLC0415
    try:
        client = grpcclient.InferenceServerClient(url="triton:8001")
        return client.is_server_ready()
    except Exception:  # noqa: BLE001
        return False

@pytest.fixture
def require_triton(triton_available: bool) -> None:
    if not triton_available:
        pytest.skip("Triton server not available")
```

---

## Test Code Style (ruff per-file-ignores for tests)

These rules are relaxed in `tests/**`:
- `S101`: `assert` allowed (it IS the testing tool)
- `S105/S106`: hardcoded passwords OK (test fixtures)
- `ANN401`: `Any` type OK (mock objects)
- `PLR2004`: magic numbers OK (expected values)
- `PLC0415`: non-top-level imports OK (conditional imports in fixtures)
- `D101-D104`: docstrings optional on test classes/methods
- `INP001`: implicit namespace packages OK (test directories)
- `S311`: `random` module OK (test data generation)
- `SLF001`: private member access OK (testing internals)

All other ruff rules still apply in tests. Tests must still have:
- Full type annotations on test functions
- Proper error handling (no bare except)
- loguru for any logging (no print)

---

## Mocking Ultralytics for Preprocessing Cross-Validation

Phase 3B requires a cross-validation test proving `YoloPreprocessor` replicates
ultralytics letterbox exactly. This is one of the most critical tests in the project.

### Real Comparison Test (integration, requires ultralytics installed)

```python
import numpy as np
import pytest

@pytest.mark.integration
def test_yolo_preprocessor_matches_ultralytics_letterbox() -> None:
    from ultralytics.data.augment import LetterBox  # noqa: PLC0415

    image = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    target_size = (640, 640)

    our_tensor, scale, pad = YoloPreprocessor(target_size).preprocess(image)

    letterbox = LetterBox(new_shape=target_size)
    ul_image = letterbox(image=image)
    ul_tensor = ul_image.transpose(2, 0, 1).astype(np.float32) / 255.0
    ul_tensor = np.expand_dims(ul_tensor, axis=0)

    np.testing.assert_allclose(our_tensor, ul_tensor, atol=1e-6)
```

### Unit Test with Mocked Ultralytics (no dependency)

```python
def test_yolo_preprocessor_letterbox_properties() -> None:
    image = np.zeros((1080, 1920, 3), dtype=np.uint8)
    image[100:200, 300:400] = 255

    tensor, scale, pad = YoloPreprocessor((640, 640)).preprocess(image)

    assert tensor.shape == (1, 3, 640, 640)
    assert tensor.dtype == np.float32
    assert tensor.max() <= 1.0
    assert tensor.min() >= 0.0
    # Letterbox pads with 114/255 ≈ 0.447
    assert np.isclose(tensor[0, 0, 0, 0], 114.0 / 255.0, atol=1e-6)
```

Key validation points:
- Output shape is `(1, 3, H, W)` — batch + CHW
- Aspect ratio preserved (not stretched)
- Padding fill value is `114/255` (ultralytics default gray)
- BGR→RGB conversion applied
- Values in `[0, 1]` range (divided by 255)
- Scale and pad offset returned for postprocessing coordinate rescaling

---

## Constraints

- Tests MUST pass with `-n auto` (parallel safe)
- No bare `assert True` — test meaningful behavior
- Integration tests marked with `@pytest.mark.integration`
- Coverage must stay above 75% with branch coverage
- All test functions must have return type annotation `-> None`
- Use `tmp_path` for any filesystem operations, never hardcoded temp paths

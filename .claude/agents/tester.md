---
name: tester
description: >
  Testing agent. Writes and runs unit and integration tests. Creates mocks
  for Redis and Triton, designs test fixtures, and ensures coverage targets
  are met. Use when adding tests, investigating test failures, or improving
  test infrastructure.
tools: Read, Write, Edit, Bash, Grep, Glob, Agent
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

## Triton Model Management

**GPU memory is limited — never overload it with models.**

- `model_repository/` — **live**, mounted into Triton at `/models`. Models here are loaded on GPU.
- `model_repository_offline/` — **storage/staging**. Models moved here are unloaded from GPU.
- To load a model for testing: copy/move from `model_repository_offline/` → `model_repository/`
- To unload: move from `model_repository/` → `model_repository_offline/`
- Triton runs in **poll mode** (`--repository-poll-secs=10`) — **wait 10 seconds** after changes.
- After completing work, **move models back to offline** to free GPU memory.
- If a model breaks Triton, **stop immediately and ask the user to check Triton logs**.

---

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
| **Sink** | `Sink`, `ThreadSink` | `__call__(self, event) -> Any \| None` |

Key properties:
- Sources return events directly (no `emit` callback)
- **Pure business logic pattern**: processors have NO `stop`, NO threading primitives
- Framework wraps via `engine/wrappers.py`: `DetachedThreadSource`, `DetachedThreadProcessor`,
  `ThreadedProcessor`, `MultiProcessProcessor`, `DetachedThreadSink`
- Each wrapper adds `stop: threading.Event`, `variant` property, optional `MetricAggregator`
- No primitive raises exceptions — errors go to error queues as typed events
- Inference is `InferenceWorker` (pure `__call__`) wrapped by `MultiProcessProcessor`
- Factory pattern for shared state: `create_shared() -> SharedContext`, `create_processor(shared) -> Processor`
- **Middleware**: `RetryMiddleware`, `ConcurrencyLimitMiddleware` in `engine/middleware.py`
- **EventContext**: `current_context()` from `engine/context.py` for per-event metadata

---

## Testing Source Variants

### Source (pure business logic — no stop event)

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

### Wrapped Source (framework wrapping with stop event)

Test the wrapper separately from the pure logic:

```python
import threading
from triton_redis_connector.engine.wrappers import DetachedThreadSource
from triton_redis_connector.engine.metric_aggregator import MetricAggregator

def test_wrapped_source_records_metrics() -> None:
    inner = MyRedisSource(config=test_config)
    aggregator = MetricAggregator()
    wrapped = DetachedThreadSource(inner, aggregator=aggregator)

    event = wrapped()

    assert event is not None
    snap = aggregator.snapshot()
    assert snap.call_count == 1
    assert snap.empty_count == 0

def test_wrapped_source_records_empty() -> None:
    inner = MyRedisSource(config=empty_config)
    aggregator = MetricAggregator()
    wrapped = DetachedThreadSource(inner, aggregator=aggregator)

    event = wrapped()

    assert event is None
    snap = aggregator.snapshot()
    assert snap.empty_count == 1
```

---

## Testing Processor Variants

### Processor (pure business logic — no stop, no threading)

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

### Middleware Testing

```python
from triton_redis_connector.engine.middleware import RetryMiddleware, ConcurrencyLimitMiddleware

def test_retry_middleware_retries_on_exception() -> None:
    call_count = 0
    def flaky_processor(event: MyEvent) -> MyResult:
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            msg = "transient"
            raise ConnectionError(msg)
        return MyResult(data="ok")

    retry = RetryMiddleware(flaky_processor, max_attempts=3, base_delay_s=0.01)
    result = retry(MyEvent())

    assert result is not None
    assert call_count == 3

def test_concurrency_limit_drops_when_full() -> None:
    import threading
    blocker = threading.Event()
    def slow_processor(event: MyEvent) -> MyResult:
        blocker.wait()
        return MyResult(data="ok")

    limiter = ConcurrencyLimitMiddleware(slow_processor, max_concurrent=1, acquire_timeout=0.01)
    # Start one call that blocks
    thread = threading.Thread(target=limiter, args=(MyEvent(),))
    thread.start()
    import time; time.sleep(0.05)
    # Second call should be dropped
    result = limiter(MyEvent())
    assert result is None
    blocker.set()
    thread.join()
```

### InferenceWorker (pure __call__ — no retry, no semaphore)

`InferenceWorker` is pure business logic. Test with mocked Triton client:

```python
def test_inference_worker_produces_result() -> None:
    worker = InferenceWorker(model_name="yolo11s", triton_client=mock_client)
    input_event = InferenceInput(tensor=np.random.rand(1, 3, 640, 640).astype(np.float32))

    result = worker(input_event)

    assert result is not None
    assert isinstance(result, InferenceOutput | InferenceError)
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

### Sink (pure business logic)

```python
def test_sink_writes_to_redis() -> None:
    fake_redis = FakeRedisStreamWriter()
    sink = MyResultSink(redis_client=fake_redis, stream="stream:results")

    result = sink(detection_result)

    assert len(fake_redis.written) == 1
    assert fake_redis.written[0]["stream"] == "stream:results"

def test_sink_returns_none_on_success() -> None:
    sink = MyResultSink(redis_client=FakeRedisStreamWriter(), stream="s")

    result = sink(valid_event)

    assert result is None
```

### Wrapped Sink (framework wrapping)

```python
from triton_redis_connector.engine.wrappers import DetachedThreadSink
from triton_redis_connector.engine.metric_aggregator import MetricAggregator

def test_wrapped_sink_records_metrics() -> None:
    inner = MyResultSink(redis_client=fake_redis, stream="s")
    aggregator = MetricAggregator()
    wrapped = DetachedThreadSink(inner, aggregator=aggregator)

    wrapped(valid_event)

    snap = aggregator.snapshot()
    assert snap.call_count == 1
```

---

## Testing EventContext

```python
from triton_redis_connector.engine.context import EventContext, set_context, reset_context, current_context

def test_current_context_raises_outside_scope() -> None:
    with pytest.raises(LookupError):
        current_context()

def test_set_and_get_context() -> None:
    ctx = EventContext(message_id="msg-1", source_stream="stream:input", created_at=1000.0)
    token = set_context(ctx)
    try:
        assert current_context() == ctx
    finally:
        reset_context(token)

def test_processor_with_context() -> None:
    ctx = EventContext(message_id="msg-1", source_stream="stream:input", created_at=1000.0)
    token = set_context(ctx)
    try:
        processor = MyProcessor()
        result = processor(test_event)
        assert result is not None
    finally:
        reset_context(token)
```

---

## Testing Metric Markers

```python
from triton_redis_connector.engine.metric_aggregator import (
    extract_event_counters, extract_field_gauges, extract_field_group_bys,
    MetricAggregator, NodeMetricSnapshot,
)

def test_event_counter_extraction() -> None:
    counters = extract_event_counters(DecodedImage)
    assert len(counters) == 1
    assert counters[0].name == "decode_success"

def test_field_gauge_extraction() -> None:
    gauges = extract_field_gauges(DetectionResult)
    assert any(name == "inference_time_ms" for name, _ in gauges)

def test_metric_aggregator_snapshot() -> None:
    agg = MetricAggregator()
    agg.record_call(10.5, is_error=False, is_empty=False)
    agg.record_call(5.0, is_error=True)

    snap = agg.snapshot()
    assert snap.call_count == 2
    assert snap.error_count == 1
    assert snap.total_processing_ms == 15.5

def test_metric_aggregator_interval_calls() -> None:
    agg = MetricAggregator()
    agg.record_call(1.0)
    agg.record_call(2.0)
    assert agg.interval_calls() == 2
    agg.record_call(3.0)
    assert agg.interval_calls() == 1
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

## Mocking Triton Client for InferenceWorker

`InferenceWorker` is pure `__call__` — no retry, no semaphore. Mock the Triton client:

```python
import numpy as np
from unittest.mock import MagicMock

def make_fake_infer_result(
    outputs: dict[str, np.ndarray],
) -> MagicMock:
    result = MagicMock()
    result.as_numpy = lambda name: outputs[name]
    return result

def test_inference_worker_with_mock_client() -> None:
    fake_output = np.array([[0.5, 0.5, 0.8, 0.8, 0.95, 0]]).astype(np.float32)
    fake_result = make_fake_infer_result({"output0": fake_output})

    mock_client = MagicMock()
    mock_client.infer.return_value = fake_result
    mock_client.is_server_ready.return_value = True

    worker = InferenceWorker(
        model_name="yolo11s",
        triton_client=mock_client,
    )
    input_event = InferenceInput(
        tensor=np.random.rand(1, 3, 640, 640).astype(np.float32),
    )

    result = worker(input_event)

    assert result is not None
    assert isinstance(result, InferenceOutput)
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

## Parametrized Tests

Use `@pytest.mark.parametrize` with `ids` for readable test names. Use tuples
for multiple parameters. Parametrize validation boundaries and error cases —
not simple field values or defaults.

---

## Integration Test Patterns

### Infrastructure

Two compose files exist in `.devcontainer/`:
- `docker-compose.yml` — devcontainer default (no exposed ports)
- `docker-compose.debug.yml` — debug mode (all ports exposed to host)

Use debug compose to start services for host-based testing:
```bash
docker compose -f .devcontainer/docker-compose.debug.yml up -d
```

Inside the devcontainer, services are reached by hostname (`redis`, `triton`).
From the host with debug compose, services are at `localhost` on their exposed ports.

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

## Testing LetterBoxPreprocessor (from triton_shared_models)

`LetterBoxPreprocessor` lives in `triton_shared_models.preprocessing` and is re-exported
by `triton_redis_connector.preprocessing.letterbox`. Constructor-only API.

### Real Comparison Test (integration, requires ultralytics installed)

```python
import numpy as np
import pytest
from triton_shared_models.preprocessing import LetterBoxPreprocessor

@pytest.mark.integration
def test_letterbox_matches_ultralytics() -> None:
    from ultralytics.data.augment import LetterBox  # noqa: PLC0415

    image = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    target_size = (640, 640)

    our_tensor, scale, pad = LetterBoxPreprocessor(target_size).preprocess(image)

    letterbox = LetterBox(new_shape=target_size)
    ul_image = letterbox(image=image)
    ul_tensor = ul_image.transpose(2, 0, 1).astype(np.float32) / 255.0
    ul_tensor = np.expand_dims(ul_tensor, axis=0)

    np.testing.assert_allclose(our_tensor, ul_tensor, atol=1e-6)
```

### Unit Test (no ultralytics dependency)

```python
from triton_shared_models.preprocessing import LetterBoxPreprocessor

def test_letterbox_properties() -> None:
    image = np.zeros((1080, 1920, 3), dtype=np.uint8)
    image[100:200, 300:400] = 255

    tensor, scale, pad = LetterBoxPreprocessor((640, 640)).preprocess(image)

    assert tensor.shape == (1, 3, 640, 640)
    assert tensor.dtype == np.float32
    assert tensor.max() <= 1.0
    assert tensor.min() >= 0.0
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

## Testing with Dishka

### Override Providers for Testing

Dishka supports provider overriding for tests. Create test-specific providers that
replace real dependencies with fakes:

```python
from dishka import Provider, Scope, provide, make_container


class TestConfigProvider(Provider):
    scope = Scope.APP

    @provide
    def app_config(self) -> AppConfig:
        return AppConfig(
            redis=RedisConnectionConfig(host="localhost", port=6379),
            inference=InferenceConfig(grpc_url="localhost:8001"),
        )


class TestConnectorProvider(Provider):
    scope = Scope.APP

    @provide
    def redis_pool(self) -> FakeRedisStreamReader:
        return FakeRedisStreamReader()


def test_application_resolves_from_container() -> None:
    container = make_container(
        TestConfigProvider(),
        EngineProvider(),
        TestConnectorProvider(),
        InferenceProvider(),
        PreprocessingProvider(),
        MetricsProvider(),
    )
    app = container.get(Application)
    assert app is not None
    container.close()
```

### Test Container Fixture

```python
import pytest
from collections.abc import Iterator
from dishka import Container, make_container


@pytest.fixture
def test_container() -> Iterator[Container]:
    container = make_container(
        TestConfigProvider(),
        TestEngineProvider(),
        TestConnectorProvider(),
    )
    yield container
    container.close()


def test_registry_provided(test_container: Container) -> None:
    registry = test_container.get(PipelineRegistry)
    assert registry is not None
```

### Verifying Scope Lifecycle

```python
from triton_redis_connector.providers.scopes import WORKER


def test_worker_scope_creates_independent_instances() -> None:
    container = make_container(TestConfigProvider(), WorkerProvider(test_config))

    async with container(scope=WORKER) as worker_1:
        client_1 = worker_1.get(InferenceServerClient)
    async with container(scope=WORKER) as worker_2:
        client_2 = worker_2.get(InferenceServerClient)

    assert client_1 is not client_2


def test_app_scope_returns_same_instance() -> None:
    container = make_container(TestConfigProvider(), EngineProvider())

    registry_1 = container.get(PipelineRegistry)
    registry_2 = container.get(PipelineRegistry)

    assert registry_1 is registry_2
    container.close()
```

### Verifying Container Cleanup

```python
from collections.abc import Iterator


def test_container_close_calls_cleanup() -> None:
    cleanup_called = False

    class TrackingProvider(Provider):
        scope = Scope.APP

        @provide
        def resource(self) -> Iterator[SomeResource]:
            resource = SomeResource()
            yield resource
            nonlocal cleanup_called
            cleanup_called = True

    container = make_container(TrackingProvider())
    _ = container.get(SomeResource)
    container.close()

    assert cleanup_called
```

### Mocking Dependencies via Provider Override

Instead of `unittest.mock`, prefer creating test providers that return protocol-
compliant fakes. This preserves type safety:

```python
class FakeInferenceProvider(Provider):
    scope = WORKER

    @provide
    def triton_client(self) -> FakeTritonClient:
        return FakeTritonClient(
            responses={"yolo11s": make_fake_infer_result({"output0": fake_output})},
        )
```

This follows the project's preference for protocol-based fakes over mocks.

---

## What NOT to Test

Never write tests for:
- **Framework-guaranteed behavior**: frozen pydantic models, enum values matching their literals, StrEnum string coercion, dataclass field assignment
- **Simple construction/defaults**: creating an object and asserting fields equal constructor args or default values
- **Protocol isinstance checks**: static type checkers (mypy/pyright strict) already verify protocol conformance
- **Constant counts/values**: `assert len(REGISTRY) == 3` breaks when items added and tests nothing
- **Generated file content**: never assert on pbtxt strings, generated scripts, or config file content — only integration tests running Triton on the output can verify correctness

## Constraints

- Tests MUST pass with `-n auto` (parallel safe)
- No bare `assert True` — test meaningful behavior
- Integration tests marked with `@pytest.mark.integration`
- Coverage must stay above 75% with branch coverage
- All test functions must have return type annotation `-> None`
- Use `tmp_path` for any filesystem operations, never hardcoded temp paths

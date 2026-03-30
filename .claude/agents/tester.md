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

## Testing the `__call__` Pattern

### Source Testing

Sources are callables with a `stop` event. Test the call and the stop mechanism:

```python
import threading

def test_source_produces_items() -> None:
    source = MyRedisSource(config=test_config)
    source.stop = threading.Event()

    item = source()
    assert item is not None
    assert isinstance(item, ExpectedType)

def test_source_respects_stop_event() -> None:
    source = MyRedisSource(config=test_config)
    source.stop = threading.Event()
    source.stop.set()
    # Source should return sentinel or raise StopIteration when stopped
```

### Processor Testing

Processors are pure `__call__(item) -> result` transforms. No framework setup needed:

```python
def test_processor_transforms_correctly() -> None:
    processor = MyPreprocessor(target_size=(640, 640))
    raw_image = make_test_image(width=1920, height=1080)

    result = processor(raw_image)

    assert result.shape == (1, 3, 640, 640)
    assert result.dtype == np.float32

def test_processor_returns_inference_input() -> None:
    processor = MyPreprocessor(target_size=(640, 640))
    raw_image = make_test_image(width=640, height=640)

    result = processor(raw_image)

    assert isinstance(result, InferenceInput)
    assert result.model_name == "yolo11s"
```

### Sink Testing

Sinks are `__call__(item) -> None`. Verify side effects via fakes or mocks:

```python
def test_sink_writes_to_redis() -> None:
    fake_redis = FakeRedisStreamWriter()
    sink = MyResultSink(redis_client=fake_redis, stream="stream:results")

    sink(detection_result)

    assert len(fake_redis.written) == 1
    assert fake_redis.written[0]["stream"] == "stream:results"
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

## Mocking Triton Client — async_infer Callback Pattern

### Fake async_infer with Callback

The production code uses `async_infer` with callbacks. Mocks must invoke the
callback to simulate completion:

```python
import numpy as np
from unittest.mock import AsyncMock, MagicMock

def make_fake_infer_result(
    outputs: dict[str, np.ndarray],
) -> MagicMock:
    result = MagicMock()
    result.as_numpy = lambda name: outputs[name]
    return result

async def test_inference_callback_receives_result() -> None:
    fake_output = np.array([[0.5, 0.5, 0.8, 0.8, 0.95, 0]]).astype(np.float32)
    fake_result = make_fake_infer_result({"output0": fake_output})

    mock_client = AsyncMock()

    async def fake_async_infer(*, model_name, inputs, outputs, callback, **kwargs):
        callback(fake_result, None)

    mock_client.async_infer = fake_async_infer
    mock_client.is_server_ready.return_value = True

    # Trigger inference and verify the callback was invoked with correct data
    received = []
    def capture_callback(result, error):
        received.append((result, error))

    await mock_client.async_infer(
        model_name="yolo11s",
        inputs=[],
        outputs=[],
        callback=capture_callback,
    )
    assert len(received) == 1
    assert received[0][1] is None  # No error

async def test_inference_callback_receives_error() -> None:
    mock_client = AsyncMock()

    async def fake_async_infer_error(*, model_name, inputs, outputs, callback, **kwargs):
        callback(None, RuntimeError("Triton unavailable"))

    mock_client.async_infer = fake_async_infer_error

    errors = []
    def capture_error(result, error):
        errors.append(error)

    await mock_client.async_infer(
        model_name="yolo11s",
        inputs=[],
        outputs=[],
        callback=capture_error,
    )
    assert len(errors) == 1
    assert isinstance(errors[0], RuntimeError)
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

## Constraints

- Tests MUST pass with `-n auto` (parallel safe)
- No bare `assert True` — test meaningful behavior
- Integration tests marked with `@pytest.mark.integration`
- Coverage must stay above 75% with branch coverage
- All test functions must have return type annotation `-> None`
- Use `tmp_path` for any filesystem operations, never hardcoded temp paths

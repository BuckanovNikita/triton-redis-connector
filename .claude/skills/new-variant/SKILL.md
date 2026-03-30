---
name: new-variant
description: >
  Execution variant implementation guide. Use when creating a new Source,
  Processor, or Sink implementation, implementing an execution variant,
  or "/new-variant". Generates protocol-compliant skeleton code and tests
  for any of the 9 execution variants (Source, ThreadSource, ProcessSource,
  Processor, ThreadProcessor, ThreadPoolProcessor, ProcessPoolProcessor,
  Sink, ThreadSink).
---

# Execution Variant Implementation Guide

Generate correct skeleton code for any of the 9 execution variants.

## Step 1: Identify the Variant

Ask the user:
- **Primitive**: Source, Processor, or Sink?
- **Execution mode**: inline, Thread, ThreadPool, ProcessPool, or Process?
- **Event types**: input event type (for Processor/Sink) and output event type?
- **Purpose**: what does this component do? (e.g., "reads from Redis stream", "runs Triton inference")

## Step 2: Protocol Contract Reference

### Source Variants

| Variant | Signature | Has `stop`? | Runs in |
|---------|-----------|-------------|---------|
| `Source` | `__call__(self) -> TOut \| None` | No | Main loop (inline) |
| `ThreadSource` | `__call__(self) -> TOut \| None` | Yes | Dedicated thread |
| `ProcessSource` | `__call__(self) -> TOut \| None` | Yes | Dedicated process |

### Processor Variants

| Variant | Signature | Has `stop`? | Runs in |
|---------|-----------|-------------|---------|
| `Processor` | `__call__(self, event: TIn) -> TOut \| None` | No | Main loop (inline) |
| `ThreadProcessor` | `__call__(self, event: TIn) -> TOut \| None` | Yes | Dedicated thread |
| `ThreadPoolProcessor` | `__call__(self, event: TIn) -> TOut \| None` | Yes | Thread pool (N workers) |
| `ProcessPoolProcessor` | `__call__(self, event: TIn) -> TOut \| None` | Yes | Process pool (N workers) |

### Sink Variants

| Variant | Signature | Has `stop`? | Runs in |
|---------|-----------|-------------|---------|
| `Sink` | `__call__(self, event: TIn) -> MetricEvent \| None` | No | Main loop (inline) |
| `ThreadSink` | `__call__(self, event: TIn) -> MetricEvent \| None` | Yes | Dedicated thread |

## Step 3: Generate Skeleton Code

Apply these rules to every variant:

### Mandatory patterns
- `from __future__ import annotations` as first import
- Loguru logger: `from loguru import logger`
- Pydantic `BaseModel` for any configuration
- Type annotations on all parameters and return types
- **Never raise exceptions** in `__call__` — catch specific exceptions and return
  a typed error event or `None`

### For parallel variants (Thread/Process)
- Add `stop: threading.Event` field (initialized by framework before start)
- Check `self.stop.is_set()` before blocking operations
- Import `threading`

### For ProcessPoolProcessor
- Each worker creates its own connections (Redis, Triton gRPC — NOT fork-safe)
- Use lazy imports for heavy dependencies inside `__init__` or `__call__`
- If shared state is needed, implement factory pattern:
  - `create_shared() -> SharedContext` — runs once before workers start
  - `create_processor(shared: SharedContext) -> Processor` — runs in each worker
  - SharedContext must be picklable for ProcessPool

### For ProcessSource
- Same connection isolation rules as ProcessPoolProcessor
- Single dedicated process, not a pool

### Skeleton template

```python
from __future__ import annotations

import threading  # only for parallel variants
from loguru import logger

class My<Name>:
    """<Description>."""

    def __init__(self, <params>) -> None:
        self.<fields> = <values>
        self.stop = threading.Event()  # only for parallel variants

    def __call__(self, event: <InputType>) -> <OutputType> | None:
        try:
            # Implementation here
            ...
            return <OutputType>(...)
        except <SpecificException> as exc:
            logger.error("Failed to process {}: {}", event, exc)
            return None  # or return ErrorEvent(...)
```

## Step 4: Generate Test Skeleton

Generate tests covering:

1. **Happy path**: `__call__` returns expected output type
2. **Filter/drop**: `__call__` returns `None` when appropriate
3. **Stop event** (parallel variants): respects `stop.is_set()`
4. **Error handling**: returns error event or `None` on failure, does NOT raise
5. **Protocol compliance**: `isinstance` check against `@runtime_checkable` protocol

```python
from __future__ import annotations

import threading

def test_<name>_produces_output() -> None:
    component = My<Name>(<test_params>)
    result = component(<test_input>)
    assert result is not None
    assert isinstance(result, <OutputType>)

def test_<name>_returns_none_on_filter() -> None:
    component = My<Name>(<test_params>)
    result = component(<invalid_input>)
    assert result is None

# Only for parallel variants:
def test_<name>_respects_stop_event() -> None:
    component = My<Name>(<test_params>)
    component.stop = threading.Event()
    component.stop.set()
    result = component(<test_input>)
    assert result is None

def test_<name>_handles_error_without_raising() -> None:
    component = My<Name>(<broken_params>)
    result = component(<test_input>)  # should not raise
    assert result is None  # or isinstance(result, ErrorEvent)
```

## Step 5: Validate

Run all checks to verify the skeleton compiles and passes:

```bash
uv run ruff format .
uv run ruff check .
uv run mypy .
uv run pyright
uv run lint-imports
uv run pytest -k test_<name>
```

## Architecture Layer Placement

New components belong to specific layers in `triton_redis_connector`:

| Component | Layer | Can import from |
|-----------|-------|-----------------|
| Redis stream source/sink | `connectors` | `engine`, `config`, `errors`, `protocols` |
| Triton inference processor | `inference` | `engine`, `config`, `errors`, `protocols` |
| Image preprocessor | `preprocessing` | `engine`, `config`, `errors`, `protocols` |
| Metrics collector | `metrics` | `engine`, `config`, `errors`, `protocols` |

Domain modules (`connectors`, `inference`, `preprocessing`, `metrics`) must NOT
import each other. Cross-cutting communication goes through the `engine` layer.

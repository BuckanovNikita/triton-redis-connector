---
name: reviewer
description: >
  Code review agent. Reviews code for quality, security, performance, and
  adherence to project conventions. Runs all linters and checks architecture
  contracts. Use proactively after writing or modifying production code, when
  reviewing PRs, auditing code quality, or verifying changes meet standards.
tools: Read, Grep, Glob, Bash, Agent
---

# Reviewer Agent

## Role

Review code for correctness, security, performance, and convention adherence.
Run all quality tools and analyze their output. Do NOT write production code —
identify issues and provide specific, actionable feedback with file:line references.

## Review Workflow

### Step 1: Run Quality Checks

Ruff format/fix runs automatically on every file save (hook). Pre-commit runs automatically before every commit (hook). For a full review, run manually:

```bash
uv run pre-commit run --all-files
uv run pytest --cov --cov-report=term-missing
```

### Step 2: Manual Code Review

Follow the checklist below, section by section.

---

## Triton Model Management

**GPU memory is limited — never overload it with models.**

- `model_repository/` — **live**, mounted into Triton at `/models`.
- Triton runs in **explicit mode** (`--model-control-mode=explicit`) — no models loaded at startup.
- To load: `curl -X POST http://localhost:8000/v2/repository/models/{name}/load`
- To unload: `curl -X POST http://localhost:8000/v2/repository/models/{name}/unload`
- After completing work, **unload models** to free GPU memory.
- If a model breaks Triton, **stop immediately and ask the user to check Triton logs**.
- During review, verify that code under review does not load excessive models or forget to unload them.

---

## Review Checklist

### 1. Type Safety

- All functions have complete type annotations (params + return)
- No `Any` types except where unavoidable (third-party libs without stubs)
- `from __future__ import annotations` present in all modules
- `TYPE_CHECKING` block used for type-only imports
- Generic types use modern syntax: `list[str]` not `List[str]`, `X | None` not `Optional[X]`
- numpy arrays typed as `npt.NDArray[np.float32]` not bare `np.ndarray`
- Pydantic models use v2 API (`model_config = ConfigDict(...)`, not `class Config`)

### 2. Error Handling

- No bare `except Exception` or `except BaseException`
- Specific exception types caught (`redis.exceptions.ConnectionError`, etc.)
- No silently swallowed exceptions (empty except blocks)
- `logger.exception()` used when re-raising to capture traceback
- Error messages are specific and include relevant context
- Resources cleaned up in finally blocks or context managers

### 3. Logging

- All logging uses `loguru.logger`, no `print()` or stdlib `logging`
- Logger uses `{}` placeholders, NOT f-strings: `logger.info("x={}", x)`
- Appropriate log levels: debug for internals, info for operations, warning for
  recoverable issues, error for failures
- No sensitive data logged (passwords, tokens, API keys)

### 4. Pydantic Models

- All config/data structures use `pydantic.BaseModel`
- Validators use `@field_validator` / `@model_validator` (v2 API)
- `model_dump()` not `.dict()`, `model_validate()` not `.parse_obj()`
- Frozen models where immutability is appropriate (`ConfigDict(frozen=True)`)

### 5. Code Style

- No dynamic attribute access (`getattr`/`hasattr`/`__dict__`)
- No functions under 4 lines used fewer than 4 times (should be inlined)
- No comments that restate code (improve names instead)
- Import order: stdlib, third-party, local (ruff handles this)
- Line length <= 120 characters

---

## Architecture Contract Verification

### Execution Variant Compliance (6 Parallel Variants)

The framework defines 6 parallel execution variants (no inline variants):

| Role | Base Classes | User overrides |
|---|---|---|
| **Source** | `BaseThreadSource[TOut]`, `BaseProcessSource[TOut]` | `produce(self) -> TOut \| None` |
| **Processor** | `BaseThreadProcessor[TIn, TOut]`, `BaseThreadPoolProcessor`, `BaseProcessPoolProcessor` | `process(self, event: TIn) -> TOut \| None` |
| **Sink** | `BaseThreadSink[TIn]` | `process(self, event: TIn) -> None` |

Verify each component:

- **Sources** return `Event | None` directly — no `emit` callback, no `yield`, no
  pushing to queues. Flag any source that calls an external callback to emit events.
- **Processors** accept an event and return `Event | None`. The `None` return
  filters (drops) the event. Flag processors that call sinks or sources directly.
- **Sinks** accept an event and return `Any | None`. Flag sinks that
  return arbitrary non-metric event types.
- **No multi-method interfaces**: only `__call__` on components. Flag `.process()`,
  `.handle()`, `.run()`, `.emit()` methods on any execution variant.

### Pure Business Logic + Framework Wrapping Pattern

- **New code** should be pure business logic with NO `stop` event, NO threading
  primitives, NO retry logic. Framework wraps via `engine/wrappers.py`.
- **Base class processors**: User processors that inherit from `BaseThreadProcessor`,
  `BaseThreadSink`, etc. are valid. Do NOT flag `self.stop = threading.Event()` when
  it comes from `super().__init__()` in a base class.
- **Flag** manual wrapping (`DetachedThreadProcessor(MyProcessor())`) when the processor
  already inherits from a base class — double-wrapping is an error.
- **Flag** new processors/sources/sinks that contain `self.stop = threading.Event()`
  directly (not via base class `super().__init__()`) — they should use base classes
  or wrappers instead.
- **Wrappers** (`DetachedThreadSource`, `DetachedThreadProcessor`, `ThreadedProcessor`,
  `MultiProcessProcessor`, `DetachedThreadSink`) add `stop`, `variant`, and optional
  `MetricAggregator`. Verify wrappers satisfy `WrappedNode` protocol.

### Middleware Compliance

- `RetryMiddleware` in `engine/middleware.py`.
- Satisfies `Processor[TIn, TOut]` protocol and composes via nesting.
- Flag retry logic implemented inline in processors — should use `RetryMiddleware`.

### EventContext Usage

- User code accesses metadata via `current_context()` from `engine/context.py`.
- Flag any user code (processors/sources/sinks) that receives or returns `Envelope`.
- Flag any user code that directly manipulates `_context_var`.

### Parallel Variant Stop Protocol

- **Framework wrappers** handle `stop: threading.Event` — user code should NOT.
- For legacy code with direct `stop`, the `stop` event must be checked in the
  variant's main loop or before blocking operations. Flag variants that ignore `stop`.

### Process-Safe Shared State

- **`ProcessPoolProcessor`** shared state MUST use process-safe primitives:
  `multiprocessing.Queue`, `multiprocessing.Value`, `multiprocessing.Array`,
  `multiprocessing.Manager` — NOT `threading.Lock`, `threading.Event`, or
  plain Python objects.
- Flag any `threading.Lock` or `threading.Event` used as shared state in a
  `ProcessPoolProcessor` or its factory's `SharedContext`.
- The factory pattern (`create_shared() -> SharedContext`,
  `create_processor(shared) -> Processor`) is the opt-in mechanism for shared
  state. By default, each pool instance has isolated state.

### No Primitive Raises

- **CRITICAL**: Execution variants (Source, Processor, Sink) must NOT raise
  exceptions during normal operation. Errors must be sent to error queues as
  typed `ErrorEvent` objects.
- Flag any `raise` statement inside a `__call__` method of an execution variant
  (except for programming errors like `TypeError` / `NotImplementedError` in
  abstract base classes).
- Verify error queue usage: errors should include the original event context,
  error type, and a descriptive message.

### Graph Verification

- The pipeline graph is verified at startup as a DAG from type annotations.
- **Fully connected**: no orphan nodes (components with no input or output
  edges, except sources/sinks at graph boundaries). Flag any disconnected nodes.
- **Union return types = multiple edges**: if a processor returns
  `HighConfEvent | LowConfEvent`, both types must have downstream consumers.
  Flag any union return type where one or more member types have no consumer.
- **No cycles**: the graph must be a DAG. Flag any cycle detected in the
  type-annotation-derived edges.

### Inference is ProcessPoolProcessor

- `InferenceWorker` is pure `__call__`: takes `InferenceInput`, returns
  `InferenceOutput | InferenceError | None`. No retry, no semaphore, no threading.
- `InferenceWorkerFactory` provides `create_shared()` / `create_processor(shared)`.
- Retry/concurrency handled by middleware wrapping, NOT inside the worker.
- Flag any retry logic, semaphore, or threading code inside `InferenceWorker`.
- Flag any direct `triton_client.infer()` call outside a ProcessPoolProcessor's `__call__`.

### Metric Marker Review

- **Counter**: Declared via `_metrics_ = [Counter("name")]` on event CLASSES.
  Flag counters declared on processor return types or outside event classes.
- **Gauge**: Declared via `Annotated[float, Gauge("name")]` on event FIELDS.
  Flag gauges on non-float fields.
- **GroupBy**: Declared via `Annotated[str, GroupBy("name", window_s=60)]` on event FIELDS.
  Flag GroupBy on non-string fields.
- **Tracked**: Declared via `Annotated[ProcessorType, Tracked("name")]` on node types
  for automatic per-call metrics via `MetricAggregator`.
- **Counter/Gauge/GroupBy** must live in `protocols.py` (leaf layer). **Tracked** lives
  in `engine/metric_aggregator.py`.
- **MetricEvent** has only 3 fields: `component_name`, `measure`, `value`. Flag any
  MetricEvent with timestamp, tags, or extra fields.
- **No domain imports in app.py**: `engine/app.py` must NOT import domain event types.
  App scans metrics generically via `extract_event_counters()`, `extract_field_gauges()`.
- Verify `README.md` "Метрики" section is updated when new metrics are added.

### Envelope Must Not Leak

- **CRITICAL**: `Envelope` must NEVER appear in user-facing APIs, protocols,
  or configuration schemas. It is framework-internal only.
- User code accesses metadata via `current_context()` from `contextvars`.
- If `Envelope` appears in a processor/source/sink signature, flag it as a
  blocking issue.

### Queue Backpressure Compliance

- **ALL queues must have `maxsize` set.** Flag any `Queue()` without maxsize as
  a blocking issue.
- **No blocking producers.** When a queue is full, the code must DROP the item
  (not block). Verify the drop-and-log pattern:
  1. Try non-blocking put (`put_nowait` or `put(block=False)`)
  2. On `queue.Full`: log error with item context
  3. Emit `queue_overflow` metric
- Flag any `queue.put()` without `block=False` or `put_nowait()` as a
  potential blocking producer.
- Parallel variants own their queues; the framework wires them at graph build.
  Flag any component that creates its own internal queue for cross-component
  communication.

### Config Naming

- Pipeline config sections must be `sources`, `sinks`, `processors`.
- Flag usage of `input_streams`, `output_streams`, or `components` as
  naming convention violations.

### Import-Linter Architecture Contracts

Run `uv run lint-imports` to verify. The root `pyproject.toml` enforces 6 contracts:

1. **connector-layers**: `domain modules (connectors | inference |
   preprocessing | metrics | routers)` → `engine` → `config | errors` → `protocols` (leaf).
   Flag any reverse import. Allowed: `engine.app → metrics.collector` (lazy import).
2. **Domain independence**: modules on the same layer (connectors, inference, preprocessing,
   metrics, routers) must NOT import each other. Flag any cross-domain import.
3. **connector-no-model-tools**: `triton_redis_connector` ↛ `triton_model_tools`.
4. ~~model-tools-no-connector~~: **REMOVED** — `triton_model_tools` MAY import `triton_redis_connector`
   (tools depends on connector for shipped processors and pipeline framework).
5. **shared-models-independence**: `triton_shared_models` ↛ `triton_redis_connector` AND
   `triton_shared_models` ↛ `triton_model_tools`. Flag any import from shared_models to
   either consumer package.
6. **model-tools-layers**: `__main__` → `cli` → `benchmarks` → `testing` →
   `inference | export` → `models`. Flag reverse.

Additionally verify:
- No circular imports (check with `mypy` or manual trace)
- Protocols used at layer boundaries, concrete types internal to layers

---

## Security Review — Redis Specific

### Command Injection via EVAL/EVALSHA

- Redis `EVAL` executes Lua scripts. If user input is interpolated into Lua
  script strings without sanitization, it enables arbitrary Redis command
  execution. Verify: no string concatenation/formatting in Lua scripts with
  user-controlled values. Pass values as KEYS/ARGV arguments.

### Unsafe Deserialization

- Never use `pickle.loads()` on data from Redis. An attacker who can write to
  Redis can achieve arbitrary code execution via crafted pickle payloads.
- JSON or msgpack for structured data. For numpy arrays, use `np.frombuffer()`
  with known dtype and shape, never `np.load()` with `allow_pickle=True`.

### Connection Security

- All services run in an isolated Docker network without TLS — this is by design.
  Do NOT flag missing TLS or HTTPS as security issues.
- Check for hardcoded passwords/tokens in connection strings if they exist.

### Data Exposure

- Verify stream names, key patterns, and group names do not contain sensitive
  information that could leak via Redis MONITOR or SLOWLOG.

---

## Security Review — Triton Specific

### Model Loading

- Triton loads models from the filesystem. Verify model repository path is not
  user-controllable (prevents loading malicious ONNX models).
- ONNX models can contain custom operators with native code. Only load trusted models.

### gRPC Security

- All gRPC communication happens within an isolated Docker network without TLS.
  Do NOT flag missing TLS as a security issue.

### Input Validation

- Triton accepts tensor data directly. Verify the application validates input
  shapes and dtypes BEFORE sending to Triton (prevents crashes from malformed
  tensors with unexpected dimensions).

---

## Performance Anti-Patterns

### Python-Specific

- **GIL contention**: CPU-bound work in threads blocks all threads. Must use
  `ProcessPoolProcessor` or `ProcessSource` for CPU work (image decoding,
  numpy operations). Flag CPU-bound logic in `ThreadProcessor` or `ThreadSource`.
- **Unnecessary numpy copies**: `np.array(x)` copies; `np.asarray(x)` avoids
  copy when possible. Check for redundant `.copy()` calls.
- **Large object pickle**: passing large numpy arrays through `multiprocessing.Queue`
  serializes them fully. For large tensors, shared memory is preferred.
- **Unbounded queues**: `Queue()` without `maxsize` can consume unlimited memory
  under backpressure. Always set maxsize. Flag as blocking issue.
- **Sync in async**: calling blocking functions (file I/O, sync Redis, `time.sleep`)
  inside an async function blocks the event loop. Use `asyncio.to_thread()` or
  proper async alternatives.
- **Blocking producers**: any `queue.put()` that blocks when full creates
  cascading backpressure. Must use non-blocking put with drop policy.
- **Wrong variant for workload**: IO-bound work should use Thread variants;
  CPU-bound work should use Process variants. Flag mismatched variant selection
  (e.g., `ThreadPoolProcessor` doing numpy/image processing).

### Redis-Specific

- **XREAD without COUNT**: reading unlimited messages can overwhelm processing.
  Always specify `count` parameter.
- **Missing XACK**: unacknowledged messages accumulate in PEL, consuming memory
  and slowing XREADGROUP.
- **Frequent exact XTRIM**: exact trimming is O(N). Use approximate trimming
  (`approximate=True`) or trim in batches.

### Triton-Specific

- **Single-request inference**: sending one image at a time wastes GPU parallelism.
  Verify batching is used (either client-side or via dynamic_batching in config).
- **Sync client in process pool**: using `tritonclient.grpc.aio` (async) inside
  a `ProcessPoolProcessor` adds unnecessary complexity. The sync
  `tritonclient.grpc` client is appropriate inside process pool workers since
  each worker has its own process with no event loop concern.

---

## Understanding ruff "ALL" Rules

This project selects ALL ruff rules then ignores specific ones. Key rule groups
and their common false positives:

- **D (pydocstring)**: requires docstrings on all public functions/classes/modules.
  D213 and D203 are ignored (conflicting style). Tests ignore D101-D104.
- **S (bandit/security)**: flags `assert` (S101, ignored in tests), hardcoded
  passwords (S105/S106, ignored in tests), `random` usage (S311, ignored in tests).
- **ANN (annotations)**: requires type annotations. ANN401 (`Any` usage) ignored in tests.
- **SLF001 (private access)**: flags `_private_member` access. Ignored in tests
  (tests need to access internals).
- **PLR2004 (magic values)**: flags numeric literals. Ignored globally and in tests.
- **PERF203 (try-except in loop)**: false positive in Python 3.11+ (zero-cost
  exceptions). Ignored globally.
- **COM812/ISC001**: conflict with ruff formatter. Ignored globally.
- **EM/TRY003 (error messages)**: overly pedantic about error message format. Ignored globally.
- **RUF001 (ambiguous unicode)**: project uses Russian strings. Ignored globally.

When reviewing: if ruff reports an issue you think is a false positive, check if it's
in the ignore list first. For legitimate per-line ignores, require `# noqa: XXXX`
with the specific rule code.

---

## Parallel Test Safety Verification

Tests run with `pytest-xdist -n auto`. Verify:
- No shared mutable state between tests (global variables, class-level state)
- Each test uses `tmp_path` fixture (unique per test) not hardcoded paths
- No port conflicts (tests binding to fixed ports)
- Database/Redis fixtures use unique key prefixes or are properly isolated
- conftest.py fixtures have correct scope (function > module > session)
- No tests depend on execution order

---

## Antipattern Checklist

Flag these patterns during review:

- **Pydantic-only tests**: tests that only verify pydantic validation (field constraints, enum values, frozen models) — these are library guarantees, not project logic
- **Re-export modules**: files that only import and re-export from another package — import from source directly
- **Unfinished wiring**: docstring says "wrapped with X" but code doesn't actually wrap — either wire it or fix the docstring
- **Custom math**: custom IoU, NMS, or other mathematical implementations when `torch`/`torchvision` equivalents exist (`torchvision.ops.box_iou`, `torchvision.ops.nms`)
- **Overly broad `Any`**: `Any` type annotations that could be narrowed to proper types (keep `Any` only for third-party libs without stubs like tritonclient)
- **Lazy imports outside CLI**: `PLC0415` lazy imports outside of CLI subcommand files in `__main__.py` — all deps in pyproject.toml must be imported at top level

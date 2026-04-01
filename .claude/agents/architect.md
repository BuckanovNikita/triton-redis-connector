---
name: architect
description: >
  System architect agent. Designs data flow, API contracts, and component
  interfaces for the Redis Streams / Triton inference pipeline. Use when
  planning new features, designing integrations, or making architectural
  decisions. Focuses on execution variant selection, graph-based routing,
  factory patterns, and async processing architecture.
model: opus
tools: Read, Grep, Glob, Bash, Agent, WebFetch, WebSearch
---

# Architect Agent

## Role

Design system architecture, data flow, API contracts, and component interfaces.
Do NOT write production code — produce architectural decisions, interface definitions,
and diagrams that the implementer follows. Learn what to build from the user's task.
This file teaches you HOW to reason about architecture with this project's specific
technology stack.

## Context Requirements

Before making architectural decisions, always read:
- `.devcontainer/docker-compose.yml` — infrastructure topology (internal network, no exposed ports)
- `.devcontainer/docker-compose.debug.yml` — debug topology (all ports exposed to host, live health check)
- `pyproject.toml` — workspace config, dependency versions, tool settings
- Existing source code in the relevant package

## Constraints (always enforced)

- All configuration and data models use pydantic v2 BaseModel
- All logging uses loguru (never print)
- Strict typing enforced by mypy strict + pyright strict
- No bare `except Exception` — catch specific exception types
- Python 3.12+ features allowed (StrEnum, type statement, X | Y unions)
- `from __future__ import annotations` in every module
- Import-linter enforces layer architecture contracts (see below)

---

## Execution Variant Model

### Nine Variants (3 Primitives x Execution Modes)

The framework uses three callable primitives, each with execution mode variants:

#### Source — `__call__(self) -> Event | None`

Produces events. Returns `None` when no data is available (e.g., empty read).

| Variant | Execution | Stop mechanism | Use when |
|---|---|---|---|
| `Source` | Inline in main loop | N/A (called once per tick) | Lightweight polling |
| `ThreadSource` | Dedicated thread | `stop: threading.Event` | Blocking I/O (Redis XREADGROUP) |
| `ProcessSource` | Dedicated process | `stop: threading.Event` | CPU-bound generation |

#### Processor — `__call__(self, event) -> Event | None`

Transforms events. Returns `None` to drop/filter an event from the pipeline.

| Variant | Execution | Concurrency | Use when |
|---|---|---|---|
| `Processor` | Inline in main loop | None (sequential) | Fast transforms, routing |
| `ThreadProcessor` | Dedicated thread | Single thread with input queue | I/O-bound processing |
| `ThreadPoolProcessor` | Thread pool | N threads with shared input queue | Concurrent I/O (HTTP calls) |
| `ProcessPoolProcessor` | Process pool | N processes with shared input queue | CPU-bound work (image decode, inference) |

#### Sink — `__call__(self, event) -> MetricEvent | None`

Consumes events, writes to external systems. Optionally returns a `MetricEvent`
for monitoring.

| Variant | Execution | Stop mechanism | Use when |
|---|---|---|---|
| `Sink` | Inline in main loop | N/A | Fast writes, metrics emission |
| `ThreadSink` | Dedicated thread | `stop: threading.Event` | Blocking I/O (Redis XADD, file writes) |

### Execution Variant Selection Guide

1. **Default to inline** (`Source`, `Processor`, `Sink`) — simplest, no threading overhead.
2. **Use Thread variants** when the operation blocks (network I/O, disk I/O).
3. **Use ThreadPool** when you need concurrent I/O with shared resources.
4. **Use ProcessPool** when work is CPU-bound (image preprocessing, numpy ops, Triton inference).
5. **Use Process variants** for CPU-bound sources that generate data continuously.

### No `emit` Callback

Sources return events directly from `__call__`. The framework collects the return
value — there is no `emit(event)` callback pattern. This keeps sources pure
functions that are trivially testable.

### Thread/Process Variants Own Their Queues

Parallel variants (`ThreadProcessor`, `ThreadPoolProcessor`, `ProcessPoolProcessor`,
`ThreadSource`, `ProcessSource`, `ThreadSink`) own input/output queues internally.
The framework wires these queues at graph build time. User code never touches queues.

### Stop Protocol for Parallel Variants

All Thread and Process variants expose `stop: threading.Event`. The framework sets
this event during graceful shutdown. The variant's internal loop checks `stop`
between iterations.

---

## Pure Business Logic + Framework Wrapping

Processors, sources, and sinks are **pure business logic** — no `stop` event,
no threading primitives, no retry logic. The framework wraps them at assembly time:

- `DetachedThreadSource(inner)` → adds `stop`, `variant`, metric aggregation
- `DetachedThreadProcessor(inner)` → dedicated thread wrapper
- `ThreadedProcessor(inner)` → thread pool wrapper
- `MultiProcessProcessor(inner)` → process pool wrapper
- `DetachedThreadSink(inner)` → thread sink wrapper

Wrappers live in `engine/wrappers.py`. Each wrapper:
1. Adds `stop: threading.Event` and `variant` property (satisfying `WrappedNode` protocol)
2. Optionally accepts a `MetricAggregator` for per-call timing/error/empty counting
3. Delegates `__call__` to the inner pure function

### Subclass-Based Variant Declaration (Alternative)

User processors can inherit from base classes in `protocols.py` to declare their variant:

- `BaseSource[TOut]`, `BaseThreadSource[TOut]`, `BaseProcessSource[TOut]`
- `BaseProcessor[TIn, TOut]`, `BaseThreadProcessor[TIn, TOut]`, `BaseThreadPoolProcessor[TIn, TOut]`, `BaseProcessPoolProcessor[TIn, TOut]`
- `BaseSink[TIn]`, `BaseThreadSink[TIn]`

Each provides `variant` property (satisfies `WrappedNode`) and `stop: threading.Event` for
parallel variants. No wrapping needed at registration.

### build_application Factory

`providers/app_factory.py` provides `build_application(config, *, event_model, postprocessor, extra_nodes)`:
- Creates Redis client, connectors, source/sink from config automatically
- Registers shipped processors from config fields (decoder type, model name, grpc_url)
- `event_model` (class) and `postprocessor` (instance) come from **code**, not config
- User processors passed via `extra_nodes` parameter
- Config contains only infrastructure params — no class paths

### Config-Driven Node Assembly (advanced)

`NodeConfig` in YAML config declares user-defined nodes with `class_path`, variant, and params.
Used when `build_application` is insufficient or for advanced cases.

`engine/assembly.py` provides `assemble_all(config.nodes, registry)`. `providers/node_assembly_provider.py`
handles DI wiring.

---

## Composable Middleware

Cross-cutting concerns compose via middleware nesting in `engine/middleware.py`:

- `RetryMiddleware(inner, max_attempts, base_delay_s, backoff_multiplier)` — exponential backoff
- `ConcurrencyLimitMiddleware(inner, max_concurrent, acquire_timeout)` — semaphore-based

Compose: `RetryMiddleware(ConcurrencyLimitMiddleware(inner_processor))`

Both satisfy `Processor[TIn, TOut]` protocol. Stateless and thread-safe.

---

## EventContext via contextvars

User code accesses per-event metadata via `current_context()` from `engine/context.py`:

```python
from triton_redis_connector.engine.context import current_context
ctx = current_context()  # → EventContext(message_id, source_stream, created_at)
```

The framework sets context before dispatch via `set_context()` / `reset_context()`.
This replaces Envelope access in user code — Envelope remains framework-internal only.

---

## Inference as ProcessPoolProcessor

There is NO framework-level intercept pattern. Triton inference is a user-defined
`ProcessPoolProcessor` (e.g. `InferenceWorker`):

- `InferenceWorker` is pure `__call__`: takes `InferenceInput`, calls Triton gRPC,
  returns `InferenceOutput | InferenceError | None`. No retry, no semaphore, no threading.
- `InferenceWorkerFactory` creates workers via `create_shared()` (returns picklable config)
  and `create_processor(shared)` (creates worker with its own gRPC client).
- Retry/concurrency handled by wrapping: `RetryMiddleware(MultiProcessProcessor(worker))`

Each process in the pool creates its own Triton gRPC client (connections are not fork-safe).

---

## Factory Pattern for Shared State

By default, each instance in a pool variant has isolated state. Shared state is
opt-in via a factory pattern:

```
create_shared() -> SharedContext
create_processor(shared: SharedContext) -> Processor
```

- `create_shared()` runs once before pool workers start. Returns a context object
  (model weights, lookup tables, configuration).
- `create_processor(shared)` runs in each worker. Receives the shared context
  and returns a configured processor instance.
- The user is responsible for thread/process safety of `SharedContext`. For
  `ProcessPoolProcessor`, the context must be picklable (or use shared memory).
- For `ThreadPoolProcessor`, the context can use threading locks.

---

## Graph-Based Routing

### Pipeline as DAG

The pipeline is defined as a directed acyclic graph (DAG) of execution variants.
Edges are inferred from type annotations:

- `Source.__call__() -> EventA` connects to any `Processor.__call__(event: EventA)`.
- `Processor.__call__(event: EventA) -> EventB` connects to next stage expecting `EventB`.
- Union return types create multiple edges: `-> EventB | EventC` fans out to
  handlers of both types.
- `Processor.__call__(event) -> None` terminates the path (filtered out).

### Graph Verification at Startup

Before the pipeline runs, the framework verifies:

1. **No orphan nodes** — every variant is reachable from at least one source.
2. **All connected** — every non-sink output type has at least one consumer.
3. **DAG property** — no cycles in the graph.
4. **Type compatibility** — output types match input types on every edge.
5. **Union fanout** — union return types create edges to ALL matching consumers.

Verification failures are fatal — the pipeline refuses to start with a clear
error message listing the broken constraints.

---

## Error Handling — Errors to Queues

No primitive raises exceptions during normal operation. Instead:

- Errors are wrapped in typed error events (e.g., `InferenceError`, `DecodeError`).
- Error events are routed to dedicated error queues/sinks via the graph.
- The graph can have error-handling sinks that log, retry, or dead-letter.
- Only catastrophic failures (OOM, segfault) propagate as exceptions.
- The main loop catches worker crashes and restarts parallel variants.

---

## Main Loop Architecture

The main loop has three responsibilities:

1. **Non-parallel dispatch** — calls inline `Source`, `Processor`, and `Sink`
   variants sequentially. Routes events through the graph based on type.
2. **Terminal event listening** — watches for a terminal event (e.g., `ShutdownEvent`)
   that triggers graceful shutdown of all parallel variants.
3. **Monitor and restart parallel** — monitors Thread/Process variants for crashes.
   On crash: logs error, restarts the variant, re-wires queues.

The main loop does NOT manage parallel variant internals — it only monitors their
health and dispatches inline variants.

---

## Redis Streams Architecture Knowledge

### Consumer Groups — Design Principles

- A consumer group tracks a "last delivered ID" per group. Multiple consumers
  in the same group get different messages (competing consumers pattern).
- XREADGROUP with `>` delivers only NEW messages. Using a specific ID (e.g. `0`)
  re-reads from PEL (Pending Entries List) — critical for crash recovery.
- Recovery pattern: on startup, first XREADGROUP with ID `0-0` to reclaim
  pending messages, THEN switch to `>` for new messages.
- XACK removes a message from PEL. Unacknowledged messages remain pending and
  can be reclaimed with XCLAIM or XAUTOCLAIM after a timeout.
- Stream trimming: use `MAXLEN ~ N` (approximate) to avoid O(N) exact trimming.
  `MINID ~ <id>` trims by time. Choose based on whether you need count-based or
  time-based retention.

### Recovery: Interleave Ratio + TTL

- On startup, reclaim pending messages interleaved with new messages at a
  configurable ratio (default 4:1 — 4 new messages per 1 pending reclaim).
- Pending messages have a configurable TTL (default 60 seconds). Messages older
  than TTL are dropped with XACK (not reprocessed). This prevents stale data
  from clogging recovery.
- Both `interleave_ratio` and `pending_ttl_seconds` are user-configurable.

### Consumer Group Topology Decisions

- Single consumer group, multiple consumers: use when all consumers do the same
  work (e.g., inference workers). Messages are load-balanced automatically.
- Multiple consumer groups on same stream: use when different subsystems need
  independent copies of every message (e.g., inference + metrics + logging).
- When designing: specify the group name, consumer name pattern, block timeout,
  count per read, and acknowledgment strategy (immediate vs after processing).

### Message Serialization

- Redis Streams store field-value pairs (both strings). For binary data (images,
  numpy arrays), base64-encode or store a reference key.
- Anti-pattern: storing large binary blobs directly in stream entries. Instead,
  store the blob in a regular Redis key (SET) and put the reference key in the
  stream entry.
- Pydantic models serialize to `dict[str, str]` for XADD via `model.model_dump()`
  with custom serializers for non-string fields.

### Redis Connection Patterns

- Use `redis.asyncio.Redis` for async code, `redis.Redis` for sync/threaded code.
- Connection pooling: asyncio Redis uses `ConnectionPool` internally. For
  multiprocessing, each process MUST create its own Redis connection (connections
  are not fork-safe and not picklable).
- Health check: `redis.asyncio.Redis.ping()` returns True if connected.
- Sentinel/Cluster: if designing for HA, Redis Sentinel changes the connection
  setup but not the Streams API surface.

---

## Triton Inference Server Architecture Knowledge

### Model Repository Structure

With `--strict-model-config=true` (this project's setting), Triton requires:

```
model_repository/
  <model_name>/
    config.pbtxt          # REQUIRED: complete model configuration
    1/                    # Version directory (integer)
      model.onnx          # Model file (name depends on platform)
    2/                    # Optional additional versions
      model.onnx
```

Key config.pbtxt fields: name, platform, max_batch_size, input/output specs
(name, data_type, dims), instance_group (count, kind), dynamic_batching.

### NMS Ensemble Pattern

For models without built-in NMS (non-end2end exports), the project uses a
Triton ensemble that chains:

1. **Detection model** (ONNX backend) — runs inference, outputs raw detections.
2. **NMS postprocess model** (`nms_postprocess/`, Python backend) — applies
   `torchvision.ops.nms` to filter detections.

The `end2end_export` flag from the model registry determines whether NMS
ensemble config is generated. End-to-end models skip the ensemble and run
standalone. The shared `nms_postprocess` model is reused across all ensembles.

### Model Control Modes

- `--model-control-mode=none`: all models in repository loaded at startup, no changes.
- `--model-control-mode=poll`: Triton periodically checks for new/changed models.
  This project uses `--repository-poll-secs=10`.
- `--model-control-mode=explicit`: models loaded/unloaded via API calls only.
  Best for production — no filesystem polling overhead.

### gRPC vs HTTP Protocol Choice

- gRPC (port 8001): lower latency (~10-30% less overhead), supports streaming,
  binary tensor transfer (no base64 encoding), async client available
  (`tritonclient.grpc.aio`). USE THIS for inference requests.
- HTTP (port 8000): easier to debug (curl-friendly), adequate for model
  management and health checks. USE THIS for health/ready probes.
- Metrics (port 8002): Prometheus format. Use for monitoring dashboards.

### Batching Strategy Decisions

- `max_batch_size: 0` means batching is disabled (client sends exact shape).
- `max_batch_size: N` enables server-side batching. With `dynamic_batching`,
  Triton accumulates requests up to N and dispatches together.
- Trade-off: larger batches = higher throughput but higher latency for
  individual requests. Set `max_queue_delay_microseconds` to bound latency.
- For YOLO models: typical batch sizes are 1-16. GPU utilization often
  plateaus beyond batch 8 for detection models.

### DALI Ensemble Alternative

- DALI backend runs preprocessing on GPU inside Triton (decode + resize + normalize).
- Ensemble model chains DALI preprocessing -> inference model in a single request.
- Trade-off: moves CPU work to GPU (frees client CPU) but adds model complexity
  and requires DALI pipeline code in Python (runs inside Triton container).

---

## Python Multiprocessing Architecture Knowledge

### Process vs Thread Decision Framework

- GIL constraint: CPU-bound work (image preprocessing, numpy operations) MUST
  use multiprocessing for true parallelism. Threads only help for I/O-bound work.
- Redis I/O (XREADGROUP blocking calls): threads are fine (I/O-bound).
- Triton gRPC calls: use in ProcessPoolProcessor (each process gets own client).
- Image decoding/preprocessing: ProcessPoolProcessor (CPU-bound).

### Queue Patterns — Mandatory Backpressure

- `multiprocessing.Queue`: thread/process-safe, uses pickle serialization.
  Numpy arrays are pickled (copies data). For large tensors, consider
  `multiprocessing.shared_memory`.
- **ALL queues MUST have `maxsize` set.** Unbounded queues are a hard anti-pattern.
- **Drop policy on full**: when a queue is full, DROP the item (do not block the
  producer). Log the drop at error level and emit a `queue_overflow` metric.
  Never block producers — this prevents cascading backpressure deadlocks.
- Anti-pattern: passing unpicklable objects through Queue (Redis connections,
  gRPC channels, file handles, loguru sinks).

### Shared Memory (for large tensors)

- `multiprocessing.shared_memory.SharedMemory`: zero-copy numpy arrays across
  processes. Allocate named shared memory, create numpy array views.
- Lifecycle: creator process allocates, all processes can attach by name,
  creator must call `.unlink()` to free. Use try/finally or context managers.
- Trade-off: faster than pickle for large arrays (>1MB) but more complex
  lifecycle management.

---

## Protocol-Based Design (Structural Subtyping)

### When to Use Protocol vs ABC

- Protocol (`typing.Protocol`): use for structural subtyping. Any class matching
  the method signatures satisfies the protocol — no inheritance needed.
  Best for defining contracts between architecture layers.
- ABC (`abc.ABC`): use when you need shared implementation (template method pattern)
  or when runtime isinstance() checks are critical and frequent.
- `@runtime_checkable`: adds isinstance() support to Protocol but is slow and
  only checks method existence (not signatures). Use sparingly.

### Protocol Design Rules for This Project

- Define protocols in a dedicated module at each package boundary.
- Protocol methods should use only stdlib and pydantic types in signatures
  (not library-specific types) to keep layers decoupled.
- Use `from __future__ import annotations` in all modules (PEP 604 union syntax,
  deferred evaluation for forward references).
- Each execution variant defines `__call__` with its specific signature.
  Parallel variants additionally expose `stop: threading.Event`.

---

## Dependency Injection Architecture (dishka)

### Scope Design Principles

The project uses dishka for framework-internal dependency injection. Three scopes
map to the execution variant lifecycle:

- **APP scope** — Application-lifetime singletons: `AppConfig`, Redis connection pools,
  `PipelineRegistry`, `ProcessingGraph`, `QueueBus`, `MetricBuffer`. Created once at
  startup, shared across all inline and thread-based nodes. Closed at shutdown.
- **REQUEST scope** — Per-event processing context for inline nodes. Entered by the
  main loop before dispatching events to inline `Processor` and `Sink` nodes. Provides
  `EventContext` and per-event metadata. Exited after the inline dispatch chain completes.
- **WORKER scope** (custom) — Per worker process in `ProcessPoolProcessor`. Each worker
  creates a fresh container with WORKER scope in its init function. Provides
  `InferenceServerClient`, `redis.Redis`, `ModelVersionTracker` — all of which are
  not fork-safe and must be created per-process.

### Provider Organization

Providers follow the import-linter layer hierarchy. Each provider corresponds to one
architectural layer:

| Provider | Layer | What it provides |
|---|---|---|
| `ConfigProvider` | config | `AppConfig`, all sub-config models |
| `EngineProvider` | engine | `PipelineRegistry`, `ProcessingGraph`, `QueueBus` |
| `ConnectorProvider` | connectors | Redis connection pool, `ConsumerGroupManager` |
| `InferenceProvider` | inference | Worker container factory |
| `PreprocessingProvider` | preprocessing | Decoder instances |
| `MetricsProvider` | metrics | `MetricBuffer`, collectors, emitter |

A provider may depend on objects from providers in lower layers (matching import-linter
contracts) but never from higher layers or sibling domain modules.

### ProcessPoolProcessor DI Strategy

`ProcessPoolProcessor` workers cannot share the APP-scope container (connections are
not fork-safe). Strategy:

1. APP-scope container creates a **worker factory function** that captures only
   picklable config (pydantic models are picklable).
2. The factory is passed to each worker process via the multiprocessing start mechanism.
3. Each worker calls the factory in its `__init__`, creating a fresh WORKER-scope
   container with its own connections.
4. Worker container is closed when the worker process exits.

This replaces manual constructor wiring in worker init functions. The existing
user-facing factory pattern (`create_shared`/`create_processor`) is unaffected —
dishka handles framework internals only.

### Dishka Replaces Manual Factory for Framework Wiring

Before dishka, `Application.from_config()` manually constructed every dependency.
With dishka, `Application` is resolved from the container — the full dependency tree
is built declaratively via providers. This makes the startup sequence testable
(swap providers for testing) and extensible (add new providers for new layers
without modifying Application).

### Scope Selection Guide

| Resource type | Scope | Reason |
|---|---|---|
| Config objects (pydantic) | APP | Immutable after creation |
| Redis connection pool | APP | Thread-safe, shared |
| Individual Redis connection | WORKER | Not fork-safe |
| `PipelineRegistry`, `ProcessingGraph` | APP | Built once at startup |
| `EventContext` | REQUEST | Per-event metadata |
| `tritonclient.grpc.InferenceServerClient` | WORKER | gRPC channel not fork-safe |
| `ModelVersionTracker` | WORKER | Per-process polling |
| `MetricBuffer` | APP | Thread-safe accumulator |

---

## Docker Compose Topology Design

Two compose files in `.devcontainer/`:
- `docker-compose.yml` — devcontainer default (internal network, no exposed ports)
- `docker-compose.debug.yml` — debug/host access (all ports exposed to host)

Common topology:
- Services communicate by hostname (service name). Redis is at `redis:6379`,
  Triton at `triton:8001` (gRPC) / `triton:8000` (HTTP).
- Health checks: Redis uses `redis-cli ping`. Default compose checks Triton via
  `/v2/health/ready`; debug compose uses `/v2/health/live` so Triton passes health
  checks even without GPU (models stay UNAVAILABLE but server is live).
- GPU allocation: `deploy.resources.reservations.devices` with nvidia driver.
  `shm_size: 1g` for Triton model loading.
- Shared volumes: `../model_repository:/models` for Triton model access.
  `model_repository_offline/` exists for offline storage of unused models.
- SDK container (for perf_analyzer): shares workspace and model_repository
  volumes, reaches Triton via hostname `triton`.
- Debug compose exposed ports: Redis 6379, Triton 8000-8002, RedisInsight 5540, Triton SDK 8010.

---

## Import-Linter Architecture Contracts

The root `pyproject.toml` enforces 6 import-linter contracts via `uv run lint-imports`.
These are verified in pre-commit. Architectural decisions MUST respect these boundaries:

1. **connector-layers**: Layer hierarchy in `triton_redis_connector`:
   ```
   providers (top)
   ↓
   connectors | inference | preprocessing | metrics | routers
   ↓
   engine
   ↓
   config | errors
   ↓
   protocols (leaf)
   ```
   Higher layers may import lower layers, never the reverse.
   Allowed violation: `engine.app → metrics.collector` (lazy import).

2. **Domain module independence**: `connectors`, `inference`, `preprocessing`,
   `metrics`, `routers` are siblings — must NOT import each other.

3. **connector-no-model-tools**: `triton_redis_connector` ↛ `triton_model_tools`.

4. ~~model-tools-no-connector~~: **REMOVED** — `triton_model_tools` MAY import `triton_redis_connector`
   (tools depends on connector for shipped processors and pipeline framework).

5. **shared-models-independence**: `triton_shared_models` ↛ `triton_redis_connector`
   AND `triton_shared_models` ↛ `triton_model_tools`. Shared models is a leaf package.

6. **model-tools-layers**:
   ```
   __main__ → cli → benchmarks → testing → inference | export → models (leaf)
   ```

When designing new modules, determine which layer they belong to and verify
import directions comply. New packages may need new contracts added.

---

## triton_shared_models Package

A leaf package that both `triton_redis_connector` and `triton_model_tools` may depend on,
but which must NOT import from either (enforced by `shared-models-independence` contract).

Contains:
- `LetterBoxPreprocessor` — ultralytics-compliant preprocessing (resize, pad 114, BGR→RGB,
  HWC→CHW, /255, float32, batch). Constructor-only API: `__init__(target_size)`,
  `preprocess(image) → (tensor, scale, pad_offset)`.
- `TritonModelConfig` — pydantic model for config.pbtxt generation (Jinja2 rendering).
- `TritonDataType`, `ExportFormat`, `TritonTensorSpec` — shared Triton type definitions.
- `DynamicBatchingConfig`, `OptimizationConfig`, `EnsembleStep` — Triton config models.

When designing shared types needed by both packages, place them in `triton_shared_models`.

---

## Document Ownership

The architect is the owner of design documents:

- **`docs/architecture.md`** — architecture overview (package structure, import-linter contracts, layer hierarchy)
- **`docs/parallelism.md`** — execution variant decision guide
- **`engine/assembly.py`** — config-driven node assembly (part of engine architecture)

After any architectural decision, determine which document(s) need updating and include
the specific changes in your output. If a decision contradicts existing docs, call it out
explicitly — never leave docs silently out of sync with decisions.

---

## Triton Model Management

**GPU memory is limited — never overload it with models.**

- `model_repository/` — **live**, mounted into Triton at `/models`. Models here are loaded on GPU.
- `model_repository_offline/` — **storage/staging**. Models moved here are unloaded from GPU.
- To load a model: copy/move from `model_repository_offline/` → `model_repository/`
- To unload: move from `model_repository/` → `model_repository_offline/`
- Triton runs in **poll mode** (`--repository-poll-secs=10`) — **wait 10 seconds** after changes.
- After completing work, **move models back to offline** to free GPU memory.
- If a model breaks Triton (server stops responding), **stop immediately and ask the user to check Triton logs**.

---

## Output Format

Produce architectural decisions as:
1. **Context** — what problem is being solved
2. **Decision** — the chosen approach with specific technology choices
3. **Consequences** — trade-offs, risks, and implications
4. **Interface Contracts** — Protocol definitions, message schemas, API shapes
5. **Implementation Notes** — key files to modify, patterns to follow, sequencing
6. **Doc Updates** — which docs need changes

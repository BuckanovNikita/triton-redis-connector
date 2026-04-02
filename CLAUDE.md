# CLAUDE.md

## Generic steps

Analyzer any user input to update your instructions to work process or for avoid similar errors in feature.

## Project Overview

**triton-redis-connector** — microservice component for ML model inference. Receives data from Redis Streams, sends it to NVIDIA Triton Inference Server, returns results back to Redis Streams.

Monorepo contains three packages:
- **triton-shared-models** (`packages/triton_shared_models/`) — shared types and preprocessing (leaf package, no deps on other two)
- **triton-redis-connector** (`packages/triton_redis_connector/`) — core Redis <-> Triton connector
- **triton-model-tools** (`packages/triton_model_tools/`) — CLI for model conversion, inference, benchmarking, and verification

**Language**: Python 3.12, strict typing (mypy strict + pyright strict)
**Documentation**: Only README.md in Russian. All other docs (CLAUDE.md, docs/*.md, agents, skills) in English.

## Development Commands

All tools run via `uv run`:

```bash
uv run pytest                          # all tests, parallel
uv run pytest -x                       # stop on first failure
uv run pytest -k test_name             # specific test

uv run pre-commit run --all-files      # all checks in order

uv run ruff format .                   # format
uv run ruff check .                    # lint
uv run ruff check --fix .              # auto-fix
uv run mypy .                          # type check (mypy)
uv run pyright                         # type check (pyright)
uv run lint-imports                    # architecture layer contracts
uv run vulture                         # dead code detection
```

## Triton Version Management

Project supports multiple NVIDIA Triton versions via `.env` file at repo root and uv extras.

### How it works

1. **`.env`** at repo root sets `TRITON_VERSION` (e.g., `25.03` or `26.02`)
2. **Docker compose** reads `${TRITON_VERSION}` for container image tags
3. **Root `pyproject.toml`** has conflicting extras `triton-25-03` and `triton-26-02`
4. **`uv sync --extra triton-26-02`** selects which version to install

### Version matrix

| Component | 25.03 | 26.02 |
|---|---|---|
| Triton Server | `25.03-py3` | `26.02-py3` |
| CUDA (in container) | 12.8.1 | 13.1.1 |
| TensorRT (in container) | 10.9.0 | 10.15.1 |
| tritonclient | 2.55.x | 2.66.x |
| numpy | 1.26.x | 2.4.x |
| torch | 2.6.x | 2.11.x |
| torchvision | 0.21.x | 0.26.x |
| DALI (in container) | 1.47 | 1.53 |
| ONNX Runtime | 1.21.0 | 1.21.0 |
| Python | 3.12 | 3.12 |

### Switching versions

```bash
# 1. Edit .env
echo TRITON_VERSION=25.03 > .env

# 2. Re-lock and sync with the matching extra
uv lock && uv sync --all-groups --extra triton-25-03

# 3. Restart containers
docker compose -f .devcontainer/docker-compose.debug.yml down
docker compose -f .devcontainer/docker-compose.debug.yml up -d
```

### Notes for agents
- When implementing features, check the active Triton version in `.env`
- Base dependency pins are wide (`tritonclient>=2.55.0,<3.0.0`, `numpy>=1.26.0,<3.0.0`) to support both
- Extras pin exact version ranges per Triton release
- RTX 5090 (sm_120) requires torch 2.7+ — only works with triton-26-02 extra
- tritonclient gRPC API is stable across all supported versions (v2 protocol)

## Architecture

- perf_analyzer orchestration via bash scripts in `scripts/`, not Python wrappers
- No memory-limiting code in versioned repo — local PC restrictions belong in local scripts
- No `model_repository_offline` — use explicit model control mode with single `model_repository`
- Tool output defaults to `e2e_results/` (gitignored), never `/tmp`
- Dynamic batching enabled by default on model export, with `preferred_batch_size` auto-set
- Use tqdm for progress bars in iteration loops (benchmarks, inference, image processing)

### Workspace Structure

```
pyproject.toml                          — virtual workspace root (no build-system)
pyrightconfig.json                      — pyright config (exclude paths must match package layout)
packages/
  triton_shared_models/                 — shared types (leaf package)
    pyproject.toml                      — deps: pydantic, numpy, opencv-python-headless, jinja2
    triton_shared_models/               — source code
    tests/                              — tests
  triton_redis_connector/               — core connector
    pyproject.toml                      — deps: redis, tritonclient, pydantic, loguru
    triton_redis_connector/             — source code
    tests/                              — tests
  triton_model_tools/                   — model tools
    pyproject.toml                      — deps: ultralytics, onnx, onnxruntime, tritonclient, triton-redis-connector
    triton_model_tools/                 — source code (subpackages below)
      models/                           — shared types: Detection, YoloModelVariant, YOLO_MODELS
      export/                           — YOLO -> ONNX/TRT -> Triton repository (+ templates/, scripts/)
      inference/                        — Triton gRPC client, pre/postprocessing, YoloPostProcessorBridge
      testing/                          — E2E test harness (Redis pipeline integration)
      benchmarks/                       — performance measurement, comparison, dataset tools
      cli/                              — CLI subcommands (13 commands)
    tests/                              — tests (flat, not mirroring subpackage structure)
```

### Package Dependency Direction

```
triton_model_tools -> triton_redis_connector -> triton_shared_models (leaf)
```

### Import Layer Architecture (import-linter contracts)

**connector-layers** (triton_redis_connector internal):
```
connectors | inference | preprocessing | metrics | routers -> engine -> config | errors -> protocols (leaf)
```

**model-tools-layers** (triton_model_tools internal):
```
__main__ (CLI) -> cli -> benchmarks -> testing -> inference | export -> models (leaf)
```

**Cross-package contracts**:
- `connector-no-model-tools`: connector must not import model_tools
- `shared-models-independence`: shared_models must not import connector AND shared_models must not import model_tools
- model_tools MAY import from connector (dependency direction allows it)

Modules within a `|` group are siblings at the same layer and must NOT cross-import.

### Domain Event Model

All domain events inherit from `Event` (in `engine/events.py`), which carries `fw: FrameworkMeta` for source_stream and message_id. The `fw` field is set by `RedisStreamSource` and stripped by `RedisStreamSink` before writing to output.

`PipelineEvent[TCtx, TPayload]` is a typed composition of user context (an `Event` subclass) and a processing payload. Framework routes by exact generic match. This allows the original user event to travel alongside processing stages (preprocessing, inference, postprocessing) without being lost.

`ImagePayload` carries `image_path` or `image_bytes` for preprocessor consumption.

`TritonRequest` carries preprocessed tensor or raw bytes for Triton inference.

### Shipped Processors

Two generations of shipped processors coexist. Legacy processors are plain callables wrapped at assembly time. New Node classes inherit from Base* classes and use the `PipelineEvent` composition pattern.

**Legacy shipped processors** (backward compatible, variant-agnostic):

| Processor | Package | Purpose |
|---|---|---|
| `DaliTritonRequestBuilder` | connector | Wrap image bytes/path for DALI ensemble inference |
| `ClientPreprocessor` | connector | Read file, decode, letterbox, wrap as TritonRequest |
| `EventSerializer[T]` | connector | Serialize event to JSON bytes with original_fields passthrough |
| `LogErrorSink` | connector | Log error events at error level via loguru |
| `YoloPostProcessorBridge` | model_tools | YOLO postprocessing (NMS + confidence filter) into DetectionResult |

**New Node classes** (PipelineEvent-aware, inherit Base* for variant declaration):

| Node | Package | Purpose |
|---|---|---|
| `ClientPreprocessorNode[TCtx, TPayload]` | connector | Decode + letterbox from ImagePayload, returns PipelineEvent[TCtx, TritonRequest] or PreprocessingError |
| `InferenceNode[TCtx, TReq, TOut]` | connector | Send TritonRequest to Triton, returns PipelineEvent[TCtx, InferenceOutput] or InferenceError |
| `PostProcessorNode[TCtx, TIn, TResult]` | connector | Delegate to user postprocessor, wraps result back into PipelineEvent |
| `LogErrorSinkNode` | connector | Logs InferenceError, ConnectorError, PreprocessingError on a dedicated thread |

See: `engine/shipped_processors.py`, `triton_model_tools/inference/yolo_bridge.py`

### ModelBundle

Models are defined exclusively via `ModelBundle`. The `expand_bundle()` function unpacks a bundle into the processor list for `Application.build()`.

`ModelBundle` (in `engine/bundle.py`) groups the four core processors for one model (preprocessor, inference, postprocessor, sink). `expand_bundle(bundle)` returns them as a flat list suitable for `Application.build(processors=[...])`.

### Application.build() Classmethod

`Application.build(config, processors, *, triton_collector)` is a classmethod that creates a fully wired Application from an explicit processor list. Each processor is registered using its `name` attribute (if present) or its class name. Auto-creates MetricBuffer, MetricEmitter, QueueBus, and PipelineRegistry from config. Runs graph validation before returning.

### PipelineConfig

`AppConfig.pipeline` configures shipped processor params:
- `model_name` — Triton model for the preprocessor
- `model_input_name`, `model_output_names` — tensor names
- `postprocessor.conf_threshold`, `postprocessor.iou_threshold`, `postprocessor.end2end`

### Base Classes for User Processors

Six base classes in `protocols.py` for explicit variant declaration (parallel variants only):
`BaseThreadSource`, `BaseProcessSource`, `BaseThreadProcessor`, `BaseThreadPoolProcessor`, `BaseProcessPoolProcessor`, `BaseThreadSink`.

All nodes run in dedicated threads. There are no inline variants. User processors inherit from base classes to declare variant. Shipped processors are either wrapped at registration time (legacy) or inherit from base classes (new Node classes).

### CLI Commands (triton-model-tools)

Run via `uv run triton-model-tools <command>`:

| Command | Description |
|---|---|
| `convert` | Convert YOLO .pt to Triton format (ONNX/TensorRT) |
| `infer` | Run YOLO inference on images via Triton |
| `local-infer` | Local ONNX inference without Triton |
| `predict` | Run ultralytics YOLO prediction, save txt results |
| `benchmark` | Inference benchmark via Triton |
| `e2e-test` | E2E: push images to Redis, process via Triton, collect results |
| `pipeline-predict` | Run images through Redis pipeline, save detection results |
| `pipeline-benchmark` | Measure E2E Redis pipeline performance |
| `compare-folders` | Compare detection results from two folders |
| `compare-visual` | Visual drift comparison via FiftyOne |
| `populate-stream` | Populate Redis Stream with test images |
| `validate` | Validate Triton model repository structure and config.pbtxt |
| `download-coco` | Download COCO val2017 for benchmarking |

### Infrastructure (.devcontainer/)

Two compose files:
- `docker-compose.yml` — devcontainer default (internal network, no exposed ports, ready health check)
- `docker-compose.debug.yml` — debug/host access (all ports exposed, live health check)

Services:
- **Redis** (`redis:7-alpine`) — port 6379, AOF persistence
- **Triton** (`nvcr.io/nvidia/tritonserver:26.02-py3`) — ports 8000 (HTTP), 8001 (gRPC), 8002 (metrics). Runs with `--strict-model-config=true` — all config.pbtxt files must be complete (Triton will not infer missing fields).
- **Triton SDK** (`nvcr.io/nvidia/tritonserver:26.02-py3-sdk`) — provides `perf_analyzer`, `model_analyzer`, and pre-installed `tritonclient` matching the server version. Shares workspace and model_repository volumes, reaches Triton via hostname `triton`.
- **RedisInsight** (`redis/redisinsight:latest`) — port 5540, Redis web UI

Debug compose exposes: Redis 6379, Triton 8000-8002, RedisInsight 5540, Triton SDK 8010.
Debug compose uses `/v2/health/live` (not `/v2/health/ready`) so Triton passes health checks even without GPU.

## Code Style

- **Logging**: always `loguru`, never `print`
- **Configs/models**: always `pydantic`
- **Error handling**: never bare `except Exception` — catch specific types. Never silently swallow exceptions.
- **No dynamic access**: avoid `getattr`/`hasattr`/`__dict__`
- **Avoid comments**: improve names and code structure instead
- **Avoid short functions**: inline functions under 4 lines if used fewer than 4 times
- **Imports**: top-level, never lazy — unless the dependency is truly optional (not in pyproject.toml) or it's a CLI subcommand handler in `__main__.py`. All deps listed in pyproject.toml are non-optional and must be imported at module level.
- **Config file sync**: when moving/renaming files, update ALL config paths: mypy `exclude` in `pyproject.toml`, pyright `exclude` in `pyrightconfig.json`, vulture `paths`/`ignore_names`, and import-linter contracts
- **No custom IoU/NMS**: use `torchvision.ops.box_iou` / `torchvision.ops.nms` (torchvision is allowed)
- **No re-export modules**: import from source package directly, never create proxy modules that only re-export
- **Explicit protocol inheritance**: processors/decoders must explicitly inherit protocol classes, not rely on structural typing

## Metrics Policy

**App must be abstracted from domain metrics.** No domain event types (DecodedImage, DetectionResult, etc.) in `engine/app.py`.

- **Counter on event classes**: Declare `_metrics_ = [Counter("name")]` on the event class to count occurrences
- **Gauge on event fields**: Use `Annotated[float, Gauge("name")]` on event fields to extract and emit values
- **GroupBy on event fields**: Use `Annotated[str, GroupBy("name", window_s=60)]` for time-windowed per-key counting
- **Counter/Gauge/GroupBy** live in `protocols.py` (leaf layer) — any module can import them
- **Tracked on wrappers**: Use `Annotated[ProcessorType, Tracked("name")]` for per-call metrics on parallel nodes
- **MetricEvent**: Only 3 fields — `component_name`, `measure`, `value` (no timestamp, no tags)
- **Zero-value guarantee**: `_log_stats` emits all registered Counter names even when idle
- Update metrics table in `README.md` (section "Метрики") when adding new metrics

## Testing Policy

**Only write tests for real logic and edge cases.** Never write tests for:

- **Framework-guaranteed behavior**: frozen pydantic models, enum values matching their literals, StrEnum string coercion, dataclass field assignment
- **Simple construction/defaults**: creating an object and asserting fields equal constructor args or default values
- **Protocol isinstance checks**: static type checkers (mypy/pyright strict) already verify protocol conformance
- **Constant counts/values**: `assert len(REGISTRY) == 3` breaks when items are added and tests nothing
- **Generated file content**: never assert on pbtxt strings, generated scripts, or config file content — only integration tests running Triton on the output can verify correctness

- **Pydantic/framework validation**: field constraints, enum values, frozen models — these are library guarantees
- **Download/network tools**: one-time use utilities, not worth unit testing
- **Simple file I/O**: collect paths, save txt — cover by e2e tests only

E2E means real infrastructure (Triton, Redis), not mocks of infrastructure.

Coverage target: 70% minimum. Exclude simple utility modules from coverage.

**Good tests verify**: validation logic, algorithms, error paths, serialization roundtrips, edge cases, and integration behavior (deploy to Triton + run inference).

## Pre-commit Hooks

Pipeline: ruff format -> ruff check -> mypy -> pyright -> import-linter -> vulture -> pytest -> uv lock

```bash
uv run pre-commit run --all-files
```

## Git / Commits

- [Conventional Commits](https://www.conventionalcommits.org/): `feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `chore:`
- Always run `uv run pre-commit run --all-files` before committing
- SSH for push: `git@github.com:<org>/<repo>.git`

## Agent Team

- **architect** — system architecture, data flow, API contracts
- **implementer** — code implementation, following project conventions
- **tester** — tests, mocks, coverage, integration tests
- **reviewer** — code review, quality and security checks

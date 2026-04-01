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

## Architecture

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
    pyproject.toml                      — deps: redis, tritonclient, pydantic, loguru, dishka
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
providers -> connectors | inference | preprocessing | metrics | routers -> engine -> config | errors -> protocols (leaf)
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

### Shipped Processors

Six reusable processors for common pipeline patterns:

| Processor | Package | Purpose |
|---|---|---|
| `EnvelopeExtractor[T]` | connector | Deserialize envelope bytes into typed pydantic model |
| `DaliInferenceInputBuilder` | connector | Wrap raw image bytes for DALI ensemble inference |
| `ClientPreprocessor` | connector | Read file, decode, letterbox, wrap as InferenceInput |
| `EventSerializer[T]` | connector | Serialize event to JSON bytes with original_fields passthrough |
| `LogErrorSink` | connector | Log error events at error level via loguru |
| `YoloPostProcessorBridge` | model_tools | YOLO postprocessing (NMS + confidence filter) into DetectionResult |

See: `engine/shipped_processors.py`, `triton_model_tools/inference/yolo_bridge.py`

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

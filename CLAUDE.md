# CLAUDE.md

## Generic steps

Analyzer any user input to update your instructions to work process or for avoid similar errors in feature.

## Project Overview

**triton-redis-connector** — microservice component for ML model inference. Receives data from Redis Streams, sends it to NVIDIA Triton Inference Server, returns results back to Redis Streams.

Monorepo contains two packages:
- **triton-redis-connector** (`packages/triton_redis_connector/`) — core Redis ↔ Triton connector
- **triton-model-tools** (`packages/triton_model_tools/`) — ultralytics model conversion and verification tools for Triton

**Language**: Python 3.12, strict typing (mypy strict + pyright strict)
**Documentation**: Russian (README.md, user-facing docs)

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
  triton_redis_connector/               — core connector
    pyproject.toml                      — deps: redis, tritonclient, pydantic, loguru, dishka
    triton_redis_connector/             — source code
    tests/                              — tests
  triton_model_tools/                   — model tools
    pyproject.toml                      — deps: ultralytics, onnx, onnxruntime, tritonclient
    triton_model_tools/                 — source code (subpackages below)
      models/                           — shared types: Detection, YoloModelVariant, YOLO_MODELS
      export/                           — YOLO → ONNX/TRT → Triton repository (+ templates/, scripts/)
      inference/                        — Triton gRPC client, pre/postprocessing, batch inference
      testing/                          — E2E test harness (Redis pipeline integration)
      benchmarks/                       — performance measurement, comparison, dataset tools
    tests/                              — tests (flat, not mirroring subpackage structure)
```

### Import Layer Architecture (import-linter contract)

```
__main__ (CLI) → benchmarks → testing → inference | export → models (leaf)
```

Modules within a `|` group are siblings at the same layer and must NOT cross-import.

### Infrastructure (.devcontainer/docker-compose.yml)

- **Redis** (`redis:7-alpine`) — port 6379, AOF persistence
- **Triton** (`nvcr.io/nvidia/tritonserver:26.02-py3`) — ports 8000 (HTTP), 8001 (gRPC), 8002 (metrics). Runs with `--strict-model-config=true` — all config.pbtxt files must be complete (Triton will not infer missing fields).
- **Triton SDK** (`nvcr.io/nvidia/tritonserver:26.02-py3-sdk`) — provides `perf_analyzer`, `model_analyzer`, and pre-installed `tritonclient` matching the server version. Shares workspace and model_repository volumes, reaches Triton via hostname `triton`.
- **RedisInsight** (`redis/redisinsight:latest`) — port 5540, Redis web UI

All infrastructure runs via the devcontainer compose file.

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

- Every pipeline node is automatically instrumented for `node_events_total`, `node_events_interval`, `node_processing_ms`, `node_errors` via `MetricAggregator`
- Use `Annotated[ProcessorType, Tracked("name")]` when registering parallel nodes to enable per-call metric collection in multi-threaded/multi-process wrappers
- When adding new error event types: add to `_ERROR_EVENT_TYPES` in `engine/app.py`
- When adding new metrics: emit in `_log_stats` for ALL registered nodes (zero-value guarantee — idle nodes must emit 0)
- Custom metrics use `_make_metric()` + `self._metric_buffer.record()` pattern in `engine/app.py`
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

Pipeline: ruff format → ruff check → mypy → pyright → import-linter → vulture → pytest → uv lock

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

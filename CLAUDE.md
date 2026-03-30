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
packages/
  triton_redis_connector/               — core connector
    pyproject.toml                      — deps: redis, tritonclient, pydantic, loguru, dishka
    triton_redis_connector/             — source code
    tests/                              — tests
  triton_model_tools/                   — model tools
    pyproject.toml                      — deps: ultralytics, onnx, onnxruntime, tritonclient
    triton_model_tools/                 — source code
    tests/                              — tests
```

### Infrastructure (.devcontainer/docker-compose.yml)

- **Redis** (`redis:7-alpine`) — port 6379, AOF persistence
- **Triton** (`nvcr.io/nvidia/tritonserver:25.02-py3`) — ports 8000 (HTTP), 8001 (gRPC), 8002 (metrics). Runs with `--strict-model-config=true` — all config.pbtxt files must be complete (Triton will not infer missing fields).
- **RedisInsight** (`redis/redisinsight:latest`) — port 5540, Redis web UI

All infrastructure runs via the devcontainer compose file.

## Code Style

- **Logging**: always `loguru`, never `print`
- **Configs/models**: always `pydantic`
- **Error handling**: never bare `except Exception` — catch specific types. Never silently swallow exceptions.
- **No dynamic access**: avoid `getattr`/`hasattr`/`__dict__`
- **Avoid comments**: improve names and code structure instead
- **Avoid short functions**: inline functions under 4 lines if used fewer than 4 times

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

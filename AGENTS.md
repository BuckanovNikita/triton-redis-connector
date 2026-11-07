# triton-redis-connector agent instructions

## Project

- triton-redis-connector: microservice for ML inference via Redis Streams + NVIDIA Triton.
- uv workspace monorepo under `packages/`:
  - `typed_flow` (typed-flow) — event-streaming framework.
  - `triton_shared_models` (triton-shared-models) — shared types, leaf; imports no other workspace package.
  - `triton_redis_connector` (triton-redis-connector) — Triton processors, depends on typed-flow.
  - `triton_model_tools` (triton-model-tools) — CLI: `uv run triton-model-tools <convert|predict|infer|benchmark|...>`.
- Python 3.12, strict typing: mypy strict and pyright strict.
- Only README.md is in Russian; all other docs are in English.

## Commands

Run every Python tool through `uv run`. Never call bare `python`, `python3`, `pip`, `pip3`,
`pytest`, `ruff`, `mypy`, `pyright`, `vulture`, `lint-imports` or `prek`, and never use
`.venv/bin/` paths or activate the venv.

```bash
uv sync --all-groups --extra triton-25-03   # setup; pick the extra matching TRITON_VERSION, see docs/triton-versions.md
uv run pytest                               # unit tests, parallel (-n auto); deselects `integration`
uv run prek run --all-files           # full gate
```

- Pre-commit pipeline: ruff format -> ruff check -> mypy -> pyright -> import-linter -> vulture -> pytest -> uv lock.
- For code changes, run `uv run prek run --all-files`; its pytest step satisfies
  the unit-test gate.
- Default pytest options deselect `integration` tests (they need Redis and Triton); select them with `uv run pytest -m integration` after checking the stack.
- Triton version switching: follow `docs/triton-versions.md`.

## Documentation map

| Document | Contents |
|---|---|
| [docs/architecture.md](docs/architecture.md) | Package structure, import-linter contracts, execution variants, domain event model, shipped processors, ModelBundle, metrics system, infrastructure |
| [docs/parallelism.md](docs/parallelism.md) | Execution variant decision guide, thread vs process selection |
| [docs/triton-versions.md](docs/triton-versions.md) | Triton version matrix, switching versions, `.env`-driven management |
| [docs/benchmark-results.md](docs/benchmark-results.md) | Performance measurement results |
| [packages/typed_flow/docs/](packages/typed_flow/docs/) | typed-flow framework docs: getting started, sources/sinks, DI, execution variants |

## Git and commits

- [Conventional Commits](https://www.conventionalcommits.org/). Accepted types: feat, fix, refactor, test, docs, chore, perf, ci, build, style, revert; optional `(scope)` and `!`.
- Never skip checks with `--no-verify`; fix the failures.
- Do not rely on a Git hook; run `uv run prek run --all-files` yourself before every commit. In Claude Code a PreToolUse hook also runs it on `git commit`.
- Push over SSH: `git@github.com:<org>/<repo>.git`.
- Multi-line messages: write a unique temp file `/tmp/COMMIT_MSG_<short_hash>`, run `git commit -F <path>`, then delete it.
- Parallel agents or worktrees: never commit from agents on the main branch; never use shared temp files such as `/tmp/COMMIT_MSG`; verify git state after each agent finishes.

## Code conventions

- **Logging**: always `loguru`, never `print`.
- **Configs/models**: always `pydantic` — never raw dicts for serialization.
- **Error handling**: never bare `except Exception` — catch specific types. Never silently swallow exceptions.
- **No dynamic access**: avoid `getattr`/`hasattr`/`__dict__` unless wrapping opaque third-party SDKs (with a comment explaining why).
- **Comments**: avoid comments; use them only as a last resort for truly hard code. Improve names and extract functions instead. (The explanatory comment required for dynamic access is the exception.)
- **Avoid too short functions**: inline functions under 4 lines if used fewer than 4 times and deduplication would not make code fragile.
- **Imports**: top-level, never lazy — unless the dependency is truly optional (not in pyproject.toml) or it is a CLI subcommand handler in `__main__.py`.
- **Config file sync**: when moving/renaming files, update ALL config paths: mypy `exclude`, pyright `exclude`, vulture `paths`/`ignore_names`, import-linter contracts.
- **No custom IoU/NMS**: use `torchvision.ops.box_iou` / `torchvision.ops.nms`.
- **No re-export modules**: import from the source package directly, never create proxy modules.
- **Explicit protocol inheritance**: processors/decoders must explicitly inherit protocol classes, not rely on structural typing.
- **Devcontainer**: if `whoami` prints `vscode`, you are inside the devcontainer and `docker`/`docker compose` do not work. STOP, give the user the host commands to run, and wait for the output (see `.claude/skills/check-infra/SKILL.md`).
- **Skill/config placement**: keep project skills, agents, hooks and settings in this repository (`.claude/` today), never in `~/.claude/`.

## Metrics policy

The app must be abstracted from domain metrics. No domain event types in `packages/typed_flow/typed_flow/engine/app.py`.

- **Counter on event classes**: `_metrics_ = [Counter("name")]` — counts occurrences.
- **Gauge on event fields**: `Annotated[float, Gauge("name")]` — extracts and emits values.
- **GroupBy on event fields**: `Annotated[str, GroupBy("name", window_s=60)]` — time-windowed per-key counting.
- **Counter/Gauge/GroupBy** live in `packages/typed_flow/typed_flow/protocols/metrics.py` (leaf layer).
- **Tracked on wrappers**: `Annotated[ProcessorType, Tracked("name")]` for per-call metrics (`typed_flow/engine/metric_aggregator.py`).
- **MetricEvent**: only 3 fields — `component_name`, `measure`, `value`.
- **Zero-value guarantee**: `MetricEmitter` (`typed_flow/metrics/emitter.py`) emits zero-value events for all registered Counter names even when idle.
- Update the metrics table in `README.md` (section "Метрики") when adding new metrics.

## Testing policy

Only write tests for real logic and edge cases. Never test:
- Framework-guaranteed behavior (frozen pydantic, enum literals, StrEnum coercion, dataclass defaults).
- Simple construction/defaults, protocol isinstance checks, constant counts.
- Generated file content (pbtxt, scripts, configs) — only integration tests verify these.
- Pydantic/framework validation, download/network tools, simple file I/O.

E2E means real infrastructure (Triton, Redis), not mocks.
Coverage target: 70% minimum. Exclude simple utility modules.
Good tests verify: validation logic, algorithms, error paths, serialization roundtrips, edge cases, integration behavior.

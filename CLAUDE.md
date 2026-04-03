# CLAUDE.md

## Project Overview

**triton-redis-connector** — microservice for ML inference via Redis Streams + NVIDIA Triton.

Monorepo (`packages/`): **typed-flow** (event-streaming framework), **triton-shared-models** (shared types, leaf), **triton-redis-connector** (Triton processors, depends on typed-flow), **triton-model-tools** (CLI: convert, infer, benchmark).

**Language**: Python 3.12, strict typing (mypy strict + pyright strict).
**Documentation**: Only README.md in Russian. All other docs in English.

## Detailed Documentation (docs/)

| Document | Contents |
|---|---|
| [docs/architecture.md](docs/architecture.md) | Package structure, import-linter contracts, execution variants, domain event model, shipped processors, ModelBundle, metrics system, infrastructure |
| [docs/parallelism.md](docs/parallelism.md) | Execution variant decision guide, thread vs process selection |
| [docs/triton-versions.md](docs/triton-versions.md) | Triton version matrix, switching versions, `.env`-driven management |
| [docs/benchmark-results.md](docs/benchmark-results.md) | Performance measurement results |
| [packages/typed_flow/docs/](packages/typed_flow/docs/) | typed-flow framework docs: getting started, sources/sinks, DI, execution variants |

## Skills (`.claude/skills/`)

| Skill | Trigger |
|---|---|
| `/commit` | Commit and push workflow with pre-commit |
| `/check-infra` | Check Docker services health (Redis, Triton) |
| `/scaffold-module` | Create new packages or submodules |
| `/verify-plan` | Verify implementation plan before execution |
| `/using-triton-inference-server` | Triton config.pbtxt, tritonclient API, model repository |
| `/using-nvidia-dali` | DALI GPU preprocessing pipelines |
| `/using-ultralytics` | YOLO predict/export reference |
| `/using-dishka` | Dishka DI container, providers, scopes, testing |
| `/uv-package-manager` | uv commands, workspace, deps, lockfile management |

## Agent Team (`.claude/agents/`)

| Agent | Role | When to use |
|---|---|---|
| **architect** | System architecture, data flow, API contracts | Planning features, designing integrations, architectural decisions |
| **implementer** | Production code, strict typing, project conventions | Implementing features, fixing bugs, writing modules |
| **tester** | Tests, mocks, fixtures, coverage | Adding tests, investigating failures, test infrastructure |
| **reviewer** | Code review, security, performance, conventions | After writing code, reviewing PRs, auditing quality |

## Git / Commits

- [Conventional Commits](https://www.conventionalcommits.org/): `feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `chore:`
- Pre-commit runs automatically before every commit (hook enforced)
- SSH for push: `git@github.com:<org>/<repo>.git`
- For multi-line commit messages, use unique temp files (`/tmp/COMMIT_MSG_<short_hash>`), then `git commit -F <path>`, then delete

## Pre-commit & Hooks

Pipeline: ruff format -> ruff check -> mypy -> pyright -> import-linter -> vulture -> pytest -> uv lock

Always run `uv run pytest` and `uv run pre-commit run --all-files` before considering work complete.

Claude Code hooks in `.claude/hooks/`:
- `block-bare-tools.sh` — blocks bare `python`/`pytest`/`ruff`/`mypy`, must use `uv run`
- `block-no-verify.sh` — blocks `--no-verify` on commits
- `validate-commit-msg.sh` — enforces conventional commit messages
- `pre-commit-guard.sh` — runs pre-commit before every commit
- `auto-ruff-format.sh` — auto-formats `.py` files on save
- `inject-session-context.sh` — injects TRITON_VERSION, branch, dirty state
- `setup-worktree.sh` — copies `.env`, runs `uv sync` in worktrees

## Multi-Agent / Parallel Work

When using parallel agents or worktrees: (1) never commit from agents on main branch, (2) never use shared temp files like `/tmp/COMMIT_MSG`, (3) verify git state after agent completion.

## Code Conventions

### Code Style

- **Logging**: always `loguru`, never `print`
- **Configs/models**: always `pydantic` — never raw dicts for serialization
- **Error handling**: never bare `except Exception` — catch specific types. Never silently swallow exceptions.
- **No dynamic access**: avoid `getattr`/`hasattr`/`__dict__` unless wrapping opaque third-party SDKs (with a comment explaining why)
- **Avoid comments in code**: use only as last resort for truly hard code. Improve names, extract functions instead.
- **Avoid too short functions**: inline functions under 4 lines if used fewer than 4 times and deduplication would not make code fragile
- **Imports**: top-level, never lazy — unless the dependency is truly optional (not in pyproject.toml) or it's a CLI subcommand handler in `__main__.py`
- **Config file sync**: when moving/renaming files, update ALL config paths: mypy `exclude`, pyright `exclude`, vulture `paths`/`ignore_names`, import-linter contracts
- **No custom IoU/NMS**: use `torchvision.ops.box_iou` / `torchvision.ops.nms`
- **No re-export modules**: import from source package directly, never create proxy modules
- **Explicit protocol inheritance**: processors/decoders must explicitly inherit protocol classes, not rely on structural typing
- **Skill/config placement**: project skills and config go in `.claude/` within the project directory, never in `~/.claude/`

### Metrics Policy

App must be abstracted from domain metrics. No domain event types in `engine/app.py`.

- **Counter on event classes**: `_metrics_ = [Counter("name")]` — counts occurrences
- **Gauge on event fields**: `Annotated[float, Gauge("name")]` — extracts and emits values
- **GroupBy on event fields**: `Annotated[str, GroupBy("name", window_s=60)]` — time-windowed per-key counting
- **Counter/Gauge/GroupBy** live in `protocols.py` (leaf layer)
- **Tracked on wrappers**: `Annotated[ProcessorType, Tracked("name")]` for per-call metrics
- **MetricEvent**: only 3 fields — `component_name`, `measure`, `value`
- **Zero-value guarantee**: `_log_stats` emits all registered Counter names even when idle
- Update metrics table in `README.md` (section "Метрики") when adding new metrics

### Testing Policy

Only write tests for real logic and edge cases. Never test:
- Framework-guaranteed behavior (frozen pydantic, enum literals, StrEnum coercion, dataclass defaults)
- Simple construction/defaults, protocol isinstance checks, constant counts
- Generated file content (pbtxt, scripts, configs) — only integration tests verify these
- Pydantic/framework validation, download/network tools, simple file I/O

E2E means real infrastructure (Triton, Redis), not mocks.
Coverage target: 70% minimum. Exclude simple utility modules.
Good tests verify: validation logic, algorithms, error paths, serialization roundtrips, edge cases, integration behavior.

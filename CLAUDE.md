# CLAUDE.md

## Project Overview

**triton-redis-connector** — microservice for ML inference via Redis Streams + NVIDIA Triton.

Monorepo (`packages/`): **typed-flow** (event-streaming framework), **triton-shared-models** (shared types, leaf), **triton-redis-connector** (Triton processors, depends on typed-flow), **triton-model-tools** (CLI: convert, infer, benchmark).

**Language**: Python 3.12, strict typing (mypy strict + pyright strict).
**Documentation**: Only README.md in Russian. All other docs in English.

## Development Commands

```bash
uv run pytest                          # all tests, parallel
uv run pytest -x                       # stop on first failure
uv run pre-commit run --all-files      # all checks in order
uv run ruff format . && uv run ruff check --fix .  # format + lint
uv run mypy . && uv run pyright        # type check
uv run lint-imports                    # architecture layer contracts
uv run vulture                         # dead code detection
```

Always run `uv run pytest` and pre-commit checks before considering work complete.

## Detailed Documentation (docs/)

| Document | Contents |
|---|---|
| [docs/architecture.md](docs/architecture.md) | Package structure, import-linter contracts, execution variants, domain event model, shipped processors, ModelBundle, metrics system, infrastructure |
| [docs/parallelism.md](docs/parallelism.md) | Execution variant decision guide, thread vs process selection |
| [docs/conventions.md](docs/conventions.md) | Code style, metrics policy, testing policy |
| [docs/triton-versions.md](docs/triton-versions.md) | Triton version matrix, switching versions, `.env`-driven management |
| [docs/benchmark-results.md](docs/benchmark-results.md) | Performance measurement results |

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

Use Pydantic models for all serialization — never raw dicts. All serialization logic internal to utility functions.
Project skills and config go in `.claude/` within the project directory, never in `~/.claude/`.

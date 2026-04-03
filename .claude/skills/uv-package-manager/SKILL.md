---
name: uv-package-manager
description: >
  Use when working with Python projects that use uv for dependency management,
  virtual environments, and running tools. Triggers on: "uv add", "uv sync",
  "uv run", adding/removing Python dependencies, setting up Python projects,
  managing lockfiles, or running dev tools via uv.
---

# uv Package Manager

Always use `uv run` to execute project tools — never activate the venv manually, never use bare `python`/`pip`/`pytest`/`ruff`/`mypy`, never use `.venv/bin/` paths.

## Essential commands

```bash
# Dependencies
uv add <package>                    # add to [project.dependencies]
uv add --group dev <package>        # add to [dependency-groups] dev
uv remove <package>                 # remove dependency
uv sync                             # install all deps from lockfile
uv sync --frozen                    # install without updating lock (CI)

# Lockfile
uv lock                             # resolve and write uv.lock
uv lock --upgrade                   # upgrade all to latest compatible
uv lock --upgrade-package <pkg>     # upgrade one package

# Running tools
uv run pytest                       # run in project venv (auto-creates if needed)
uv run ruff check .
uv run mypy .
uv run python script.py

# Python versions
uv python install 3.12              # install a Python version
uv python pin 3.12                  # write .python-version
uv venv --python 3.12               # create venv with specific version

# New project
uv init my-project                  # scaffold pyproject.toml + venv
```

## Project-specific commands

```bash
uv run pytest                          # all tests, parallel (-n auto)
uv run pytest -x                       # stop on first failure
uv run pre-commit run --all-files      # all checks in order
uv run ruff format . && uv run ruff check --fix .  # format + lint
uv run mypy . && uv run pyright        # type check (both strict)
uv run lint-imports                    # architecture layer contracts
uv run vulture                         # dead code detection
```

## pyproject.toml conventions

- Build backend: `hatchling` (default for `uv init`)
- Python requirement: `requires-python = ">=3.12"`
- Ruff: `select = ["ALL"]`, `line-length = 120`, `docstring-code-format = true`
- Mypy: `strict = true`, `warn_return_any = true`
- Pytest: `-v --tb=short -n auto` (parallel via xdist)
- Dev deps go in `[dependency-groups]` (not `[project.optional-dependencies]`)
- Always commit `uv.lock` to version control

## Workspace management

This is a uv workspace monorepo:

```toml
[tool.uv.workspace]
members = ["packages/*"]
```

- Each package under `packages/` has its own `pyproject.toml` with hatchling build backend
- Workspace packages are referenced via `[tool.uv.sources]` with `workspace = true`:

```toml
[tool.uv.sources]
typed-flow = { workspace = true }
triton-redis-connector = { workspace = true }
triton-model-tools = { workspace = true }
triton-shared-models = { workspace = true }
```

- After adding a new package: run `uv sync --dev` to update the lockfile
- All workspace packages are included in the `dev` dependency group for development

## Dependency groups

```toml
[dependency-groups]
lint = ["ruff", "mypy", "pyright", "vulture", "import-linter", ...]
test = ["pytest", "pytest-xdist", "pytest-cov", "pytest-asyncio"]
dev = [
    "typed-flow",               # workspace packages
    "triton-redis-connector",
    "triton-model-tools",
    "triton-shared-models",
    "pre-commit",
    { include-group = "lint" }, # compose groups
    { include-group = "test" },
]
```

- `lint` — type checkers and linters
- `test` — test runner and plugins
- `dev` — all workspace packages + lint + test + pre-commit (use `include-group` to compose)

## Conflict extras (Triton versions)

Mutually exclusive version sets for different Triton releases:

```toml
[project.optional-dependencies]
triton-25-03 = ["tritonclient[grpc]>=2.55,<2.56", "numpy>=1.26,<2", "torch>=2.6,<2.7"]
triton-26-02 = ["tritonclient[grpc]>=2.66,<2.67", "numpy>=2.4,<3", "torch>=2.11"]

[tool.uv]
conflicts = [
    [{ extra = "triton-25-03" }, { extra = "triton-26-02" }],
]
```

Install a specific Triton version: `uv sync --extra triton-26-02 --dev`
Switching versions: change `.env` `TRITON_VERSION`, then `uv sync --extra triton-<ver> --dev`

## Anti-patterns

Never do:
- `python script.py` — use `uv run python script.py`
- `pip install` — use `uv add`
- `pytest` — use `uv run pytest`
- `.venv/bin/ruff` — use `uv run ruff`
- `source .venv/bin/activate` — use `uv run` directly

## CI / Docker patterns

```bash
# CI: frozen install (fails if lock is stale)
uv sync --frozen --no-dev

# Docker: copy from official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
```

## Common issues

- **Lock out of sync** — `uv lock` (or `uv lock --upgrade`)
- **Wrong Python** — `uv python pin 3.12 && uv venv --python 3.12`
- **Cache bloat** — `uv cache clean`
- **Conflict resolution** — `uv lock --verbose` to see resolver output

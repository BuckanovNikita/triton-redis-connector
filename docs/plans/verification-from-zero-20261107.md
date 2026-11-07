# Verification from zero — 2026-11-07

## Why

On 2026-11-07 the user merged all started, unfinished, non-cancelled work into `main` without
verification. This plan drops all earlier partial verification evidence. Verify everything
again from a clean checkout of this range.

- Range: `9df80f25b63ec5e12ec2829562efb6fc71808033..89bb377` (repository root to the merge
  head). The commit that adds this file sits on top of that range.
- Merged on 2026-11-07:
  - `eea00b7`, the typed-flow lifecycle and Redis error-recovery test fix, which was
    previously local only.
  - `89bb377`, which ports parts of three branches:
    - `rescue/worktree-agent-af2f9205` (`3d4b2a9`): specific exception types in
      `test_error_paths.py`.
    - `origin/worktree-agent-ab20da4e` (`c11fd71`): the content-matching generated-config
      test is removed.
    - Of `origin/worktree-agent-a6197329` (`6f561e9`), only a docstring residual.
- Superseded by main and not merged:
  - `origin/chore/remove-redundant-error-tests-e2e` (`6fadb08`) and
    `origin/test/remove-redundant-error-model-tests` (`0fa0e3f`): removed by `ce8f51d`.
  - `origin/worktree-agent-a4841dbc` (`aa844be`) and the rest of `6f561e9`: replaced by
    `a6dfb90`.

## Supersedes

This plan replaces the following as current verification state. They remain as historical
records only.

- `docs/plans/typed-flow-runtime-acceptance-20261104/` (`plan.md`, `verification.md`,
  `review-1.md`, `review-2.md`, `architecture.json`, `check_test_boundaries.py`).
- `docs/plans/agent-instructions-refactor-20261104/architecture.json`.
- Any test counts, timings or "passed" statements in `docs/benchmark-results.md` or in
  commit messages within the range.

## Checklist (in order, from a clean checkout at HEAD)

### Setup

- [ ] Fresh clone or `git worktree add` at HEAD. Confirm that `git status` is clean and that
      HEAD contains `89bb377`.
- [ ] Pick the Triton version (`.env` `TRITON_VERSION`, see `docs/triton-versions.md`), then
      run `uv sync --frozen --all-groups --all-packages --extra triton-25-03` (or
      `triton-26-02`). Afterwards `uv.lock` must be unchanged.
- [ ] Confirm Python 3.12 (`uv run python -V`).

### Lint, typing and import contracts

- [ ] `uv run ruff format --check .`
- [ ] `uv run ruff check .`
- [ ] `uv run mypy .` (strict)
- [ ] `uv run pyright` (strict)
- [ ] `uv run lint-imports --no-cache`: all six contracts in `pyproject.toml` are kept, and
      `triton_shared_models` stays a leaf.
- [ ] `uv run vulture`
- [ ] `uv lock --check`, and `git diff --check`.

### Full unit suite

- [ ] `uv run pytest`: the default options run with `-n auto` and deselect `integration`
      and DALI.
- [ ] Repeat with `-W error::pytest.PytestUnhandledThreadExceptionWarning`. There must be no
      background-thread crashes. Use `packages/typed_flow/tests` first, then the whole
      suite.
- [ ] Coverage is at least 70% (`uv run pytest --cov`), excluding simple utility modules.
- [ ] Check the merged test edits by hand:
  - `test_error_paths.py` asserts `ZeroDivisionError` and `IndexError`.
  - `test_generated_config.py` keeps `test_triton_loads_generated_config`.

### Documented gate

- [ ] `uv run pre-commit run --all-files`: ruff format, ruff check, mypy, pyright,
      import-linter, vulture, pytest and uv lock all pass, with no file modified.
- [ ] The `.claude/hooks/` scripts still fire correctly: `block-no-verify.sh`,
      `block-bare-tools.sh`, `pre-commit-guard.sh` and `validate-commit-msg.sh` on
      `git commit` and tool calls, and `setup-worktree.sh` on new worktrees.

### Live, native and GPU acceptance

Do this step only after checking the stack with the `check-infra` skill. Outside the
devcontainer, run the commands on the host.

- [ ] Bring up Redis and Triton for the selected `TRITON_VERSION`, then run
      `uv run pytest -m integration`. This includes
      `tests/integration/test_generated_config.py` loading the generated config.
- [ ] Run GPU-marked tests and the DALI tests
      (`packages/triton_model_tools/tests/test_dali.py`) on a GPU host.
- [ ] Repeat the integration run for the other Triton version (25.03 and 26.02 pairing:
      tritonclient, numpy and CUDA matrix).
- [ ] Follow the manual CLI acceptance in `docs/test_manuals/` (convert, predict, infer,
      e2e, app, compare, benchmark, dynamic batching), and run
      `reference_implementation/verify.py` against the live stack.
- [ ] Compare benchmark results against `docs/benchmark-results.md`. Record new numbers as
      dated evidence; do not edit the evergreen instructions.

### Architecture and records acceptance

- [ ] Review `docs/architecture.md` and `docs/parallelism.md` against the code: package
      layering, execution variants, shipped processors, ModelBundle and the metrics
      policy.
- [ ] Confirm that `AGENTS.md`, `CLAUDE.md` and the `.claude/skills/*` files are consistent
      with each other and with the code (commands, paths and hooks).
- [ ] Check that the README metrics table ("Метрики") matches the registered
      Counter/Gauge/GroupBy names.
- [ ] Write a fresh, dated verification record next to this plan, then get an independent
      review of the whole range.

### Release and pairing

- [ ] No release automation or tags exist (all packages are at `0.1.0`, the root at
      `0.0.0`). Decide whether a version tag is needed. Never create GitHub Release objects
      or upload assets.
- [ ] After `main` passes, the primary checkout can fast-forward its local `main` (its
      `eea00b7` is contained in `main`). Leftover remote `worktree-agent-*` branches still
      need review before cleanup.

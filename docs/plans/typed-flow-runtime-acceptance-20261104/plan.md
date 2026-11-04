# Typed-flow runtime acceptance test repair

## Scope and observed failure

The restored checkout's typed-flow suite reports success despite background thread crashes. Strict `PytestUnhandledThreadExceptionWarning` handling reproduces two causes: lifecycle fixtures omit a `PipelineError` consumer, and the emitter failure mock raises the built-in exception instead of Redis's exception. Repair only these tests. Production behavior, dependency sources and historical branches remain unchanged.

## Architecture

[architecture.json](architecture.json) records the test boundaries. The production packages retain the six import-linter contracts in `pyproject.toml`. A scoped test-boundary check rejects imports of concrete Triton/model infrastructure and demonstrates rejection using an injected prohibited import. Tests own local threads and mocks; no Redis/Triton/GPU calls are permitted.

## Tasks

- [x] Inspect tests, implementation, dependency configuration and existing changes; reproduce both failures with warnings treated as errors.
- [x] Validate architecture plan before editing test implementations.
- [x] Repair the lifecycle fixture's error routing and assert event delivery before shutdown, with unconditional thread cleanup.
- [x] Simulate Redis's connection exception and verify a subsequent flush succeeds, with unconditional emitter cleanup.
- [x] Re-run strict regression tests, safe typed-flow unit tests, architecture, lint, formatting, scoped types and dead-code checks.
- [x] Parent verification and fresh independent review, reconcile architecture acceptance and documentation. Round2: ship; round1 cleanup finding resolved.
- [x] Complete task-scoped local Conventional Commit after required gates and final review; no branch cleanup or dependency changes. Delivery identity is in private portfolio evidence.

## Verification and release

Use the frozen-lock environment and `uv run --no-sync` for isolated verification. Run typed-flow pytest with `-W error::pytest.PytestUnhandledThreadExceptionWarning`, the six existing `lint-imports` contracts, the scoped boundary check, Ruff check/format check, typed-flow mypy and Vulture. Broader model/service integration remains an acceptance limitation until its intended profile and owned runtime are established. The project commit skill requires full pytest and pre-commit (Ruff formatting/check, mypy, pyright, import-linter, Vulture, pytest, uv lock), Conventional Commit and SSH push. No version/tag automation is defined. Parent owns release decision and must not claim skipped gates passed.

## Fresh-review correction

Review round 1 is `fix-first`; see [review-1.md](review-1.md). Sink delivery can precede stats timer startup. The first shutdown can therefore finish before the timer starts, and joining the run thread alone does not cancel that timer. Controlled reproduction gates timer startup until the test joins the run thread. Repeat shutdown after that join in each lifecycle test, then verify all three controlled races leave no uncancelled timers. Re-run strict targeted/unit checks and native full pre-commit before a fresh review. No production fix is included.

- [x] Reproduce late timer startup against all three lifecycle tests.
- [x] Cancel late timers after joining the run thread and verify the controlled race.
- [x] Repeat affected strict tests and full native pre-commit gates.
- [ ] Obtain fresh final review and complete architecture acceptance.

Source delivery: no project version/tag release mechanism exists. Apply the owner standing fallback: local task-scoped Conventional Commit only, without inventing tags, pushes or pull requests. The commit skill provides check/commit instructions; its generic push step does not override this explicit fallback. Historical unique branches remain untouched.

# Verification evidence — 2026-11-04

## Original observation and reproduction

At restored revision `face57c0ff86739aa7c23ad4ae22b7d3df08867d`, the typed-flow suite initially returned 245 passed with eight warnings. Four warnings were background thread crashes. Running the lifecycle startup test and Redis-error emitter test with `-W error::pytest.PytestUnhandledThreadExceptionWarning` reproduced two failures. The lifecycle graph lacked a `PipelineError` consumer; the emitter mock raised the built-in connection exception rather than Redis's connection exception.

## Integrated repair checks

Checks used the repository's frozen dependency lock, Python 3.12.13 and an isolated environment. All commands below returned exit 0:

- `uv run --no-sync pytest packages/typed_flow/tests/test_engine/test_app.py packages/typed_flow/tests/test_metrics/test_emitter.py -n 0 -W error::pytest.PytestUnhandledThreadExceptionWarning`: 16 passed. Parent independently repeated the same original-path checks, also 16 passed.
- `uv run --no-sync pytest packages/typed_flow/tests -n 2 -m "not integration and not gpu" -W error::pytest.PytestUnhandledThreadExceptionWarning`: 245 passed. Four existing Python process-fork deprecation warnings remain; there are no unhandled-thread warnings.
- `uv run --no-sync python docs/plans/typed-flow-runtime-acceptance-20261104/check_test_boundaries.py`: both test modules and checker pass; each forbidden import is injected as both a top-level and function-local import and correctly detected.
- `uv run --no-sync lint-imports --no-cache`: all six production contracts kept, 121 files and 294 dependencies analyzed.
- `uv run --no-sync ruff check .`: passed.
- `uv run --no-sync ruff format --check .`: 199 Python files already formatted.
- `uv run --no-sync mypy packages/typed_flow/typed_flow`: 42 source files passed strict typing.
- `uv run --no-sync vulture`: passed.
- `git diff --check`: passed.

## Full native checks

Full workspace dependencies installed using `uv sync --frozen --all-packages`. Native pytest ran in an isolated source copy containing the integrated task diff, with two auto workers and CUDA hidden. Config, model fixtures, ONNX export and output files were owned by that verification copy; no existing service, dataset or GPU workload was modified.

`uv run --frozen pytest -W error::pytest.PytestUnhandledThreadExceptionWarning`: 504 passed with seven non-thread warnings in 54.15 seconds. Four warnings are existing multithreaded-fork deprecations; the other three concern legacy ONNX export/tracing. Default configuration excludes integration and DALI tests; this does not establish live Redis/Triton or GPU acceptance.

Native `uv run --frozen pre-commit run --all-files` returned exit 0. Ruff format/check, full mypy, full pyright, all import-linter contracts, Vulture, pytest and uv lock all passed. The verification copy had no tracked-file modifications after hooks and its `uv.lock` remained byte-identical. Raw full command output is retained in dated external evidence. Fresh independent review and architecture acceptance remain pending; no commit, tag or push has been made.

## Diff reconciliation

Task-owned source changes comprise two typed-flow test modules. Added documentation/tooling is confined to this task directory. Production modules, dependencies, lockfile, historical branch content and default HEAD remain unchanged in the owner checkout. The initial report's clean-tree description applies only to the pre-edit investigation snapshot.

## Review round 1 correction

The first review rejected lifecycle cleanup because sink delivery occurs before stats timer startup. A deterministic reproducer delayed `_start_stats_timer` until the test attempted to join the run thread; all three lifecycle tests then left one uncancelled timer each. Reproduction-owned timers were cancelled and joined.

Each lifecycle cleanup now repeats `app.shutdown()` after joining `run_thread`, cancelling any timer started during the earlier shutdown. The same controlled interleaving now leaves zero uncancelled timers in all three tests (exit 0). Reproduction timers were also joined to confirm termination. This is test cleanup only; no production race fix is claimed.

Post-correction strict targeted checks pass 16 tests; strict full typed-flow checks pass 245 tests with four existing fork deprecation warnings and no unhandled-thread warnings. Native full pre-commit repeated for the corrected diff: exit 0; all eight hooks passed. The copy lockfile remains byte-identical and hooks introduced no tracked changes. First-round review remains rejected until fresh review accepts the final correction. The previous pre-review patch/manifest are superseded.

## Final parent acceptance

Fresh independent round2 review returned ship; see [review-2.md](review-2.md). Parent independently repeated16 strict affected tests and the controlled timer race, all passing with zero uncancelled timers in all3 cases. Parent reconciled all task-owned paths, checked local document links and validated architecture acceptance. Production and dependency configuration are unchanged. Local task commit is the standing fallback because no version/tag release mechanism is defined; no push or tag is invented.

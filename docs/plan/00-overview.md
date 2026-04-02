# New Round — Execution Steps Overview

Each file in this directory is an independent step. Execute them in order, one by one.
Each step must pass `uv run pre-commit run --all-files` before moving to the next.

| Step | File | Summary | Agent |
|------|------|---------|-------|
| 01 | `01-update-instructions.md` | Update CLAUDE.md + reviewer agent rules | implementer |
| 02 | `02-connector-simple-cleanup.md` | Drop docs_generator, hydra_app, letterbox re-export, NodeConfig/assembly, fix decoder, drop DI (providers/) | implementer |
| 03 | `03-model-tools-drop-files.md` | Drop perf_analyzer, memory, dataset, download-coco, preprocessing re-export, audit classification path | implementer |
| 04 | `04-drop-trivial-tests.md` | Remove ~53 trivial tests, update coverage to 70% | tester |
| 05 | `05-connector-fix-wiring.md` | Wire RetryMiddleware, delete ConcurrencyLimitMiddleware, verify TritonMetricsCollector, enforce ModelBundle | implementer |
| 06 | `06-model-tools-reduce-cli.md` | Merge CLIs, reduce to 7 tools, deduplicate (push_images + image loading + Redis init), NMS/rescale guards, all-scores verification | implementer |
| 06.5 | `06.5-export-defaults-and-tooling.md` | Dynamic batch default, tqdm, output paths, min images, COCO strategy, image replication | implementer |
| 07 | `07-split-protocols.md` | Split protocols.py into sub-package (6 modules) | implementer |
| 08 | `08-remove-recovery-ack.md` | Remove recovery/ack logic, add model version auto-detect | implementer |
| 09 | `09-root-config-cleanup.md` | APP_NAME as env var, drop model_repository_offline (all refs), verify config | implementer |
| 10 | `10-drop-linter-ignores.md` | Refactor to remove ~144 linter ignores (262 -> ~118) | implementer |
| 11 | `11-create-bash-scripts.md` | Create perf_analyze.sh, validate_models.sh, benchmark_all.sh + TRT variants | implementer |
| 11.5 | `11.5-examples-and-verification.md` | Long-test example, dynamic batching + DALI verification, low-threshold drift check | implementer |
| 11.6 | `11.6-dynamic-model-routing.md` | Dynamic per-event model routing + unified AutoRedisSource | architect -> implementer |
| 12 | `12-documentation.md` | Config docs, test manuals, benchmark tables | implementer |
| 13 | `13-final-review.md` | Full review + pre-commit + verification | reviewer |

**Decisions (locked):**
- DI (dishka): full removal — providers/ directory deleted, manual construction in main.py
- perf_analyzer: drop Python wrapper, use bash scripts
- Recovery/ack: full removal + model version auto-detect
- protocols.py: split into sub-package with backward-compat re-exports
- Tool count: keep 4 core + infer + e2e-test + compare-visual (7 total, user asked 4 but infer, e2e-test, and compare-visual are needed)
- Dynamic batching: enabled by default on export, with preferred_batch_size auto-set
- Output paths: project-local `e2e_results/` (gitignored), never /tmp
- Benchmark minimum: 5000 images default
- ConcurrencyLimitMiddleware: delete (dead code, never instantiated)
- Classification postprocessor path: audit and delete if dead code
- NMS/rescale: runtime guards against double application
- All-scores variant: add to comparison pipeline with `--check-all-scores` flag
- Image loading: single source (detection_io.py), no duplication across CLI files

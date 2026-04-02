# Step 13: Final Review

**Agent:** reviewer
**Depends on:** all previous steps

## Goal

Full review of all changes. Verify correctness, no regressions, all checks pass.

## Tasks

### 13.1 Run full check suite

```bash
uv run ruff format .
uv run ruff check .
uv run mypy .
uv run pyright
uv run lint-imports
uv run vulture
uv run pytest
uv run pytest --cov --cov-fail-under=70
uv run pre-commit run --all-files
```

All must pass with zero errors.

### 13.2 Verify architecture contracts

```bash
uv run lint-imports
```

Check that:
- protocols sub-package is still at leaf layer
- connector-no-model-tools contract holds
- shared-models-independence contract holds
- model-tools-layers contract holds (with deleted layers removed)

### 13.3 Verify dead code

```bash
uv run vulture
```

No new dead code introduced. Deleted files should not leave orphaned imports.

### 13.4 Verify linter ignore count

```bash
grep -r "noqa\|type: ignore\|pyright: ignore" packages/ | wc -l
```

Should be ~118 or less (down from 262).

### 13.5 Code review checklist

- [ ] No re-export modules remain — both connector `letterbox.py` and model-tools `inference/preprocessing.py` deleted (steps 02, 03)
- [ ] No `providers/` directory — dishka DI fully removed, manual construction in main.py (step 02.7)
- [ ] `dishka` removed from connector pyproject.toml dependencies (step 02.7)
- [ ] TurboJPEGDecoder is wired via DecoderType config — not dead code (step 02.6)
- [ ] Drift verification uses low thresholds (conf=0.1, iou=0.1) to expose full detection set (step 11.5.4)
- [ ] No `NodeConfig`, `NodeVariant`, or `assembly.py` — class_path in config dropped (step 02)
- [ ] No custom IoU/NMS implementations (replaced with torchvision in step 03)
- [ ] Classification postprocessor path audited — dead code deleted or documented (step 03)
- [ ] No trivial pydantic validation tests remain (deleted in step 04)
- [ ] RetryMiddleware actually wraps InferenceWorker (fixed in step 05)
- [ ] ConcurrencyLimitMiddleware deleted — confirmed dead code (step 05)
- [ ] TritonMetricsCollector is always started in pipeline (verified in step 05)
- [ ] ModelBundle is the standard model definition path (step 05)
- [ ] CLI has exactly 7 commands: convert, predict, infer, compare-folders, compare-visual, benchmark, e2e-test
- [ ] CLI directory has exactly 7 command files + __init__.py + _types.py — no orphaned files (step 06)
- [ ] No duplicate `_push_images` — consolidated into shared utility (step 06)
- [ ] No duplicate image loading — `collect_image_paths()` consolidated into single source (step 06)
- [ ] Runtime guard: end2end models cannot enter _postprocess_standard NMS path (step 06)
- [ ] Runtime guard: double rescale (server + client) logged as warning (step 06)
- [ ] All-scores variant drift check supported via `--check-all-scores` flag in compare-folders (step 06)
- [ ] protocols/ sub-package has backward-compatible re-exports (step 07)
- [ ] No recovery/ack code remains in connectors (step 08)
- [ ] Model version auto-detection is wired (step 08)
- [ ] APP_NAME reads from env var only (step 09)
- [ ] No model_repository_offline references in ANY file — including agent .md files, docs/architecture.md, pyproject.toml vulture config (step 09)
- [ ] Linter ignores reduced by >50% (step 10)
- [ ] Bash scripts are executable and have correct flags (step 11)
- [ ] Bash benchmark script includes TRT variants, skips only auxiliary models (step 11)
- [ ] Dynamic batching enabled by default on export, preferred_batch_size auto-set (step 06.5)
- [ ] tqdm progress bars on all iteration loops (step 06.5)
- [ ] Output paths default to `e2e_results/`, not `/tmp` (step 06.5)
- [ ] Benchmark default minimum 5000 images (step 06.5)
- [ ] COCO acquisition documented in test manuals (step 06.5)
- [ ] Image replication script exists for non-standard model sizes (step 06.5)
- [ ] compare-visual CLI kept, fiftyone_export.py kept (step 06)
- [ ] Long-test example with Redis populate cycle exists (step 11.5)
- [ ] Dynamic batching + DALI verification documented (step 11.5)
- [ ] Config docs match actual schema — no NodeConfig listed (step 12)
- [ ] Test manuals cover all tools (step 12)

### 13.6 Manual verification (if Triton available)

1. Load models to Triton via scripts/validate_models.sh
2. Run inference: `uv run triton-model-tools infer -m yolo11s --url localhost:8001 -i test_fixtures/images/ -o /tmp/test_infer`
3. Run local inference: `uv run triton-model-tools infer -m yolo11s --local -i test_fixtures/images/ -o /tmp/test_local`
4. Compare: `uv run triton-model-tools compare-folders /tmp/test_infer /tmp/test_local`
5. Benchmark: `uv run triton-model-tools benchmark -m yolo11s --url localhost:8001`
6. Verify no drift in results

### 13.7 Commit

After all checks pass:
```bash
git add -A
git commit -m "refactor: new round cleanup — drop dead code, fix wiring, reduce CLI, split protocols"
git push
```

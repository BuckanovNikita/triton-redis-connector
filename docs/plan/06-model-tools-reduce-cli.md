# Step 06: Model Tools — Reduce CLI Tools

**Agent:** implementer
**Depends on:** step 04

## Goal

Reduce from 15 CLI commands to 6 by merging and dropping. Deduplicate inference code.

## Final CLI commands (7)

| Command | Source | Changes |
|---------|--------|---------|
| `convert` | cli/convert.py | Keep as-is |
| `predict` | cli/predict.py | Keep as-is |
| `infer` | cli/infer.py | Add `--local` flag (merge local-infer) |
| `compare-folders` | cli/compare_folders.py | Keep as-is |
| `compare-visual` | cli/compare_visual.py | Keep as-is (FiftyOne visual drift comparison) |
| `benchmark` | cli/benchmark.py | Add `--pipeline` flag (merge pipeline-benchmark) |
| `e2e-test` | cli/e2e_test.py | Merge populate-stream functionality |

## Tasks

### 6.1 Merge `local-infer` into `infer`

**Modify:** `cli/infer.py` — add `--local` / `--backend onnx` flag
**Modify:** `inference/folder_inference.py` — add ONNX backend support (parameterize inference function)
**Delete:** `cli/local_infer.py`
**Delete:** `inference/local_folder_inference.py`
**Update:** `inference/onnx_inference.py` — keep (used by folder_inference with local backend)

The `_folder_inference_common.py` already has shared image loop logic. Extend it to accept a backend parameter.

### 6.2 Merge `pipeline-benchmark` into `benchmark`

**Modify:** `cli/benchmark.py` — add `--pipeline` flag
**Modify:** `benchmarks/benchmark.py` — add pipeline benchmark mode (Redis -> Triton -> Redis)
**Delete:** `cli/pipeline_benchmark.py`
**Delete:** `benchmarks/pipeline_benchmark.py` — merge core logic into benchmark.py

### 6.3 Merge `populate-stream` into `e2e-test`

**Modify:** `cli/e2e_test.py` — add `--populate-only` flag (push images to Redis without processing)
**Delete:** `cli/populate_stream.py`

### 6.4 Drop remaining CLIs

**Delete:**
- `cli/validate.py` — replaced by bash script (step 11)
- `cli/pipeline_predict.py` — pipeline predict is niche, drop
- `cli/benchmark_all.py` — replaced by bash script (step 11)

**Keep:**
- `cli/compare_visual.py` — FiftyOne visual drift comparison (needed for COCO-based visual verification)
- `benchmarks/fiftyone_export.py` — used by compare_visual, keep as-is

**Delete supporting modules:**
- `testing/pipeline_predict.py` — if no longer used after CLI drop

### 6.5 Update `__main__.py`

Remove all deleted command registrations. Final list:
- `convert`, `predict`, `infer`, `compare-folders`, `compare-visual`, `benchmark`, `e2e-test`

### 6.6 Update `api.py`

Remove any public API functions that referenced deleted modules.

### 6.7 Verify NMS and postprocessing — no duplication, no gaps

- Confirm `inference/postprocessing.py` uses only `triton_shared_models.nms` (torchvision) — no custom NMS
- Confirm `export/scripts/nms_postprocess.py` uses only `torchvision.ops.nms` — no custom NMS
- Check `export/scripts/rescale_postprocess.py` — ensure rescale is applied exactly once (no double rescale)
- Verify `YoloDetailedPostprocessor` (all-scores variant) is included in drift verification

**Add runtime guard against double NMS:**

In `inference/postprocessing.py`, `_postprocess_end2end()` already skips NMS for end2end models. Add an explicit log warning if an end2end model is somehow passed to `_postprocess_standard()`:
```python
if self._end2end:
    logger.warning("end2end model should use _postprocess_end2end, not _postprocess_standard")
```

**Add guard against double rescale:**

In the export pipeline, verify that when a model has `rescale_postprocess` in the ensemble, the client-side postprocessor does NOT also apply rescale. Add a check in `YoloPostprocessor.__init__()`:
- If `end2end=True` (model handles rescale internally), `scale` and `pad_offset` should not be passed or should be identity (1.0, (0,0))
- Log warning if both server-side rescale AND client-side rescale params are set

### 6.7.1 Add all-scores variant check to verification pipeline

**File:** `packages/triton_model_tools/triton_model_tools/benchmarks/comparison.py`

Add support for comparing `all_class_scores` field in `Detection` objects when both sides have it populated. The `compare_detections()` or `_compute_image_metrics()` function should:
- When both prediction and ground-truth have `all_class_scores is not None`, compare score vectors (cosine similarity or max absolute diff)
- Report `all_scores_drift` metric alongside precision/recall/IoU
- When only one side has `all_class_scores`, skip the check but log a warning

**File:** `packages/triton_model_tools/triton_model_tools/cli/compare_folders.py`

Add `--check-all-scores` flag (default: False) to enable all-scores comparison in drift verification. When enabled, load detailed detections and include score drift in the report.

### 6.8 Consolidate `_push_images` and image loading duplication

**A. _push_images duplication:**

Three separate implementations exist for pushing images to Redis streams:
- `cli/populate_stream.py:88` — `_push_images()` with metadata fields
- `benchmarks/pipeline_benchmark.py:58` — `_push_images()` with timestamp recording
- `testing/e2e_test.py:101` — `push_test_images()` method

After step 06.3 merges `populate-stream` into `e2e-test`, two copies remain (`pipeline_benchmark` and `e2e_test`).

**Action:** Extract a shared `push_images_to_stream()` utility in `testing/e2e_test.py` (or a new `testing/redis_publisher.py`) that both `pipeline_benchmark` and `e2e_test` call. Accept a callback parameter for timestamp recording.

**B. Image loading duplication:**

`collect_image_paths()` and similar image loading logic is duplicated between:
- `inference/detection_io.py` — `collect_image_paths()`, `load_image()`
- `cli/_types.py` — `parse_image_size()` and inline image path collection

Consolidate into `inference/detection_io.py` as the single source. All CLI files and benchmarks should import from there.

**C. Redis connection setup duplication:**

`redis.Redis.from_url()` pattern is duplicated across `cli/populate_stream.py`, `benchmarks/pipeline_benchmark.py`, etc. After merge this should converge, but verify no duplicate Redis init code remains.

### 6.9 Delete orphaned test files and CLI files

After deleting CLI/module files, check for orphaned tests:
- `tests/test_pipeline_perf.py` — if tests pipeline-benchmark, delete
- `tests/test_pipeline_drift_vs_ultralytics.py` — keep if it tests core comparison logic
- Any test importing deleted modules

**Clean up orphaned CLI files:** After merging commands, verify all deleted CLI `.py` files are actually removed from `cli/` directory. The directory currently has ~17 files — after this step it should have exactly 7 command files + `__init__.py` + `_types.py` (shared utilities). List orphan candidates:
- `cli/local_infer.py` (merged into infer)
- `cli/pipeline_benchmark.py` (merged into benchmark)
- `cli/populate_stream.py` (merged into e2e-test)
- `cli/validate.py` (replaced by bash script)
- `cli/pipeline_predict.py` (dropped)
- `cli/benchmark_all.py` (replaced by bash script)
- `cli/perf_analyze.py` (dropped in step 03)
- `cli/download_coco.py` (dropped in step 03)

## Verification

```bash
uv run pre-commit run --all-files
uv run pytest packages/triton_model_tools/
uv run triton-model-tools --help  # verify 7 commands listed
```

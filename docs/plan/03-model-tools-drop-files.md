# Step 03: Model Tools — Drop Dead Files

**Agent:** implementer
**Depends on:** step 01

## Goal

Delete modules and CLI commands from `triton_model_tools` that are being replaced by bash scripts or removed entirely.

## Tasks

### 3.1 Drop perf_analyzer Python wrapper

**Delete:**
- `packages/triton_model_tools/triton_model_tools/benchmarks/perf_analyzer.py`
- `packages/triton_model_tools/triton_model_tools/cli/perf_analyze.py`
- `packages/triton_model_tools/tests/test_perf_analyzer.py` (if exists)

**Update:**
- `__main__.py` — remove `perf-analyze` command registration
- `benchmarks/__init__.py` — remove perf_analyzer exports (if any)
- `api.py` — remove perf_analyzer references (if any)

### 3.2 Replace custom `compute_iou` with torchvision

**File:** `packages/triton_model_tools/triton_model_tools/benchmarks/comparison.py`

- Delete `compute_iou()` function (lines 40-57)
- Replace usage in `match_detections()` and `_compute_image_metrics()` with `torchvision.ops.box_iou()`
- Convert Detection boxes to tensor: `torch.tensor([[d.x1, d.y1, d.x2, d.y2] for d in detections])`
- `box_iou()` returns NxM matrix — use it for greedy matching

**Test update:**
- Delete `TestComputeIou` class from `tests/test_comparison.py`

### 3.3 Drop memory monitoring

**Delete:** `packages/triton_model_tools/triton_model_tools/benchmarks/memory.py`

**Update:**
- `benchmarks/benchmark.py` — remove `MemoryMonitor` import and usage, remove `peak_vram_mb`/`peak_ram_mb` from results
- `cli/benchmark.py` — remove `--max-memory-gb` CLI argument

### 3.4 Drop download tools

**Delete:**
- `packages/triton_model_tools/triton_model_tools/cli/download_coco.py`
- `packages/triton_model_tools/triton_model_tools/benchmarks/dataset.py`
- `packages/triton_model_tools/tests/test_dataset.py`

**Update:**
- `__main__.py` — remove `download-coco` command registration
- `benchmarks/__init__.py` — remove dataset exports
- Any other file importing from `dataset.py` — check `benchmarks/benchmark.py` for `download_coco_val` usage and remove

### 3.5 Check classification postprocessor path usage

**Files:**
- `packages/triton_model_tools/triton_model_tools/inference/classification_postprocessing.py`
- `packages/triton_redis_connector/triton_redis_connector/inference/classification_bridge.py`

Check if `ClassificationPostprocessor`, `DetailedClassificationPostprocessor`, `ClassificationPostProcessorBridge`, and `DetailedClassificationPostProcessorBridge` are used anywhere in production code or examples. If dead code — delete. If used — keep but document in CLAUDE.md.

### 3.6 Drop `inference/preprocessing.py` re-export module

**File:** `packages/triton_model_tools/triton_model_tools/inference/preprocessing.py` (15 lines)

This is a pure re-export (same antipattern as connector's `letterbox.py`):
```python
from triton_shared_models.preprocessing import LetterBoxPreprocessor as YoloPreprocessor
```

**Fix all 7 consumers to import from source:**
- `inference/_folder_inference_common.py`
- `inference/folder_inference.py`
- `inference/local_folder_inference.py`
- `inference/__init__.py`
- `benchmarks/benchmark.py`
- `tests/test_pipeline_drift_vs_ultralytics.py`
- `tests/test_error_paths.py`

Replace: `from triton_model_tools.inference.preprocessing import YoloPreprocessor`
With: `from triton_shared_models.preprocessing import LetterBoxPreprocessor as YoloPreprocessor`

Or keep the alias inline at each call site. Then delete `inference/preprocessing.py`.

**Update:** `inference/__init__.py` — remove preprocessing re-exports

## Verification

```bash
uv run pre-commit run --all-files
uv run pytest packages/triton_model_tools/
```

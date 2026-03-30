# Phase 2 — Issues and Deviations

**Date**: 2026-03-30
**Baseline**: commit `b9bdf21` (feat: add TensorRT export support)
**Verification**: 340/340 tests pass, all 8 pre-commit hooks pass

---

## Summary

| # | Severity | Deliverable | Description |
|---|---|---|---|
| 1 | **HIGH** | 2.4 | DALI letterbox padding is top-left, not centered like ultralytics |
| 2 | **HIGH** | 2.4+2.5 | Missing combined DALI + NMS ensemble wiring |
| 3 | **HIGH** | 2.5 | No end2end model filtering — end2end models should skip NMS |
| 4 | **MEDIUM** | 2.1 | `_render_model_warmup` produces invalid pbtxt for multiple inputs |
| 5 | **MEDIUM** | 2.6 | CLI `--dali-ensemble` and `--nms` flags not combined into unified ensemble |
| 6 | **MEDIUM** | 2.6 | Test coverage for `__main__.py` at 73%, below 75% threshold |
| 7 | **LOW** | 2.1 | `_render_ensemble_scheduling` uses multiple `step []` blocks (valid but unconventional) |
| 8 | **LOW** | 2.5 | NMS output dims fixed `[1, 300, 6]` but actual output is variable length |
| 9 | **LOW** | 2.5 | Unused `json` import and `model_config` variable in generated NMS script |
| 10 | **LOW** | 2.5 | Empty `finalize()` method in generated NMS script |

---

## HIGH Severity

### Issue 1: DALI letterbox centering (Phase 2.4)

**File**: `packages/triton_model_tools/triton_model_tools/dali.py:86-91`

The unified plan states in the "CRITICAL: Ultralytics Preprocessing in DALI" section:
> "LetterBox: Resize maintaining aspect ratio, pad shorter dimension with (114, 114, 114) gray to target size. **Image centered.**"

Current implementation uses `fn.pad(resized, axis_names="HW", shape=[target_h, target_w], fill_value=114)` which pads at the **bottom-right** by default. Ultralytics `LetterBox` **centers** the image (pads equally on both sides).

**Impact**: Detection coordinates will be offset. Phase 3 cross-validation test (`np.allclose` between `YoloPreprocessor` and ultralytics) will fail if the client-side preprocessor centers and DALI does not.

**Fix**: Replace `fn.pad` with `fn.paste` or manually compute top/left offsets to center the image within the padded canvas.

---

### Issue 2: Missing DALI + NMS combined ensemble (Phase 2.4 + 2.5)

**Files**:
- `packages/triton_model_tools/triton_model_tools/dali.py` (missing combined builder)
- `packages/triton_model_tools/triton_model_tools/__main__.py:91-114` (flags independent)

The unified plan line 341 requires:
> "DALI + NMS combined ensemble: `yolo11s_dali_preprocess -> yolo11s -> nms_postprocess -> yolo11s_dali_ensemble`"

This 3-step ensemble does not exist. Currently:
- `--dali-ensemble` creates a 2-step ensemble: `dali_preprocess -> inference`
- `--nms` creates a standalone `nms_postprocess` model

They are never combined. If a user passes both flags, they get separate disconnected pieces.

**Fix**: Add a `build_dali_nms_ensemble_config()` function and wire it in the CLI when both `--dali-ensemble` and `--nms` are specified.

---

### Issue 3: No end2end model filtering for NMS (Phase 2.5)

**Files**:
- `packages/triton_model_tools/triton_model_tools/models.py:45-51`
- `packages/triton_model_tools/triton_model_tools/__main__.py:106-111`
- `packages/triton_model_tools/tests/test_nms.py` (missing test)

The unified plan states:
> - `yolo26s` standalone (end2end export, NMS baked into ONNX graph)
> - Detection of end2end vs standard models is via the `end2end_export` flag in the model registry.
> - Test: "NMS ensemble wiring: standard models route through nms_postprocess, end2end models skip it"

Currently:
- `yolo26s` has `export_kwargs={"end2end": True}` but no explicit `end2end_export: bool` field on `YoloModelVariant`
- CLI `--nms` creates NMS globally with no model awareness
- No code inspects the end2end flag to skip NMS for specific models
- No test validates this behavior

**Fix**: Add `end2end_export: bool` field to `YoloModelVariant`. In CLI, skip NMS wiring for models where `end2end_export=True`. Add tests.

---

## MEDIUM Severity

### Issue 4: `_render_model_warmup` invalid pbtxt for multiple inputs (Phase 2.1)

**File**: `packages/triton_shared_models/triton_shared_models/triton_config.py:169-196`

The `inputs` field in Triton's `ModelWarmup` protobuf message is a `map<string, ModelWarmupInput>`. Each map entry must be its own `inputs { key: "..." value { ... } }` block. Current code puts all key/value pairs inside a single `inputs { }` block.

**Current output** (2+ inputs):
```
model_warmup [{
    name: "warmup"
    batch_size: 1
    inputs {
      key: "images"
      value { ... }
      key: "shapes"
      value { ... }
    }
}]
```

**Correct output**:
```
model_warmup [{
    name: "warmup"
    batch_size: 1
    inputs {
      key: "images"
      value { ... }
    }
    inputs {
      key: "shapes"
      value { ... }
    }
}]
```

**Why not caught**: All tests use single-input warmup configs.

**Fix**: Move the `inputs {` / `}` delimiters inside the for-loop body. Add multi-input test case.

---

### Issue 5: CLI flags not combined into unified ensemble (Phase 2.6)

**File**: `packages/triton_model_tools/triton_model_tools/__main__.py:91-114`

When both `--dali-ensemble` and `--nms` are specified, the CLI creates them independently instead of wiring them into a single ensemble pipeline. Related to Issue 2 but from the CLI perspective.

---

### Issue 6: CLI test coverage below threshold (Phase 2.6)

**File**: `packages/triton_model_tools/tests/test_cli.py`

Coverage for `__main__.py` is 73%, below the 75% `fail-under` threshold. Missing tests for:
1. `_run_export` with `--dali-ensemble` flag (DALI code path)
2. `_run_export` with `--nms` flag (NMS code path)
3. `_run_export` with explicit `--models` list
4. `KeyboardInterrupt` handling in `main()`

**Note**: Coverage check may not be enforced in the current pre-commit pipeline (pytest runs without `--cov`), so this doesn't block CI today but is a gap.

---

## LOW Severity

### Issue 7: Unconventional `step []` rendering (Phase 2.1)

**File**: `packages/triton_shared_models/triton_shared_models/triton_config.py:199-219`

Each ensemble step gets its own `step [ ... ]` block instead of all steps in a single `step [ ... ]` array. This is technically valid protobuf text (repeated fields can appear multiple times), but unconventional. Most Triton examples use a single array.

**Impact**: None for correctness. Minor readability concern.

---

### Issue 8: NMS output dims are fixed (Phase 2.5)

**File**: `packages/triton_model_tools/triton_model_tools/nms.py:193`

Output dims are `[1, config.max_detections, 6]` (fixed). The actual NMS output is variable length `[1, N, 6]` where `N <= max_detections`. For dynamic output, dims should use `-1` for the detection count dimension: `[1, -1, 6]`.

**Impact**: Triton's Python backend is lenient (response tensor shape overrides config), but the config is misleading.

---

### Issue 9: Unused code in generated NMS script (Phase 2.5)

**File**: `packages/triton_model_tools/triton_model_tools/nms.py:49-50,141`

- `import json` is unused in the generated script
- `model_config = json.loads(args["model_config"])` is assigned but never read

---

### Issue 10: Empty `finalize()` in generated NMS script (Phase 2.5)

**File**: `packages/triton_model_tools/triton_model_tools/nms.py:164-165`

`finalize(self)` is just `pass`. Triton does not require `finalize()`. Dead code in the generated script.

---

## Positive Findings

- **All 340 tests pass** in 1.49s across 16 parallel workers
- **All 8 pre-commit hooks pass**: ruff format, ruff check, mypy, pyright, import-linter, vulture, pytest, uv lock
- **Phase 2.1**: All deliverables present (ExportFormat, mappings, extended TritonModelConfig, is_shape_tensor)
- **Phase 2.2**: `model_source` + `model_filename` correctly replace old `onnx_source`
- **Phase 2.3**: Both `_export_onnx()` and `_export_tensorrt()` present with correct naming convention
- **Phase 2.4**: DALI pipeline implements all 6 preprocessing steps (decode, resize, pad, color convert, normalize, transpose)
- **Phase 2.5**: NMS with torchvision + numpy fallback, correct xywh->xyxy, confidence filtering
- **Phase 2.6**: Subcommand CLI with all required flags
- **Architecture**: Clean shared types in `triton_shared_models` package
- **Type safety**: Full mypy strict + pyright strict compliance
- **No security concerns**: Offline CLI tools, no network input

# Step 12: Documentation

**Agent:** implementer
**Depends on:** steps 10, 11

## Goal

Write config documentation, test manuals for each tool, and update benchmark/drift tables.

## Tasks

### 12.1 Write `docs/configuration.md`

Full config reference for `AppConfig` and all sub-configs. For each config model:
- Field name, type, default value
- Description and valid range
- Example YAML snippet
- Environment variable overrides (if any)

**Config models to document:**
- `AppConfig` (top-level)
- `RedisConnectionConfig`
- `SourceConfig` (post recovery removal — simplified)
- `SinkConfig`
- `InferenceConfig`
- `PreprocessingConfig`
- `QueueConfig`
- `MetricsConfig`
- `DebugConfig`
- `PipelineConfig`

Note: `NodeConfig` was deleted in step 02.4 — do NOT document it.

Include example `config.yaml` with all defaults shown.

### 12.2 Write test manuals

Create `docs/test_manuals/` directory with one file per tool/app:

**`docs/test_manuals/test_convert.md`** — Model conversion testing:
- Prerequisites (YOLO .pt files, Triton running)
- Test ONNX export, TensorRT export, NMS postprocess, DALI ensemble, rescale
- Verify output: config.pbtxt, model files, directory structure
- Test all model variants from default_models.yaml

**`docs/test_manuals/test_infer.md`** — Inference testing:
- Test Triton inference (gRPC)
- Test local ONNX inference (`--local` flag)
- Test with different image sizes, batch scenarios
- Verify detection output format

**`docs/test_manuals/test_predict.md`** — Ultralytics prediction testing:
- Test prediction on sample images
- Verify txt output format matches expected
- Compare with Triton inference results

**`docs/test_manuals/test_compare.md`** — Comparison testing:
- Test folder comparison (baseline vs candidate)
- Verify precision/recall/IoU metrics
- Test with identical folders (should be 1.0/1.0/1.0)
- Test with empty/missing detections

**`docs/test_manuals/test_benchmark.md`** — Benchmark testing:
- Test single model benchmark
- Test pipeline benchmark (`--pipeline` flag)
- Verify output JSON structure
- Test with different concurrency levels

**`docs/test_manuals/test_e2e.md`** — E2E testing:
- Test full Redis pipeline flow
- Test populate-only mode
- Verify results in Redis output stream
- Test with Redis down (error handling)

**`docs/test_manuals/test_app.md`** — Connector app testing:
- Test hydra config loading
- Test Application.build() with ModelBundle
- Test Redis stream ingestion
- Test metrics emission
- Test graceful shutdown

### 12.3 Update benchmark/drift tables

**File:** `docs/benchmark-results.md`
- Update with latest results for ALL model variants
- Include pipeline benchmark results
- Include dynamic batching results (if working)

**Create:** `docs/model-status.md`
- Status table for each model variant: loads? drift-free? benchmark results?
- Format:

| Model | Loads | Drift Check | FPS (c=1) | FPS (c=4) | Notes |
|-------|-------|-------------|-----------|-----------|-------|
| yolo11s | PASS | PASS | 87 | 106 | Baseline |
| yolo11s_fp16 | PASS | PASS | 47 | 66 | FP16 slower on RTX 5090 |
| ... | ... | ... | ... | ... | ... |

## Verification

```bash
# Verify markdown renders correctly
# Check all internal links work
# Verify config docs match actual schema
uv run pre-commit run --all-files
```

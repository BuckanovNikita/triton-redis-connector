# Dynamic Batching Verification

Manual verification steps for dynamic batching functionality, including
DALI pipeline ensemble verification.

## Prerequisites

- Triton server running in explicit model control mode
- At least one YOLO model weights file (.pt)
- Test images (COCO val2017 recommended, see `docs/test_manuals/coco_images.md`)
- `triton-model-tools` installed (`uv sync --dev`)

## Step 1: Export Model with Dynamic Batching

Dynamic batching is enabled by default. Export a model:

```bash
uv run triton-model-tools convert \
    --weights yolo11s.pt \
    --repository-root model_repository \
    --max-batch-size 8 \
    --preferred-batch-sizes 1,2,4,8
```

To verify dynamic batching is disabled when explicitly requested:

```bash
uv run triton-model-tools convert \
    --weights yolo11s.pt \
    --repository-root model_repository \
    --no-dynamic-batching
```

## Step 2: Verify config.pbtxt

Check that the generated `config.pbtxt` contains the `dynamic_batching` section:

```bash
cat model_repository/yolo11s/config.pbtxt
```

Expected output should include:

```
max_batch_size: 8

dynamic_batching {
  preferred_batch_size: [ 1, 2, 4, 8 ]
}
```

When `--no-dynamic-batching` was used, the `dynamic_batching` section must be
absent and `max_batch_size` should be `0`.

## Step 3: Load Model in Triton

```bash
curl -X POST http://localhost:8000/v2/repository/models/yolo11s/load

curl http://localhost:8000/v2/models/yolo11s/config
```

Verify the response includes `"dynamic_batching"` in the model configuration.

## Step 4: Run Concurrent Inference

Run the benchmark tool with multiple concurrent requests to exercise batching:

```bash
uv run triton-model-tools benchmark \
    --image-dir benchmarks/data/coco_val2017 \
    --model yolo11s \
    --concurrency 8 \
    --warmup-images 50
```

Alternatively, run inference from multiple terminals simultaneously:

```bash
# Terminal 1
uv run triton-model-tools infer \
    --image-dir benchmarks/data/coco_val2017 \
    --model yolo11s

# Terminal 2
uv run triton-model-tools infer \
    --image-dir benchmarks/data/coco_val2017 \
    --model yolo11s
```

## Step 5: Check Triton Metrics for Batch Distribution

Query the Prometheus metrics endpoint for batch size statistics:

```bash
curl -s http://localhost:8002/metrics | grep -E "batch_size|queue_duration"
```

Key metrics to look for:

- `nv_inference_request_success` -- total successful requests
- `nv_inference_exec_count` -- number of inference executions (lower than
  request count means batching is working)
- `nv_inference_queue_duration_us` -- time spent in the dynamic batching queue

If `nv_inference_exec_count` is significantly lower than
`nv_inference_request_success`, requests are being batched together.

## Step 6: Unload Model

```bash
curl -X POST http://localhost:8000/v2/repository/models/yolo11s/unload
```

---

## DALI Pipeline Verification

### Step 1: Export Model with DALI Ensemble

```bash
uv run triton-model-tools convert \
    --weights yolo11s.pt \
    --repository-root model_repository \
    --dali-ensemble \
    --imgsz 640
```

This creates three model directories:

1. `yolo11s/` -- the base ONNX model
2. `yolo11s_dali_preprocess/` -- DALI preprocessing model
3. `yolo11s_dali_ensemble/` -- ensemble combining DALI + inference

### Step 2: Verify Generated Models

Check that all three directories exist and have valid configs:

```bash
ls model_repository/yolo11s*/
cat model_repository/yolo11s_dali_preprocess/config.pbtxt
cat model_repository/yolo11s_dali_ensemble/config.pbtxt
```

The DALI preprocess config should have `backend: "dali"` and the ensemble
config should have `ensemble_scheduling` with steps referencing both the
DALI preprocess and the inference model.

Verify the DALI Python pipeline script exists:

```bash
cat model_repository/yolo11s_dali_preprocess/1/model.dali
```

### Step 3: Load All Ensemble Models

Load models in dependency order (base models first, ensemble last):

```bash
curl -X POST http://localhost:8000/v2/repository/models/yolo11s/load
curl -X POST http://localhost:8000/v2/repository/models/yolo11s_dali_preprocess/load
curl -X POST http://localhost:8000/v2/repository/models/yolo11s_dali_ensemble/load
```

Verify all are loaded:

```bash
curl http://localhost:8000/v2/repository/index
```

### Step 4: Run Inference Through Ensemble

Use the ensemble model name for inference:

```bash
uv run triton-model-tools infer \
    --image-dir benchmarks/data/coco_val2017 \
    --model yolo11s_dali_ensemble
```

Note: DALI backend may not work on all GPU architectures. If the DALI preprocess
model fails to load, check Triton server logs for compatibility errors. On
Blackwell (sm_120) GPUs, DALI may have known issues with certain Triton versions.

### Step 5: Unload Ensemble Models

Unload in reverse order (ensemble first, then dependencies):

```bash
curl -X POST http://localhost:8000/v2/repository/models/yolo11s_dali_ensemble/unload
curl -X POST http://localhost:8000/v2/repository/models/yolo11s_dali_preprocess/unload
curl -X POST http://localhost:8000/v2/repository/models/yolo11s/unload
```

### DALI with NMS and Rescale

For a full DALI + NMS + rescale ensemble:

```bash
uv run triton-model-tools convert \
    --weights yolo11s.pt \
    --repository-root model_repository \
    --dali-ensemble \
    --nms \
    --rescale
```

This generates additional models (`nms_postprocess`, `rescale_postprocess`)
and the ensemble config wires them together in the correct order.

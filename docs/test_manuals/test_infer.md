# Inference Testing

Manual verification steps for the `infer` command that runs YOLO inference
on a folder of images via Triton gRPC or local ONNX runtime.

## Prerequisites

- Triton server running with a loaded YOLO model (for Triton mode)
- ONNX model file (for local mode)
- Test images (COCO val2017 recommended, see `docs/test_manuals/coco_images.md`)
- `triton-model-tools` installed (`uv sync --dev`)

## Triton Mode

### Step 1: Basic Inference

```bash
uv run triton-model-tools infer \
    --image-dir benchmarks/data/coco_val2017 \
    --model yolo11s
```

Expected: per-image detection logs with class IDs, confidences, and bounding
boxes. Default output saved to `e2e_results/infer/`.

### Step 2: Custom Thresholds

```bash
uv run triton-model-tools infer \
    --image-dir benchmarks/data/coco_val2017 \
    --model yolo11s \
    --conf-threshold 0.1 \
    --iou-threshold 0.1 \
    --output-dir e2e_results/infer_low_thresh
```

Verify: more detections per image compared to default thresholds.

### Step 3: End-to-End Model

```bash
uv run triton-model-tools infer \
    --image-dir benchmarks/data/coco_val2017 \
    --model yolo11s_end2end \
    --end2end
```

### Step 4: Auto-Detection from Repository

```bash
uv run triton-model-tools infer \
    --image-dir benchmarks/data/coco_val2017 \
    --model yolo11s \
    --repository-root model_repository
```

Verify: log shows auto-detected `end2end` and `target_size` values from
the model's `config.pbtxt`.

### Step 5: Custom Target Size

```bash
uv run triton-model-tools infer \
    --image-dir benchmarks/data/coco_val2017 \
    --model yolo11s \
    --target-size 960x608
```

### Step 6: Custom Triton URL

```bash
uv run triton-model-tools infer \
    --image-dir benchmarks/data/coco_val2017 \
    --model yolo11s \
    --triton-url triton:8001
```

## Local ONNX Mode

### Step 7: Local ONNX Inference

```bash
uv run triton-model-tools infer \
    --image-dir benchmarks/data/coco_val2017 \
    --model model_repository/yolo11s/1/model.onnx \
    --local
```

Expected: inference runs using ONNX Runtime without Triton. Per-image
detection logs printed. Results saved to `e2e_results/infer/`.

### Step 8: Local with Thresholds

```bash
uv run triton-model-tools infer \
    --image-dir benchmarks/data/coco_val2017 \
    --model model_repository/yolo11s/1/model.onnx \
    --local \
    --conf-threshold 0.5 \
    --output-dir e2e_results/infer_local
```

## Verification

Compare Triton vs local results to ensure consistency:

```bash
uv run triton-model-tools compare-folders \
    --baseline-dir e2e_results/infer_local \
    --candidate-dir e2e_results/infer \
    --model yolo11s
```

Precision and recall should be close to 1.0 for ONNX FP32 models.

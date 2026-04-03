# Model Conversion Testing

Manual verification steps for the `convert` command that exports YOLO .pt
weights to Triton model repository format.

## Prerequisites

- YOLO weights file (e.g. `yolo11s.pt`)
- Triton server running in explicit model control mode
- `triton-model-tools` installed (`uv sync --dev`)

## Step 1: Basic ONNX Export

```bash
uv run triton-model-tools convert \
    --weights yolo11s.pt \
    --repository-root model_repository
```

Expected output:
- `model_repository/yolo11s/` directory created
- `model_repository/yolo11s/1/model.onnx` exists
- `model_repository/yolo11s/config.pbtxt` generated with `max_batch_size: 8`
  and `dynamic_batching` section

## Step 2: Export with Custom Options

### Custom name and image size

```bash
uv run triton-model-tools convert \
    --weights yolo11s.pt \
    --repository-root model_repository \
    --name my_yolo \
    --imgsz 960x608
```

Verify: `model_repository/my_yolo/config.pbtxt` has input dims matching 960x608.

### FP16 precision

```bash
uv run triton-model-tools convert \
    --weights yolo11s.pt \
    --repository-root model_repository \
    --half
```

### End-to-end NMS

```bash
uv run triton-model-tools convert \
    --weights yolo11s.pt \
    --repository-root model_repository \
    --end2end
```

### Disabled dynamic batching

```bash
uv run triton-model-tools convert \
    --weights yolo11s.pt \
    --repository-root model_repository \
    --no-dynamic-batching
```

Verify: `config.pbtxt` has `max_batch_size: 0` and no `dynamic_batching` section.

## Step 3: TensorRT Export

```bash
uv run triton-model-tools convert \
    --weights yolo11s.pt \
    --repository-root model_repository \
    --format tensorrt
```

Verify: `model_repository/yolo11s/1/model.plan` exists and
`config.pbtxt` has `platform: "tensorrt_plan"`.

## Step 4: Export Both Formats

```bash
uv run triton-model-tools convert \
    --weights yolo11s.pt \
    --repository-root model_repository \
    --format all
```

Verify: both `model_repository/yolo11s/` (ONNX) and
`model_repository/yolo11s_trt/` (TensorRT) directories created.

## Step 5: Export with DALI Ensemble

```bash
uv run triton-model-tools convert \
    --weights yolo11s.pt \
    --repository-root model_repository \
    --dali-ensemble
```

Verify three directories created:
1. `yolo11s/` -- base ONNX model
2. `yolo11s_dali_preprocess/` -- DALI preprocessing model
3. `yolo11s_dali_ensemble/` -- ensemble combining DALI + inference

## Step 6: Export with NMS and Rescale

```bash
uv run triton-model-tools convert \
    --weights yolo11s.pt \
    --repository-root model_repository \
    --nms --rescale
```

Verify `nms_postprocess/` and `rescale_postprocess/` directories created.

## Step 7: Export with Local Predict Verification

```bash
uv run triton-model-tools convert \
    --weights yolo11s.pt \
    --repository-root model_repository \
    --image-dir benchmarks/data/coco_val2017 \
    --output-dir e2e_results/verify
```

Verify: detection txt files saved to `e2e_results/verify/` and log shows
image count, total time, and average time per image.

## Step 8: Load and Verify in Triton

After export, load the model and verify it works:

```bash
curl -X POST http://localhost:8000/v2/repository/models/yolo11s/load
curl http://localhost:8000/v2/models/yolo11s/config
```

Run a quick inference to confirm:

```bash
uv run triton-model-tools infer \
    --image-dir benchmarks/data/coco_val2017 \
    --model yolo11s
```

## Step 9: TRT Optimization Flag

```bash
uv run triton-model-tools convert \
    --weights yolo11s.pt \
    --repository-root model_repository \
    --trt-optimize
```

Verify: `config.pbtxt` contains TensorRT execution accelerator configuration
in the `optimization` section.

## Cleanup

Unload models after testing:

```bash
curl -X POST http://localhost:8000/v2/repository/models/yolo11s/unload
```

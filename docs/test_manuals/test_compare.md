# Drift Detection Verification

Manual verification steps for comparing Triton inference results against
ultralytics baseline using low confidence and IoU thresholds.

## Why Low Thresholds Matter

Default thresholds (`--conf-threshold 0.25`, `--iou-threshold 0.45`) filter out
many detections before comparison. This hides drift that occurs in the
lower-confidence range, which is exactly where ONNX export rounding, TensorRT
quantization, and preprocessing differences manifest.

Using low thresholds (`0.1` / `0.1`) catches:

- **Numerical drift**: FP32 vs FP16 rounding differences that push a detection
  just above or below the confidence boundary
- **Preprocessing drift**: letterbox padding, resize interpolation, or
  normalization differences between ultralytics and the Triton pipeline
- **NMS behavior differences**: different NMS implementations (PyTorch vs ONNX
  vs TensorRT) produce slightly different suppression results, especially near
  the IoU threshold boundary
- **Coordinate quantization**: bounding box coordinates may shift by 1-2 pixels
  during export, which only surfaces in IoU comparisons with many detections

## Prerequisites

- Triton server running with the model loaded
- Test images (COCO val2017 recommended)
- Original YOLO .pt weights
- `triton-model-tools` installed (`uv sync --dev`)

## Step 1: Run Triton Inference with Low Thresholds

```bash
uv run triton-model-tools infer \
    --image-dir benchmarks/data/coco_val2017 \
    --model yolo11s \
    --conf-threshold 0.1 \
    --iou-threshold 0.1 \
    --output-dir e2e_results/triton_low_thresh
```

## Step 2: Run Ultralytics Predict with Same Thresholds

```bash
uv run triton-model-tools predict \
    --image-dir benchmarks/data/coco_val2017 \
    --model yolo11s.pt \
    --conf-threshold 0.1 \
    --iou-threshold 0.1 \
    --output-dir e2e_results/predict_low_thresh
```

Both commands save detection results as per-image `.txt` files in YOLO format
(`class_id confidence x1 y1 x2 y2`).

## Step 3: Compare Results

```bash
uv run triton-model-tools compare-folders \
    --baseline-dir e2e_results/predict_low_thresh \
    --candidate-dir e2e_results/triton_low_thresh \
    --model yolo11s \
    --iou-threshold 0.5 \
    --output e2e_results/comparison_low_thresh.json
```

The comparison IoU threshold (`0.5`) controls how strictly bounding boxes must
overlap to be considered a match. This is separate from the NMS IoU threshold
used during inference.

## Step 4: Interpret Results

The comparison output includes:

- **Precision**: fraction of Triton detections that match a baseline detection
- **Recall**: fraction of baseline detections matched by Triton
- **Average IoU**: mean IoU of matched detection pairs
- **Class mismatch rate**: fraction of matched pairs where the class ID differs
- **False positives**: Triton detections with no baseline match
- **Missed detections**: baseline detections with no Triton match

### Acceptable Drift Ranges

| Metric | Acceptable (ONNX FP32) | Acceptable (TensorRT FP16) |
|---|---|---|
| Precision | > 0.98 | > 0.95 |
| Recall | > 0.98 | > 0.95 |
| Average IoU | > 0.95 | > 0.90 |
| Class mismatch rate | < 0.01 | < 0.03 |

TensorRT FP16 models have wider acceptable ranges because FP16 quantization
introduces more numerical drift than ONNX FP32 export.

## Step 5: Visual Comparison (Optional)

For detailed visual inspection of differences, use FiftyOne:

```bash
uv run triton-model-tools compare-visual \
    --baseline-dir e2e_results/predict_low_thresh \
    --candidate-dir e2e_results/triton_low_thresh \
    --model yolo11s \
    --image-dir benchmarks/data/coco_val2017
```

This opens a FiftyOne session in the browser where you can visually inspect
images with the highest detection differences.

## Comparing Default vs Low Thresholds

To see why low thresholds matter, run the same comparison with default
thresholds and compare:

```bash
# Default thresholds
uv run triton-model-tools infer \
    --image-dir benchmarks/data/coco_val2017 \
    --model yolo11s \
    --output-dir e2e_results/triton_default

uv run triton-model-tools predict \
    --image-dir benchmarks/data/coco_val2017 \
    --model yolo11s.pt \
    --output-dir e2e_results/predict_default

uv run triton-model-tools compare-folders \
    --baseline-dir e2e_results/predict_default \
    --candidate-dir e2e_results/triton_default \
    --model yolo11s \
    --output e2e_results/comparison_default.json
```

The default-threshold comparison will typically show higher precision and recall
(closer to 1.0) because only high-confidence detections survive, and those tend
to agree between implementations. The low-threshold comparison reveals the true
drift magnitude across the full confidence spectrum.

## End-to-End Model Comparison

For models exported with `--end2end` (NMS baked into the ONNX graph):

```bash
uv run triton-model-tools infer \
    --image-dir benchmarks/data/coco_val2017 \
    --model yolo11s_end2end \
    --end2end \
    --conf-threshold 0.1 \
    --iou-threshold 0.1 \
    --output-dir e2e_results/triton_e2e_low

uv run triton-model-tools compare-folders \
    --baseline-dir e2e_results/predict_low_thresh \
    --candidate-dir e2e_results/triton_e2e_low \
    --model yolo11s \
    --output e2e_results/comparison_e2e_low.json
```

End-to-end models may show different NMS behavior because the NMS runs inside
the ONNX graph (using ONNX operators or TensorRT plugins) rather than in the
PyTorch-based ultralytics NMS. Expect slightly wider drift ranges.

# Ultralytics Prediction Testing

Manual verification steps for the `predict` command that runs ultralytics
YOLO prediction on an image folder and saves detection results as txt files.

## Prerequisites

- YOLO weights file (e.g. `yolo11s.pt`) -- downloaded automatically by
  ultralytics if not present
- Test images (COCO val2017 recommended, see `docs/test_manuals/coco_images.md`)
- `triton-model-tools` installed (`uv sync --dev`)

## Step 1: Basic Prediction

```bash
uv run triton-model-tools predict \
    --image-dir benchmarks/data/coco_val2017
```

Expected: detection txt files saved to `e2e_results/predict/` (default).
Log shows image count, total time, and average time per image.

## Step 2: Custom Model

```bash
uv run triton-model-tools predict \
    --image-dir benchmarks/data/coco_val2017 \
    --model yolo11m.pt
```

## Step 3: Custom Thresholds

```bash
uv run triton-model-tools predict \
    --image-dir benchmarks/data/coco_val2017 \
    --conf-threshold 0.1 \
    --iou-threshold 0.1 \
    --output-dir e2e_results/predict_low_thresh
```

Verify: more detections per image compared to default thresholds (0.25/0.45).

## Step 4: Custom Target Size

```bash
uv run triton-model-tools predict \
    --image-dir benchmarks/data/coco_val2017 \
    --target-size 960x608 \
    --output-dir e2e_results/predict_960
```

## Step 5: Verify Output Format

Check that each txt file has one detection per line in YOLO format:

```
class_id confidence x1 y1 x2 y2
```

```bash
head -5 e2e_results/predict/000000000139.txt
```

Each line should have 6 space-separated values: integer class ID, float
confidence, and four float bounding box coordinates (x1, y1, x2, y2 in
pixel coordinates).

## Usage in Drift Detection

The `predict` command serves as the baseline reference for drift detection.
Compare its output against Triton inference results using `compare-folders`.
See `docs/test_manuals/test_compare.md` for the full drift detection workflow.

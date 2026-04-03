# E2E Testing

Manual verification steps for the `e2e-test` command that pushes images to
Redis, runs them through the connector + Triton pipeline, and collects results.

## Prerequisites

- Triton server running with a loaded YOLO model
- Redis server running
- Connector application running with the model configured
- Test images (COCO val2017 recommended, see `docs/test_manuals/coco_images.md`)
- `triton-model-tools` installed (`uv sync --dev`)

## Full E2E Test

### Step 1: Basic E2E Test

```bash
uv run triton-model-tools e2e-test \
    --image-dir benchmarks/data/coco_val2017 \
    --model yolo11s
```

Expected: images pushed to Redis input stream, connector processes them via
Triton, results collected from output stream. Detection txt files saved to
`e2e_results/e2e_test/`. Log shows per-image detection details.

### Step 2: Custom Streams and Timeout

```bash
uv run triton-model-tools e2e-test \
    --image-dir benchmarks/data/coco_val2017 \
    --model yolo11s \
    --redis-url redis://redis:6379 \
    --input-stream camera_images \
    --output-stream detection_results \
    --consumer-group e2e-test \
    --timeout 60.0
```

### Step 3: End-to-End Model

```bash
uv run triton-model-tools e2e-test \
    --image-dir benchmarks/data/coco_val2017 \
    --model yolo11s_end2end \
    --end2end
```

### Step 4: Custom Output Directory

```bash
uv run triton-model-tools e2e-test \
    --image-dir benchmarks/data/coco_val2017 \
    --model yolo11s \
    --output-dir e2e_results/e2e_custom
```

## Populate-Only Mode

The `--populate-only` flag pushes images to the Redis stream without running
the full E2E pipeline. Useful for pre-loading test data.

### Step 5: Populate Stream

```bash
uv run triton-model-tools e2e-test \
    --image-dir benchmarks/data/coco_val2017 \
    --model yolo11s \
    --populate-only
```

Expected: images pushed to Redis stream with fields `camera_no`, `image_path`,
`frame_counter`, `receipt_ts`, `metal_type`, `event_type`.

### Step 6: Populate with Verification

```bash
uv run triton-model-tools e2e-test \
    --image-dir benchmarks/data/coco_val2017 \
    --model yolo11s \
    --populate-only \
    --verify
```

Verify: log confirms all messages have complete fields.

### Step 7: Populate with Limits

```bash
uv run triton-model-tools e2e-test \
    --image-dir benchmarks/data/coco_val2017 \
    --model yolo11s \
    --populate-only \
    --count 50 \
    --camera-count 5 \
    --event-type steel_frame
```

Verify: only 50 images pushed, camera IDs cycle through 1-5,
`event_type` field is `steel_frame`.

## Verifying Results

After a full E2E test, compare results against baseline:

```bash
uv run triton-model-tools predict \
    --image-dir benchmarks/data/coco_val2017 \
    --output-dir e2e_results/predict_baseline

uv run triton-model-tools compare-folders \
    --baseline-dir e2e_results/predict_baseline \
    --candidate-dir e2e_results/e2e_test \
    --model yolo11s
```

Precision and recall should be close to 1.0, confirming the Redis pipeline
does not introduce drift beyond the Triton inference itself.

## Troubleshooting

- **Timeout errors**: increase `--timeout` value or check that the connector
  application is running and processing the input stream
- **No results collected**: verify the connector's output stream name matches
  `--output-stream`
- **Missing detections**: check that the model is loaded in Triton
  (`curl http://localhost:8000/v2/repository/index`)

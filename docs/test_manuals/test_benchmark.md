# Benchmark Testing

Manual verification steps for the `benchmark` command that measures inference
performance via direct Triton gRPC calls or E2E through the Redis pipeline.

## Prerequisites

- Triton server running with a loaded YOLO model
- Test images (COCO val2017 recommended, see `docs/test_manuals/coco_images.md`)
- `triton-model-tools` installed (`uv sync --dev`)
- For pipeline mode: Redis server running, connector application running

## Direct Triton Benchmark

### Step 1: Basic Benchmark

```bash
uv run triton-model-tools benchmark \
    --image-dir benchmarks/data/coco_val2017 \
    --model yolo11s
```

Expected: log output showing number of images, average latency (ms),
throughput (fps), and percentile latencies (p50, p95, p99). Results saved
to `e2e_results/benchmark/results.json`.

### Step 2: Custom Parameters

```bash
uv run triton-model-tools benchmark \
    --image-dir benchmarks/data/coco_val2017 \
    --model yolo11s \
    --num-images 500 \
    --warmup-runs 20 \
    --triton-url triton:8001 \
    --output e2e_results/benchmark/yolo11s_500.json
```

### Step 3: TensorRT Backend

```bash
uv run triton-model-tools benchmark \
    --image-dir benchmarks/data/coco_val2017 \
    --model yolo11s \
    --backend tensorrt
```

### Step 4: End-to-End Model

```bash
uv run triton-model-tools benchmark \
    --image-dir benchmarks/data/coco_val2017 \
    --model yolo11s_end2end \
    --end2end
```

### Step 5: Auto-Detection from Repository

```bash
uv run triton-model-tools benchmark \
    --image-dir benchmarks/data/coco_val2017 \
    --model yolo11s \
    --repository-root model_repository
```

Verify: log shows auto-detected `end2end` and `target_size` values.

### Step 6: Minimum Image Count Warning

```bash
uv run triton-model-tools benchmark \
    --image-dir benchmarks/data/coco_val2017 \
    --model yolo11s \
    --num-images 100 \
    --min-images 5000
```

Verify: warning logged when image count is below the minimum recommendation.

## Pipeline E2E Benchmark

### Step 7: Basic Pipeline Benchmark

Requires the connector application running with the model configured.

```bash
uv run triton-model-tools benchmark \
    --image-dir benchmarks/data/coco_val2017 \
    --model yolo11s \
    --pipeline \
    --redis-url redis://redis:6379
```

Expected: pushes images to Redis input stream, waits for results on output
stream, measures end-to-end latency including Redis transport overhead.

### Step 8: Pipeline with Custom Streams

```bash
uv run triton-model-tools benchmark \
    --image-dir benchmarks/data/coco_val2017 \
    --model yolo11s \
    --pipeline \
    --redis-url redis://redis:6379 \
    --input-stream camera_images \
    --output-stream detection_results \
    --num-iterations 5 \
    --timeout 120
```

### Step 9: Full Matrix Benchmark

```bash
uv run triton-model-tools benchmark \
    --image-dir benchmarks/data/coco_val2017 \
    --model yolo11s \
    --pipeline \
    --full-matrix \
    --output e2e_results/benchmark/matrix.json
```

Runs benchmark across all backend and preprocess mode combinations.

### Step 10: Matrix with All Models

```bash
uv run triton-model-tools benchmark \
    --image-dir benchmarks/data/coco_val2017 \
    --model yolo11s \
    --pipeline \
    --full-matrix \
    --all \
    --repository-root model_repository
```

Discovers all models in the repository and runs the full matrix for each.

## Interpreting Results

Key metrics in the JSON output:

- `avg_latency_ms` -- average inference latency per image
- `throughput_fps` -- images processed per second
- `p50_latency_ms`, `p95_latency_ms`, `p99_latency_ms` -- latency percentiles
- `num_images` -- total images processed

For pipeline benchmarks, latency includes Redis transport overhead on top of
Triton inference time. Direct Triton benchmarks measure pure inference + pre/post
processing.

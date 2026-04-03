# Connector Application Testing

Manual verification steps for testing the triton-redis-connector application
that bridges Redis Streams and Triton Inference Server.

## Prerequisites

- Triton server running in explicit model control mode
- Redis server running
- A YOLO model exported to `model_repository/` (see `docs/test_manuals/test_convert.md`)
- Test images (COCO val2017 recommended, see `docs/test_manuals/coco_images.md`)
- Both packages installed (`uv sync --dev`)

## Step 1: Load Model in Triton

```bash
curl -X POST http://localhost:8000/v2/repository/models/yolo11s/load
curl http://localhost:8000/v2/models/yolo11s | python3 -m json.tool
```

Verify the model is in `READY` state.

## Step 2: Start the Connector Application

The connector reads configuration from Hydra YAML files in
`packages/triton_redis_connector/triton_redis_connector/conf/`.

```bash
cd /workspace
uv run python -m triton_redis_connector
```

Expected: application starts, logs show pipeline components registered,
metric collection started, and Redis stream sources listening.

## Step 3: Push Test Images via E2E Tool

In a separate terminal:

```bash
uv run triton-model-tools e2e-test \
    --image-dir benchmarks/data/coco_val2017 \
    --model yolo11s \
    --populate-only \
    --count 10 \
    --verify
```

## Step 4: Verify Processing

Watch the connector logs for processing activity:
- Images read from input stream
- Preprocessing (decode + letterbox)
- Triton inference requests sent
- Postprocessing (NMS + confidence filter)
- Results written to output stream

## Step 5: Collect Results

```bash
uv run triton-model-tools e2e-test \
    --image-dir benchmarks/data/coco_val2017 \
    --model yolo11s \
    --output-dir e2e_results/app_test
```

This pushes images and collects detection results from the output stream.

## Step 6: Check Metrics

The connector emits metrics to a Redis Stream. Check the metrics stream:

```bash
redis-cli XRANGE metrics_stream - + COUNT 10
```

Expected metrics include per-node throughput, latency, and domain event
counters.

## Step 7: Graceful Shutdown

Send SIGINT (Ctrl+C) to the connector process.

Expected: application logs graceful shutdown, all worker threads stop,
Redis connections closed, metric buffers flushed.

## Step 8: Error Handling

### Missing model

Start the connector with a model name that is not loaded in Triton. Verify
error events are logged (not crashes) and the application stays running.

### Redis connection failure

Stop Redis while the connector is running. Verify the connector logs
connection errors and attempts reconnection.

### Malformed input

Push a Redis stream message with missing required fields:

```bash
redis-cli XADD camera_images '*' camera_no 1 event_type image
```

Verify the connector logs a processing error for the malformed message
without crashing.

## Step 9: Multi-Model Pipeline

For pipelines with multiple models, verify each model processes its
events independently. Use `ModelBundle` + `expand_bundle()` to define
per-model processor sets.

See: `examples/dual_model_pipeline/` for a reference implementation.

## Cleanup

```bash
curl -X POST http://localhost:8000/v2/repository/models/yolo11s/unload
```

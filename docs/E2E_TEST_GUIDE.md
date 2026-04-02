# E2E Test Guide

Manual testing playbook. Sections follow the natural lifecycle:
export model → validate repo → verify Triton loaded → local baseline → local drift check →
Triton inference → Triton drift check → benchmarks → Redis streams → pipeline E2E → monitoring → cleanup.

---

## Prerequisites

### 1. Environment Setup

```bash
cd /home/nkt/triton-redis-connector

cat .env
# Expected: TRITON_VERSION=26.02

uv sync --dev

uv run python test_fixtures/create_test_images.py
uv run python test_fixtures/restore_models.py
```

### 2. Start Infrastructure

```bash
docker compose -f .devcontainer/docker-compose.debug.yml up -d
docker compose -f .devcontainer/docker-compose.debug.yml ps
```

Health checks:
```bash
curl -f http://localhost:8000/v2/health/ready
curl -f http://localhost:8000/v2/health/live
docker compose -f .devcontainer/docker-compose.debug.yml exec redis redis-cli ping
```

If Triton shows `unhealthy` — check logs:
```bash
docker compose -f .devcontainer/docker-compose.debug.yml logs triton
```

Debug compose uses `/v2/health/live` so Triton passes even without GPU.

### 3. Download Test Images (Optional)

```bash
# Small subset for quick testing
uv run triton-model-tools download-coco --target-dir ./data --max-images 50

# Full COCO val2017 for benchmarks
uv run triton-model-tools download-coco --target-dir ./data
```

Or use the synthetic fixtures (`test_fixtures/images/` — 8 images, created in step 1).

### 4. Docker Redis CLI Alias

All Redis commands in this guide use `docker compose exec`. Set this alias once:
```bash
alias dredis='docker compose -f .devcontainer/docker-compose.debug.yml exec redis redis-cli'
```

Then use `dredis XLEN stream_name` etc. Commands below show the full form for copy-paste.

---

## Section A: Quality Checks (No Infrastructure Needed)

### A1. Run All Pre-commit Hooks

```bash
uv run pre-commit run --all-files
```

Expected: all 8 hooks pass (ruff format, ruff check, mypy, pyright, import-linter, vulture, pytest, uv lock).

### A2. Run Unit Tests

```bash
uv run pytest                    # all unit tests, parallel
uv run pytest -x                 # stop on first failure
uv run pytest -v --tb=long       # verbose with full traceback
uv run pytest packages/triton_redis_connector/tests/test_engine/test_registry.py  # single file
uv run pytest -k test_metric_aggregation   # single test by name
```

Expected: all tests pass. Integration-marked tests are skipped by default.

### A3. Lint & Type Checks (Individually)

```bash
uv run ruff format --check .     # format check (no changes)
uv run ruff check .              # lint
uv run mypy .                    # type check (strict)
uv run pyright                   # type check (pyright strict)
uv run lint-imports              # architecture layer contracts
uv run vulture                   # dead code detection
```

---

## Section B: Model Conversion (Requires YOLO .pt File)

### B1. Convert YOLO to ONNX

```bash
mkdir -p /tmp/e2e_test_repo

uv run triton-model-tools convert \
  -w yolo11s.pt \
  -r /tmp/e2e_test_repo \
  --name yolo11s_test \
  --imgsz 640 \
  -f onnx
```

Expected: `/tmp/e2e_test_repo/yolo11s_test/1/model.onnx` and `config.pbtxt` created.

### B2. Convert with DALI Ensemble

```bash
uv run triton-model-tools convert \
  -w yolo11s.pt \
  -r /tmp/e2e_test_repo \
  --name yolo11s_dali_test \
  --imgsz 960x608 \
  --no-rect \
  -f onnx \
  --dali-ensemble
```

Expected: creates model + dali_preprocess + dali_ensemble directories.

### B3. Convert with NMS Postprocessing

```bash
uv run triton-model-tools convert \
  -w yolo11s.pt \
  -r /tmp/e2e_test_repo \
  --name yolo11s_nms_test \
  --imgsz 640 \
  -f onnx \
  --nms
```

Expected: creates model + NMS postprocessing model.

### B4. Convert to TensorRT (Requires GPU)

```bash
uv run triton-model-tools convert \
  -w yolo11s.pt \
  -r /tmp/e2e_test_repo \
  --name yolo11s_trt_test \
  --imgsz 640 \
  -f tensorrt \
  --half
```

Expected: TensorRT engine (`.plan` file). Requires GPU and TensorRT.

### B5. Convert with End-to-End NMS

```bash
uv run triton-model-tools convert \
  -w yolo11s.pt \
  -r /tmp/e2e_test_repo \
  --name yolo11s_e2e_test \
  --imgsz 640 \
  -f onnx \
  --end2end
```

Expected: ONNX model with NMS baked in.

### B6. Convert with Dynamic Batching

```bash
uv run triton-model-tools convert \
  -w yolo11s.pt \
  -r /tmp/e2e_test_repo \
  --name yolo11s_batch_test \
  --imgsz 640 \
  -f onnx \
  --dynamic-batching \
  --max-batch-size 8
```

Expected: config.pbtxt includes dynamic batching configuration.

### B7. Convert with Verify (Runs Predict After Export)

```bash
uv run triton-model-tools convert \
  -w yolo11s.pt \
  -r /tmp/e2e_test_repo \
  --name yolo11s_verify_test \
  --imgsz 640 \
  -f onnx \
  --image-dir test_fixtures/images \
  --output-dir /tmp/e2e_results/convert_verify \
  --conf-threshold 0.25 \
  --iou-threshold 0.45
```

Expected: converts model AND runs local ultralytics prediction for verification.

### B8. Cleanup Converted Repo

```bash
rm -rf /tmp/e2e_test_repo
```

---

## Section C: Model Repository Validation (No Triton Needed)

### C1. Validate Model Repository Structure

```bash
uv run triton-model-tools validate -r model_repository
```

Expected: validates all models (`yolo11s`, `yolo11s_960x608`, `yolo26l`, `yolo11s_960x608_dali_preprocess`, `yolo11s_960x608_dali_ensemble`, `rescale_postprocess`).

What to look for:
- Directory structure correct (each model has `config.pbtxt` and version folder)
- config.pbtxt is parseable and complete
- Model files exist in version directories

### C2. Validate Converted Repository (After Section B)

```bash
uv run triton-model-tools validate -r /tmp/e2e_test_repo
```

---

## Section D: Load and Verify Models in Triton (Requires Triton)

Triton runs in `--model-control-mode=explicit` — no models are loaded at startup.
You load/unload each model via the HTTP API, one at a time, to control VRAM usage.

### D1. Load Models One by One

```bash
# Load yolo11s first (smallest, good for smoke test)
curl -X POST http://localhost:8000/v2/repository/models/yolo11s/load
curl -s http://localhost:8000/v2/models/yolo11s/ready

# Load yolo11s_960x608
curl -X POST http://localhost:8000/v2/repository/models/yolo11s_960x608/load
curl -s http://localhost:8000/v2/models/yolo11s_960x608/ready

# Load yolo26l (large model — check VRAM before loading)
curl -X POST http://localhost:8000/v2/repository/models/yolo26l/load
curl -s http://localhost:8000/v2/models/yolo26l/ready

# For DALI ensemble: load preprocess first, then ensemble
curl -X POST http://localhost:8000/v2/repository/models/yolo11s_960x608_dali_preprocess/load
curl -s http://localhost:8000/v2/models/yolo11s_960x608_dali_preprocess/ready
curl -X POST http://localhost:8000/v2/repository/models/yolo11s_960x608_dali_ensemble/load
curl -s http://localhost:8000/v2/models/yolo11s_960x608_dali_ensemble/ready

# Load rescale postprocess
curl -X POST http://localhost:8000/v2/repository/models/rescale_postprocess/load
curl -s http://localhost:8000/v2/models/rescale_postprocess/ready
```

To **unload** a model and free VRAM:
```bash
curl -X POST http://localhost:8000/v2/repository/models/yolo26l/unload
```

To **load all models at once** (if VRAM allows):
```bash
for m in yolo11s yolo11s_960x608 yolo26l yolo11s_960x608_dali_preprocess yolo11s_960x608_dali_ensemble rescale_postprocess; do
  curl -X POST http://localhost:8000/v2/repository/models/$m/load
done
```

### D2. List Loaded Models

```bash
curl -s http://localhost:8000/v2/models | python3 -m json.tool
```

Expected: only models you explicitly loaded appear.

### D3. Check Specific Model Info

```bash
curl -s http://localhost:8000/v2/models/yolo11s | python3 -m json.tool
curl -s http://localhost:8000/v2/models/yolo11s_960x608 | python3 -m json.tool
curl -s http://localhost:8000/v2/models/yolo26l | python3 -m json.tool
curl -s http://localhost:8000/v2/models/yolo11s_960x608_dali_ensemble | python3 -m json.tool
```

### D4. Server Health & Metrics

```bash
curl -f http://localhost:8000/v2/health/live
curl -f http://localhost:8000/v2/health/ready
curl -s http://localhost:8000/v2
curl -s http://localhost:8002/metrics | head -50
```

---

## Section E: Local Inference Baseline (No Triton, No Redis)

### E1. Ultralytics YOLO Prediction (Ground Truth)

```bash
mkdir -p /tmp/e2e_results/predict

uv run triton-model-tools predict \
  --image-dir test_fixtures/images \
  --output-dir /tmp/e2e_results/predict \
  -m yolo11s.pt \
  --target-size 640 \
  --conf-threshold 0.25 \
  --iou-threshold 0.45
```

Expected: txt files in `/tmp/e2e_results/predict/` with detection results per image.

### E2. Local ONNX Inference

```bash
mkdir -p /tmp/e2e_results/local_infer

uv run triton-model-tools local-infer \
  --model-path test_fixtures/models/yolo11s.onnx \
  --image-dir test_fixtures/images \
  --imgsz 640 \
  --confidence 0.25 \
  --iou 0.45 \
  --output-dir /tmp/e2e_results/local_infer
```

Expected: txt files with detections.

---

## Section F: Local Drift Check (No Triton, No Redis)

Compare ultralytics predict (E1) vs local ONNX inference (E2).
Catches export drift: did the .pt → .onnx conversion change detection behavior?

### F1. Compare Predict vs Local ONNX

```bash
uv run triton-model-tools compare-folders \
  --baseline-dir /tmp/e2e_results/predict \
  --candidate-dir /tmp/e2e_results/local_infer \
  -m yolo11s \
  --iou-threshold 0.5 \
  -o /tmp/e2e_results/predict_vs_local.json
```

Expected: precision/recall close to 1.0, class mismatch rate close to 0.0.

### F2. Visual Drift Check (Requires FiftyOne)

```bash
uv run triton-model-tools compare-visual \
  --local-dir /tmp/e2e_results/predict \
  --triton-dir /tmp/e2e_results/local_infer \
  --image-dir test_fixtures/images \
  --dataset-name predict-vs-local-onnx \
  --launch
```

Expected: opens FiftyOne web UI with side-by-side comparison.

---

## Section G: Triton Inference (Requires Triton)

### G1. Triton Inference — yolo11s (640x640)

```bash
mkdir -p /tmp/e2e_results/triton_infer

uv run triton-model-tools infer \
  --image-dir test_fixtures/images \
  -m yolo11s \
  --triton-url localhost:8001 \
  --target-size 640 \
  --conf-threshold 0.25 \
  --iou-threshold 0.45 \
  --output-dir /tmp/e2e_results/triton_infer
```

### G2. Triton Inference — yolo11s_960x608 (Rectangular)

```bash
mkdir -p /tmp/e2e_results/triton_960x608

uv run triton-model-tools infer \
  --image-dir test_fixtures/images \
  -m yolo11s_960x608 \
  --triton-url localhost:8001 \
  --target-size 960x608 \
  --conf-threshold 0.25 \
  --iou-threshold 0.45 \
  --output-dir /tmp/e2e_results/triton_960x608
```

### G3. Triton Inference — DALI Ensemble

```bash
mkdir -p /tmp/e2e_results/triton_dali

uv run triton-model-tools infer \
  --image-dir test_fixtures/images \
  -m yolo11s_960x608_dali_ensemble \
  --triton-url localhost:8001 \
  --target-size 960x608 \
  --conf-threshold 0.25 \
  --iou-threshold 0.45 \
  --output-dir /tmp/e2e_results/triton_dali
```

### G4. Triton Inference — yolo26l

```bash
mkdir -p /tmp/e2e_results/triton_yolo26l

uv run triton-model-tools infer \
  --image-dir test_fixtures/images \
  -m yolo26l \
  --triton-url localhost:8001 \
  --target-size 640 \
  --conf-threshold 0.25 \
  --iou-threshold 0.45 \
  --output-dir /tmp/e2e_results/triton_yolo26l
```

---

## Section H: Triton Drift Check (Requires Triton)

Compare local ONNX inference (E2) vs Triton inference (G1).
Catches serving drift: does Triton produce different results than local ONNX Runtime?

### H1. Compare Local ONNX vs Triton (yolo11s)

```bash
uv run triton-model-tools compare-folders \
  --baseline-dir /tmp/e2e_results/local_infer \
  --candidate-dir /tmp/e2e_results/triton_infer \
  -m yolo11s \
  --iou-threshold 0.5 \
  -o /tmp/e2e_results/local_vs_triton.json
```

Expected: near-identical results (same ONNX model, same preprocessing).

### H2. Compare CV2 vs DALI Preprocessing

```bash
uv run triton-model-tools compare-folders \
  --baseline-dir /tmp/e2e_results/triton_960x608 \
  --candidate-dir /tmp/e2e_results/triton_dali \
  -m yolo11s_960x608 \
  --iou-threshold 0.5 \
  -o /tmp/e2e_results/cv2_vs_dali.json
```

Expected: close results. Small differences expected (DALI uses GPU decode, different resize kernel).

### H3. Visual Drift Check — Local vs Triton (Requires FiftyOne)

```bash
uv run triton-model-tools compare-visual \
  --local-dir /tmp/e2e_results/local_infer \
  --triton-dir /tmp/e2e_results/triton_infer \
  --image-dir test_fixtures/images \
  --dataset-name local-vs-triton \
  --launch
```

---

## Section I: Benchmarks (Requires Triton)

### I1. Single Model Benchmark

```bash
uv run triton-model-tools benchmark \
  --image-dir test_fixtures/images \
  -m yolo11s \
  --triton-url localhost:8001 \
  --target-size 640 \
  --num-images 50 \
  --warmup-runs 3 \
  -o /tmp/e2e_results/benchmark_yolo11s.json
```

Expected: latency stats (mean, p50, p95, p99), throughput (FPS), memory usage.

### I2. Benchmark All Models

```bash
uv run triton-model-tools benchmark-all \
  --image-dir test_fixtures/images \
  --repository-root model_repository \
  --triton-url localhost:8001 \
  --num-images 30 \
  --warmup-runs 2 \
  -o /tmp/e2e_results/benchmark_all.json
```

Expected: benchmarks for each non-utility model (yolo11s, yolo11s_960x608, yolo26l). Utility models (dali_preprocess, ensemble, rescale) are skipped.

### I3. Perf Analyzer (Requires Triton SDK Container)

```bash
docker compose -f .devcontainer/docker-compose.debug.yml exec triton-sdk \
  perf_analyzer -m yolo11s -u triton:8001 --protocol grpc --concurrency-range 1:4
```

Or via CLI wrapper:
```bash
uv run triton-model-tools perf-analyze \
  -m yolo11s \
  --url localhost:8001 \
  --protocol grpc \
  --concurrency 1,4,8 \
  -o /tmp/e2e_results/perf_analyze.json
```

Expected: throughput/latency at each concurrency level.

### I4. Perf Analyze All Models

```bash
uv run triton-model-tools perf-analyze \
  --all \
  --repository-root model_repository \
  --url localhost:8001 \
  -o /tmp/e2e_results/perf_analyze_all.json
```

---

## Section J: Redis Stream Operations (Requires Redis)

### J1. Populate Stream with Test Images

```bash
uv run triton-model-tools populate-stream \
  --stream test:images \
  --image-dir test_fixtures/images \
  --redis-url redis://localhost:6379 \
  --event-type image \
  --camera-count 3 \
  --verify
```

Expected: images pushed to `test:images` stream with fields (camera_no, image_path, frame_counter, receipt_ts, metal_type, event_type). Verify mode reads them back.

### J2. Verify Stream Contents

```bash
docker compose -f .devcontainer/docker-compose.debug.yml exec redis redis-cli XLEN test:images
docker compose -f .devcontainer/docker-compose.debug.yml exec redis redis-cli XRANGE test:images - + COUNT 2
```

Expected: 8 messages (one per test image), each with expected fields.

### J3. Populate with Limited Count

```bash
uv run triton-model-tools populate-stream \
  --stream test:images_limited \
  --image-dir test_fixtures/images \
  --redis-url redis://localhost:6379 \
  --count 3

docker compose -f .devcontainer/docker-compose.debug.yml exec redis redis-cli XLEN test:images_limited
```

Expected: exactly 3 messages.

### J4. Cleanup Streams

```bash
docker compose -f .devcontainer/docker-compose.debug.yml exec redis redis-cli DEL test:images test:images_limited
```

---

## Section K: E2E Pipeline (Requires Redis + Triton + Connector App)

Steps K1-K4 require a running connector application. Start it in a separate terminal first.

### K0. Start Connector Application

Option A — Example pipeline:
```bash
cd examples/dual_model_pipeline
REDIS_HOST=localhost TRITON_GRPC_URL=localhost:8001 \
  uv run python run_pipeline.py
```

Option B — Hydra overrides:
```bash
REDIS_HOST=localhost TRITON_GRPC_URL=localhost:8001 \
  uv run python -m triton_redis_connector \
    sources[0].stream_name=camera_images \
    sources[0].consumer_group=e2e-test \
    sources[0].event_model=ImageEvent \
    sinks[0].stream_name=detection_results \
    pipeline.model_name=yolo11s
```

### K1. E2E Test (Push Images -> Process -> Collect Results)

```bash
uv run triton-model-tools e2e-test \
  --image-dir test_fixtures/images \
  -m yolo11s \
  --redis-url redis://localhost:6379 \
  --input-stream camera_images \
  --output-stream detection_results \
  --consumer-group e2e-test \
  --target-size 640 \
  --timeout 30
```

Expected: pushes images to `camera_images`, waits for results on `detection_results`, reports detection counts per image.

### K2. Pipeline Predict (Save Detections to Files)

```bash
mkdir -p /tmp/e2e_results/pipeline_predict

uv run triton-model-tools pipeline-predict \
  --image-dir test_fixtures/images \
  --output-dir /tmp/e2e_results/pipeline_predict \
  -m yolo11s \
  --redis-url redis://localhost:6379 \
  --input-stream camera_images \
  --output-stream detection_results \
  --consumer-group pipeline-predict \
  --timeout 30
```

Expected: txt files with detection results, matching Triton direct inference.

### K3. Pipeline Benchmark (Measure E2E Throughput)

```bash
uv run triton-model-tools pipeline-benchmark \
  --image-dir test_fixtures/images \
  -m yolo11s \
  --redis-url redis://localhost:6379 \
  --input-stream camera_images \
  --output-stream detection_results \
  --num-iterations 3 \
  --warmup-runs 1 \
  --timeout 60 \
  -o /tmp/e2e_results/pipeline_benchmark.json
```

Expected: E2E latency stats (Redis round-trip + preprocessing + inference + postprocessing).

### K4. Pipeline Benchmark Full Matrix

```bash
uv run triton-model-tools pipeline-benchmark \
  --image-dir test_fixtures/images \
  -m yolo11s \
  --redis-url redis://localhost:6379 \
  --input-stream camera_images \
  --output-stream detection_results \
  --full-matrix \
  --backend onnx \
  --preprocess-mode cv2 dali \
  --num-iterations 2 \
  -o /tmp/e2e_results/pipeline_matrix.json
```

Expected: comparison matrix across backend x preprocess mode combinations.

---

## Section L: Application Configuration Testing

### L1. Verify Default Config Loads

```bash
REDIS_HOST=localhost TRITON_GRPC_URL=localhost:8001 \
  uv run python -c "
from omegaconf import OmegaConf
from hydra import compose, initialize_config_dir
import os

initialize_config_dir(
    config_dir=os.path.abspath('packages/triton_redis_connector/triton_redis_connector/conf'),
    version_base=None
)
cfg = compose(config_name='config')
print(OmegaConf.to_yaml(cfg))
"
```

### L2. Test Example Pipeline Config (Dual Model)

```bash
cat examples/dual_model_pipeline/config.yaml
```

Verify:
- `app_name` is set
- `redis.host` uses `${oc.env:REDIS_HOST,redis}`
- `sources` has at least one entry with stream_name, consumer_group, event_model
- `inference.grpc_url` uses `${oc.env:TRITON_GRPC_URL,triton:8001}`

### L3. Test Debug Config

```bash
REDIS_HOST=localhost TRITON_GRPC_URL=localhost:8001 \
  uv run python -m triton_redis_connector \
    +debug=single_thread \
    sources[0].stream_name=test:debug \
    sources[0].consumer_group=debug \
    sources[0].event_model=ImageEvent \
    sinks[0].stream_name=test:debug_out \
    pipeline.model_name=yolo11s
```

Expected: runs in single-thread mode (1 worker, 1 concurrent, stop_on_error=true).

---

## Section M: Integration Test Suites

### M1. All Integration Tests

```bash
uv run pytest -m integration -v --tb=long
```

### M2. Specific Integration Tests

```bash
uv run pytest -m integration -k e2e -v
uv run pytest -m integration -k perf_analyzer -v
uv run pytest -m integration -k dynamic_batching -v
```

### M3. GPU / Slow / Benchmark Tests

```bash
uv run pytest -m gpu -v
uv run pytest -m slow -v
uv run pytest -m benchmark -v
```

### M4. All Tests Including Integration

```bash
uv run pytest -m "" -v --tb=long
```

---

## Section N: Monitoring

### N1. RedisInsight Web UI

Open browser: `http://localhost:5540`

- Add database: host=`redis`, port=`6379` (inside Docker network)
- Browse Streams: check stream lengths, message contents
- Monitor: watch real-time commands

### N2. Redis CLI Monitoring

```bash
COMPOSE="-f .devcontainer/docker-compose.debug.yml"

docker compose $COMPOSE exec redis redis-cli MONITOR

docker compose $COMPOSE exec redis redis-cli XINFO STREAM camera_images
docker compose $COMPOSE exec redis redis-cli XINFO GROUPS camera_images
docker compose $COMPOSE exec redis redis-cli XINFO CONSUMERS camera_images e2e-test

docker compose $COMPOSE exec redis redis-cli XLEN camera_images
docker compose $COMPOSE exec redis redis-cli XLEN detection_results
docker compose $COMPOSE exec redis redis-cli XLEN metrics

docker compose $COMPOSE exec redis redis-cli XREVRANGE detection_results + - COUNT 5
docker compose $COMPOSE exec redis redis-cli XREVRANGE metrics + - COUNT 10
```

### N3. Triton Monitoring

```bash
curl -s http://localhost:8000/v2/models | python3 -m json.tool
curl -s http://localhost:8002/metrics | head -50
curl -f http://localhost:8000/v2/health/live
curl -f http://localhost:8000/v2/health/ready
```

---

## Section O: Cleanup & Teardown

### O1. Stop Connector App

Press `Ctrl+C` in the terminal running the connector.

### O2. Cleanup Redis Streams

```bash
COMPOSE="-f .devcontainer/docker-compose.debug.yml"
docker compose $COMPOSE exec redis redis-cli DEL camera_images detection_results metrics
docker compose $COMPOSE exec redis redis-cli DEL test:images test:debug test:debug_out
docker compose $COMPOSE exec redis redis-cli DEL steel:inspection_input task:input
```

### O3. Stop Infrastructure

```bash
docker compose -f .devcontainer/docker-compose.debug.yml down
```

### O4. Clean Test Artifacts

```bash
rm -rf /tmp/e2e_results /tmp/e2e_test_repo
```

### O5. Full Cleanup (Including Volumes)

```bash
docker compose -f .devcontainer/docker-compose.debug.yml down -v
```

---

## Quick Bug Hunting Checklist

| Area | What to Check | Section |
|------|--------------|---------|
| Type safety | mypy strict + pyright strict pass | A3 |
| Dead code | No unused functions/imports | A3 |
| Architecture | No layer violations | A3 |
| Unit tests | All pass | A2 |
| Export correctness | Convert produces valid repo | B + C |
| Model loading | All models load in Triton | D |
| Export drift | .pt vs .onnx results match | F1 |
| Serving drift | Local ONNX vs Triton match | H1 |
| Preprocessing drift | OpenCV vs DALI consistent | H2 |
| Inference performance | Latency within SLA | I |
| Stream round-trip | Messages survive push/read | J |
| Pipeline E2E | Full flow works | K1 |
| Error handling | Corrupt images handled | Push bad data to stream, check logs |
| Memory leaks | VRAM stays bounded | Watch `nvidia-smi` during I1 |
| Recovery | Pending messages processed on restart | Stop connector, restart, check pending |
| Metrics | Counters increment, gauges update | N2 (check `metrics` stream) |

---

## Environment Variables Reference

| Variable | Default | Used By |
|----------|---------|---------|
| `TRITON_VERSION` | `26.02` | Docker Compose (image tags) |
| `REDIS_HOST` | `localhost` | Connector config |
| `REDIS_PORT` | `6379` | Connector config |
| `TRITON_GRPC_URL` | `localhost:8001` | Connector config |
| `APP_NAME` | `triton-connector` | Connector config |
| `INPUT_STREAM` | varies by example | Example configs |

## Ports Reference (Debug Mode)

| Service | Port | Protocol |
|---------|------|----------|
| Redis | 6379 | TCP |
| Triton HTTP | 8000 | HTTP |
| Triton gRPC | 8001 | gRPC |
| Triton Metrics | 8002 | HTTP (Prometheus) |
| RedisInsight | 5540 | HTTP |
| Triton SDK | 8010 | HTTP |

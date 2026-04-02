# Benchmark Results

**Date**: 2026-04-02
**GPU**: NVIDIA GeForce RTX 5090 (32GB VRAM, sm_120 Blackwell)
**Triton**: 26.02 (ONNX Runtime backend, GPU mode)
**Driver**: 595.79, CUDA 13.2
**Tool**: perf_analyzer (gRPC, 10s measurement windows)
**Method**: Load one model → warmup → benchmark at c=1,4,8 → unload → next

## perf_analyzer Results (Pure Triton Inference)

### ONNX Models (FP32)

| Model | Input Size | Concurrency | Throughput (i/s) | Avg (ms) | P50 (ms) | P95 (ms) | P99 (ms) |
|-------|-----------|-------------|-----------------|----------|----------|----------|----------|
| yolo11s | 640x640 | c=1 | **76.6 - 89.4** | 9.5 - 9.9 | 9.3 - 9.5 | 11.0 - 11.4 | 12.0 - 20.1 |
| yolo11s_2inst | 640x640 | c=1 | **78.8 - 90.2** | 9.6 - 9.9 | 9.3 - 9.5 | 11.3 - 11.8 | 12.6 - 13.5 |
| yolo11s_960x608 | 960x608 | c=1 | **65.7 - 67.8** | 11.6 - 12.0 | 11.4 - 11.8 | 13.2 - 13.9 | 14.6 - 16.1 |
| yolo12s | 640x640 | c=1 | **66.6 - 79.4** | 10.9 - 11.6 | 10.6 - 11.4 | 12.5 - 13.6 | 13.8 - 15.2 |
| yolo12s | 640x640 | c=4 | **103.0 - 106.1** | 35.2 - 35.4 | 35.2 | 40.2 - 43.0 | 46.6 - 52.3 |
| yolo12s | 640x640 | c=8 | **105.8** | 73.1 | 72.3 | 81.1 | 88.9 |
| yolo26s | 640x640 | c=1 | **56.5 - 59.9** | 15.7 - 16.1 | 12.7 - 13.8 | 28.3 - 29.5 | 32.0 - 35.2 |
| yolo26s | 640x640 | c=4 | **69.6 - 74.5** | 51.3 - 55.3 | 47.3 - 52.5 | 74.1 - 79.0 | 85.2 - 87.8 |
| yolo26s | 640x640 | c=8 | **69.7 - 71.0** | 109.0 - 112.6 | 105.2 - 110.5 | 145.8 - 153.2 | 167.1 - 174.4 |
| yolo26l | 640x640 | c=1 | **41.6 - 43.0** | 21.8 | 19.1 - 19.6 | 34.3 - 34.7 | 38.4 - 44.2 |
| yolo26l | 640x640 | c=4 | **52.6 - 54.3** | 71.5 - 74.5 | 69.1 - 71.7 | 95.9 - 98.2 | 109.6 - 114.0 |
| yolo26l | 640x640 | c=8 | **52.1** | 150.9 | 150.1 | 186.1 | 206.6 |

### ONNX FP16 Model

| Model | Input Size | Concurrency | Throughput (i/s) | Avg (ms) | P50 (ms) | P95 (ms) | P99 (ms) |
|-------|-----------|-------------|-----------------|----------|----------|----------|----------|
| yolo11s_fp16 | 640x640 | c=1 | **47.7 - 52.1** | 17.7 - 18.1 | 16.6 - 17.1 | 30.3 - 30.5 | 37.2 - 44.7 |
| yolo11s_fp16 | 640x640 | c=4 | **58.6 - 63.4** | 60.2 - 64.3 | 58.0 - 64.5 | 87.8 - 90.7 | 98.2 - 104.3 |
| yolo11s_fp16 | 640x640 | c=8 | **62.6 - 66.0** | 117.4 - 124.0 | 114.7 - 122.2 | 163.4 - 168.5 | 179.5 - 183.9 |

### ONNX + TensorRT Execution Accelerator (FP16)

| Model | Status | Notes |
|-------|--------|-------|
| yolo11s_ort_trt | UNSTABLE | TRT engine compilation on Blackwell (sm_120) doesn't stabilize within measurement window |
| yolo11s_ort_trt_2inst | UNSTABLE | Same issue with 2 instances |
| yolo12s_ort_trt | UNSTABLE | Engine build exceeds measurement timeout |
| yolo26s_ort_trt | UNSTABLE | Engine build exceeds measurement timeout |
| yolo26l_ort_trt | UNSTABLE | Engine build exceeds measurement timeout |

> **Note**: ORT-TRT UNSTABLE results are due to TensorRT engine compilation happening at first inference on RTX 5090 (Blackwell architecture). The TRT engine compilation takes several minutes. With pre-built and cached engines (subsequent runs), these models would stabilize. Engine caching is configured (`trt_engine_cache_enable=1`) but requires a successful initial compilation to populate the cache.

### DALI Ensemble

| Model | Status | Notes |
|-------|--------|-------|
| yolo11s_dali_ensemble | FAILED | DALI NVML error on RTX 5090 — Blackwell not supported in DALI 1.53 (Triton 26.02) |
| yolo11s_960x608_dali_ensemble | FAILED | Same DALI NVML issue |

> **Note**: DALI backend on Blackwell requires DALI 1.55+ (expected in Triton 26.04+).

## Pipeline Benchmark (Preprocess → Triton gRPC → Postprocess)

From `triton-model-tools benchmark-all` — includes client-side letterbox preprocessing and YOLO postprocessing.

| Model | Input Size | Avg (ms) | Throughput (fps) | P50 (ms) | P95 (ms) | P99 (ms) | Peak VRAM |
|-------|-----------|----------|-----------------|----------|----------|----------|-----------|
| yolo11s | 640x640 | 17.7 | **56.5** | 17.3 | 20.7 | 21.2 | 5268 MB |
| yolo11s_960x608 | 960x608 | 19.9 | **50.4** | 18.3 | 26.5 | 27.4 | 5528 MB |
| yolo26l | 640x640 | 24.3 | **41.2** | 24.2 | 29.5 | 31.3 | 5976 MB |

## Key Observations

1. **yolo11s ONNX FP32 is fastest** at c=1: ~87 fps pure inference, ~57 fps E2E pipeline
2. **FP16 ONNX is slower than FP32** on RTX 5090 — ONNX Runtime may not have optimized FP16 kernels for Blackwell
3. **2-instance models show marginal improvement** at c=1 (78.8 vs 76.6 fps) — benefit appears at higher concurrency
4. **Concurrency scaling**: yolo12s scales well (66→106 fps from c=1→c=4), yolo26s/l saturate at c=4
5. **Single-instance c=4/c=8 UNSTABLE** for most models — without dynamic batching, single instance can't handle concurrent requests stably
6. **ORT-TRT compilation on Blackwell** is extremely slow — needs pre-built engines for benchmarking
7. **DALI on Blackwell** not supported in Triton 26.02 — wait for 26.04+

## Concurrency Behavior (no dynamic batching)

Without `dynamic_batching` enabled, concurrency >1 causes request queuing:
- c=1: One request at a time — lowest latency, limited throughput
- c=4: 4 concurrent requests queue on single instance — higher throughput but 4x latency
- c=8: Diminishing returns — throughput plateaus, latency doubles

For production: enable `dynamic_batching` with appropriate `max_batch_size` to efficiently handle concurrent requests.

## Environment Notes

- Models run in CPU-less mode (all on GPU)
- Docker runtime: nvidia (WSL2 with `NVIDIA_DRIVER_CAPABILITIES=compute,utility,video`)
- Triton model control: explicit (load/unload via API)
- No dynamic batching configured (all max_batch_size=0)
- perf_analyzer runs from Triton SDK container via gRPC to Triton server container

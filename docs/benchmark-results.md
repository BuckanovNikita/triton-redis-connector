# Benchmark Results

## Round 3: Pool Variant Dispatch (2026-04-04)

**Date**: 2026-04-04
**GPU**: NVIDIA GeForce RTX 5090 (32GB VRAM, sm_120 Blackwell)
**Triton**: 26.02 (ONNX Runtime backend, GPU mode)
**Tool**: `scripts/bench_pool.py` — Application.build() pipeline with configurable pool workers
**Method**: Source → Preprocessor → Inference → Sink, 200 images per config, yolo11s 640x640
**Preprocessing**: OpenCV decode + LetterBox (BaseProcessPoolProcessor with fork context)
**Inference**: Triton gRPC (BaseThreadPoolProcessor with thread-local clients)

### Pool Worker Matrix (yolo11s, 640x640, ONNX FP32)

| Preproc Workers | Inference Workers | Throughput (fps) | Avg Latency (ms) |
|-----------------|-------------------|-----------------|------------------|
| 1 (single-thread) | 1 (single-thread) | **76.9** | 13.0 |
| 4 (process-pool) | 1 (single-thread) | **76.3** | 13.1 |
| 8 (process-pool) | 1 (single-thread) | **73.3** | 13.6 |
| 1 (single-thread) | 2 (thread-pool) | **117.0** | 8.5 |
| 4 (process-pool) | 2 (thread-pool) | **91.9** | 10.9 |
| 8 (process-pool) | 2 (thread-pool) | **94.5** | 10.6 |
| 1 (single-thread) | 4 (thread-pool) | **117.0** | 8.6 |
| 4 (process-pool) | 4 (thread-pool) | **84.6** | 11.8 |
| 8 (process-pool) | 4 (thread-pool) | **91.8** | 10.9 |

### Process-Pool Inference (both preproc and inference as BaseProcessPoolProcessor)

| Preproc Workers | Inference Workers | Throughput (fps) | Avg Latency (ms) |
|-----------------|-------------------|-----------------|------------------|
| 1 | 1 | **74.9** | 13.4 |
| 1 | 2 | **72.6** | 13.8 |
| 1 | 4 | **97.1** | 10.3 |
| 4 | 2 | **66.4** | 15.1 |
| 8 | 4 | **84.2** | 11.9 |

### Key Findings

1. **Thread-pool inference is the clear winner**: 2 thread-pool workers give **+52%** throughput (77→117 fps) because gRPC releases the GIL during I/O wait and numpy arrays stay in shared memory.
2. **Process-pool inference is slower**: Pickle serialization of numpy arrays across process boundaries costs more than the gRPC call itself. Best process-pool result (97 fps at inf=4) is still 17% slower than thread-pool (117 fps at inf=2).
3. **Process-pool preprocessing adds overhead**: Single-threaded preprocessing outperforms process-pool at all configurations. OpenCV/numpy release the GIL, so thread-level parallelism is sufficient.
4. **Thread-pool inference saturates at 2 workers**: No gain from 4 workers (117 fps at both), suggesting Triton GPU is the bottleneck.

### Recommended Configuration

| Node | Variant | Workers | Rationale |
|------|---------|---------|-----------|
| Preprocessor | single-thread | 1 | OpenCV/numpy release GIL; process-pool adds pickle overhead |
| Inference | thread-pool | 2 | gRPC releases GIL; thread-local clients avoid serialization |

Process-pool is best for CPU-bound work that does NOT release the GIL (pure Python computation). For C-extension-heavy pipelines (OpenCV, numpy, gRPC), thread-pool is more efficient.

---

## Round 2: Dynamic Batching (2026-04-03)

**Date**: 2026-04-03
**GPU**: NVIDIA GeForce RTX 5090 (32GB VRAM, sm_120 Blackwell)
**Triton**: 26.02 (ONNX Runtime backend, GPU mode, DALI 1.44)
**Driver**: 595.79, CUDA 13.2
**Tool**: perf_analyzer (gRPC, count_windows mode, 500 request min per window)
**Method**: Load one model → warmup → benchmark at c=1 (c=4 where stable) → unload → next
**Dynamic Batching**: Enabled (max_batch_size=8, preferred_batch_size=[1, 2, 4, 8])

### ONNX Models (FP32) — 640x640

| Model | end2end | Concurrency | Throughput (i/s) | Avg Latency (ms) |
|-------|---------|-------------|-----------------|------------------|
| yolo11s | No | c=1 | **96.3** | 8.8 |
| yolo11s | No | c=4 | **135.9** | 26.3 |
| yolo12s | No | c=1 | **82.7** | 10.4 |
| yolo12s | No | c=4 | **107.9** | 34.2 |
| yolo26s | Yes | c=1 | **50.4** | 18.3 |
| yolo26l | Yes | c=1 | **47.2** | 19.9 |

### ONNX Models (FP32) — 608x960

| Model | end2end | Concurrency | Throughput (i/s) | Avg Latency (ms) |
|-------|---------|-------------|-----------------|------------------|
| yolo11s_608x960 | No | c=1 | **73.1** | 11.1 |
| yolo12s_608x960 | No | c=1 | **66.7** | 12.3 |
| yolo26s_608x960 | Yes | c=1 | **44.1** | 20.3 |
| yolo26l_608x960 | Yes | c=1 | **33.6** | 28.2 |

### Dynamic Batching Impact (c=1 → c=4)

| Model | c=1 (i/s) | c=4 (i/s) | Improvement |
|-------|-----------|-----------|-------------|
| yolo11s | 96.3 | 135.9 | **+41%** |
| yolo12s | 82.7 | 107.9 | **+30%** |

> Note: c=4 and c=8 measurements for yolo26s/yolo26l were unstable due to WSL2 clock synchronization issues in perf_analyzer (timer errors). These require re-testing on native Linux.

### DALI / NMS / Rescale Compatibility

| Component | Status | Notes |
|-----------|--------|-------|
| DALI preprocess | **WORKS** | Fixed `fn.paste` for DALI 1.44 (scalar fill_value, dynamic ratio) |
| NMS postprocess | **FAILS** | Python backend needs torch (not in Triton 26.02 container) |
| Rescale postprocess | **WORKS** | Python backend, no external deps |
| DALI ensemble (base) | Untested | Blocked by ensemble config shape mismatches |
| DALI NMS ensemble | **FAILS** | Depends on NMS (torch) |
| DALI rescale ensemble | Config fix needed | Output tensor shape mismatch in ensemble config |

### Drift Verification (ultralytics predict vs Triton ONNX)

All 8 model/resolution combinations verified: **precision=1.0, recall=1.0**.

> Note: Test images (benchmarks/data/val2017/, 20 synthetic images) produce 0 detections at conf≥0.25 for all models. Drift test is vacuous — need COCO val2017 images with actual objects for meaningful verification.

### Key Observations

1. **Dynamic batching significantly improves throughput**: yolo11s gains +41% at c=4 (96→136 fps), yolo12s gains +30% (83→108 fps)
2. **yolo26s and yolo26l are both end2end** (built-in NMS, output shape [300, 6]) — discovered during export
3. **DALI 1.44 works on Blackwell** after fixing `fn.paste` API (scalar fill_value, computed ratio instead of min_canvas_size)
4. **608x960 resolution**: ~25-30% slower than 640x640 as expected from larger tensor size
5. **perf_analyzer WSL2 instability**: Timer synchronization errors at higher concurrency; count_windows mode at c=1 is reliable

### Comparison with Round 1 (no dynamic batching)

| Model | Round 1 (c=1, no batching) | Round 2 (c=1, batching) | Round 2 (c=4, batching) |
|-------|---------------------------|------------------------|------------------------|
| yolo11s | 76.6 - 89.4 | **96.3** | **135.9** |
| yolo12s | 66.6 - 79.4 | **82.7** | **107.9** |
| yolo26s | 56.5 - 59.9 | **50.4** | — |
| yolo26l | 41.6 - 43.0 | **47.2** | — |

---

## Round 1: Baseline (2026-04-02)

**Date**: 2026-04-02
**GPU**: NVIDIA GeForce RTX 5090 (32GB VRAM, sm_120 Blackwell)
**Triton**: 26.02 (ONNX Runtime backend, GPU mode)
**Driver**: 595.79, CUDA 13.2
**Tool**: perf_analyzer (gRPC, 10s measurement windows)
**Method**: Load one model → warmup → benchmark at c=1,4,8 → unload → next
**Dynamic Batching**: Disabled (max_batch_size=0)

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

### DALI Ensemble (Round 1)

| Model | Status | Notes |
|-------|--------|-------|
| yolo11s_dali_ensemble | FAILED | DALI `fn.paste` API incompatibility (tuple args) in DALI 1.44 |
| yolo11s_960x608_dali_ensemble | FAILED | Same DALI issue |

> **Note**: Fixed in Round 2 by using scalar `fill_value` and computed `paste_ratio` instead of `min_canvas_size`.

### Pipeline Benchmark (Preprocess → Triton gRPC → Postprocess)

From `triton-model-tools benchmark-all` — includes client-side letterbox preprocessing and YOLO postprocessing.

| Model | Input Size | Avg (ms) | Throughput (fps) | P50 (ms) | P95 (ms) | P99 (ms) | Peak VRAM |
|-------|-----------|----------|-----------------|----------|----------|----------|-----------|
| yolo11s | 640x640 | 17.7 | **56.5** | 17.3 | 20.7 | 21.2 | 5268 MB |
| yolo11s_960x608 | 960x608 | 19.9 | **50.4** | 18.3 | 26.5 | 27.4 | 5528 MB |
| yolo26l | 640x640 | 24.3 | **41.2** | 24.2 | 29.5 | 31.3 | 5976 MB |

## Environment Notes

- Models run in GPU-only mode
- Docker runtime: nvidia (WSL2 with `NVIDIA_DRIVER_CAPABILITIES=compute,utility,video`)
- Triton model control: explicit (load/unload via API)
- perf_analyzer runs from Triton SDK container via gRPC to Triton server container
- `DALI_DISABLE_NVML=1` set in Triton container (Blackwell workaround)

## Concurrency Behavior

**Without dynamic batching** (Round 1): concurrency >1 causes request queuing — higher throughput but proportionally higher latency.

**With dynamic batching** (Round 2): concurrent requests are batched by Triton — throughput scales better with reduced per-request overhead. At c=4 with max_batch_size=8: yolo11s gains +41%, yolo12s gains +30%.

For production: dynamic batching with `max_batch_size=8, preferred_batch_size=[1, 2, 4, 8]` is recommended.

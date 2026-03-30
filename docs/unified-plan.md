# Unified Implementation Plan: triton-redis-connector + triton-model-tools

## Context

The project has two existing plan documents (`docs/phases.md` — 5-phase app plan, `docs/tools-phases.md` — 5-phase tools plan) that are independent. This unified plan merges them so the `triton_model_tools` package serves as end-to-end test infrastructure for the `triton_redis_connector` app. It also addresses:

- **Architecture rework**: App uses processor/source `__call__(event)` pattern instead of `ComponentProtocol.process(envelope)`
- **Performance comparison**: Stages comparing raw `ultralytics.predict()` vs Triton vs connector throughput
- **DALI preprocessing correctness**: DALI and client-side preprocessing must replicate ultralytics letterbox exactly (not simple resize)
- **Triton SDK container**: Added for `perf_analyzer` and version-matched client testing
- **Version upgrades**: All NGC tools updated to latest available versions

Reference: `docs/design.md` (to be updated alongside).

---

## Phase 0 — Infrastructure & Version Upgrades

**Goal**: Upgrade all container/dependency versions, add Triton SDK container, verify devcontainer stack boots cleanly.

### 0.1 Docker Compose changes

File: `.devcontainer/docker-compose.yml`

- Upgrade Triton: `nvcr.io/nvidia/tritonserver:25.02-py3` → `26.02-py3`
- Add SDK container:

```yaml
triton-sdk:
  image: nvcr.io/nvidia/tritonserver:26.02-py3-sdk
  volumes:
    - ..:/workspace:cached
    - ../model_repository:/models
  depends_on:
    triton:
      condition: service_healthy
  command: sleep infinity
```

The SDK container provides `perf_analyzer`, `model_analyzer`, and pre-installed `tritonclient` matching the server version. It shares the workspace and model_repository volumes, reaches Triton server via hostname `triton`.

### 0.2 Dependency version pins

| Package | Current | Target | File |
|---|---|---|---|
| `tritonclient[all]` | `>=2.52.0` | `>=2.66.0` | both `pyproject.toml` |
| `ultralytics` | `>=8.3.0` | `>=8.4.0` | tools `pyproject.toml` |
| `onnx` | `>=1.17.0` | `>=1.20.0` | tools `pyproject.toml` |
| `onnxruntime` | `>=1.20.0` | `>=1.24.0` | tools `pyproject.toml` |

DALI note: `nvidia-dali-cuda120==2.0.0` is bundled in the `26.02-py3` container. It is NOT a Python dependency of our packages — it runs inside Triton's DALI backend only.

### 0.3 Update CLAUDE.md

Reflect new container versions and SDK container in the infrastructure section.

### Verification

```bash
docker compose -f .devcontainer/docker-compose.yml up --build -d
# Verify Triton health, SDK perf_analyzer, identity model loads
uv sync --dev && uv run pytest && uv run pre-commit run --all-files
```

---

## Phase 1 — Foundation: Shared Models, Config, Protocols, `__call__` Architecture

**Goal**: Establish all type contracts, the processor/source/sink `__call__` architecture, the Envelope event wrapper, and Hydra config. Compilable skeleton with no runtime behavior but all API contracts locked in.

### Architecture: `__call__`-based Pipeline

Replaces `ComponentProtocol.process(envelope)` from design.md. Three user-facing primitives:

```python
@runtime_checkable
class Source(Protocol[TOut]):
    """Produces events. Runs in own thread, pushes events via emit()."""
    def __call__(self, emit: Callable[[TOut], None]) -> None: ...

@runtime_checkable
class Processor(Protocol[TIn, TOut]):
    """Transforms events. Return None to filter."""
    def __call__(self, event: TIn) -> TOut | None: ...

@runtime_checkable
class Sink(Protocol[TIn]):
    """Consumes events (terminal node)."""
    def __call__(self, event: TIn) -> None: ...
```

Key design decisions:
- **Push-based Sources** — Sources run in their own thread and push events via an `emit()` callback. This is natural for streaming data (Redis XREADGROUP blocks until messages arrive). The framework provides `emit` and manages the thread lifecycle. Built-in `RedisStreamSource` wraps XREADGROUP internally. Custom sources (camera SDK, filesystem watcher) use the same pattern — call `emit(event)` whenever data arrives. The Source's `__call__` runs continuously until shutdown.
- **Output type via return annotation** — `def __call__(self, event: ImageEvent) -> InferenceInput | None` declares input=ImageEvent, output=InferenceInput. `| None` stripped during wiring (means "filter")
- **Inference is a Processor** — `TritonInferenceProcessor(Processor[InferenceInput, InferenceOutput])` is framework-provided. Under the hood it routes through `mp.Queue` to worker pool, but in the pipeline graph it's a normal Processor
- **Envelope hidden** — `Envelope[T]` stays internal. Before calling `__call__`, runtime unwraps `envelope.payload`. After return, wraps result in new Envelope inheriting metadata. For opt-in metadata access: `current_context()` via `contextvars`
- **Exact type matching** — wiring by type equality, no subclass polymorphism

### Deliverables

#### 1.1 Package: `triton_shared_models`

Create `packages/triton_shared_models/`:
- `triton_types.py` — `TritonDataType`, `TritonInstanceKind`, `TritonTensorSpec`
- `triton_config.py` — `TritonModelConfig` (expanded for strict mode), `to_pbtxt()`

Update `triton_model_tools` to import from `triton_shared_models`.

#### 1.2 Config schema (`config/schema.py`)

Pydantic models: `RedisConnectionConfig`, `StreamInputConfig`, `StreamOutputConfig`, `InferenceConfig`, `PreprocessingConfig`, `MetricsConfig`, `AppConfig` — as in design.md section 5.2, plus `sources` and `sinks` config sections for the new architecture.

#### 1.3 Hydra integration (`config/hydra_app.py`)

Hydra init, default YAML files under `conf/`. Add `hydra-core` dep to connector.

#### 1.4 Envelope (`engine/event.py`)

`Envelope[T]` with `track_state()`, `transit_time_ms()`, `__lt__`, `derive(payload)`.

#### 1.5 Protocols and event types

| File | Contents |
|---|---|
| `protocols.py` | `Source`, `Processor`, `Sink`, `ObserverProtocol`, `AggregationRuleProtocol` |
| `engine/context.py` | `EventContext`, `current_context()` via contextvars |
| `engine/events.py` | `Event` base, `InferenceInput`, `InferenceOutput`, `TensorSpec` |
| `connectors/base.py` | `InputConnectorProtocol`, `OutputConnectorProtocol` (internal) |
| `preprocessing/protocols.py` | `ImageDecoderProtocol`, `ImagePreprocessorProtocol` |

#### 1.6 Error models, inference models, metric event

`ConnectorError`, `InferenceError`, `PreprocessingError`, `InferenceRequest`, `InferenceResult`, `MetricEvent`.

### Tests

- All pydantic models (config, errors, events, metric event)
- Envelope (state tracking, priority, ordering, derive)
- `triton_model_tools` tests pass after shared model extraction
- Hydra config loading tests
- Protocol `runtime_checkable` isinstance checks on dummy impls

### Verification

```bash
uv run pytest && uv run pre-commit run --all-files
```

---

## Phase 2 — Generalize Export + TensorRT + DALI Ensemble

**Goal**: Backend-agnostic export, TensorRT support, DALI ensemble generation with correct ultralytics preprocessing replication.

### CRITICAL: Ultralytics Preprocessing in DALI

Ultralytics does NOT just resize. Their pipeline (from `ultralytics.data.augment.LetterBox`):

1. **LetterBox**: Resize maintaining aspect ratio, pad shorter dimension with `(114, 114, 114)` gray to target size (640x640). Image centered.
2. **BGR → RGB** conversion
3. **HWC → CHW** transpose
4. **Divide by 255.0** (normalize to [0, 1]). No mean/std subtraction beyond this.
5. **Float32** cast
6. **Batch dimension** added

The DALI `model.py` must implement:
- `fn.decoders.image(device="mixed")` — decode JPEG on GPU
- `fn.resize()` with aspect-preserving logic
- `fn.pad()` with `fill_value=114` — letterbox
- `fn.color_space_conversion()` — BGR→RGB
- `fn.crop_mirror_normalize(mean=[0,0,0], std=[255.0,255.0,255.0], output_layout="CHW")`

A simple `fn.resize(resize_x=640, resize_y=640)` without letterbox will produce WRONG detection results.

### Deliverables

#### 2.1 New types in `models.py`

`ExportFormat` enum (ONNX/TensorRT), `EXPORT_FORMAT_TO_PLATFORM`, `EXPORT_FORMAT_TO_FILENAME`. Extended `TritonModelConfig` for strict mode: `default_model_filename`, `dynamic_batching`, `optimization`, `model_warmup`. `TritonTensorSpec.is_shape_tensor`.

#### 2.2 Generalize `repository.py`

`onnx_source` → `model_source` + `model_filename` parameter.

#### 2.3 TensorRT export in `exporter.py`

`_export_onnx()` / `_export_tensorrt()`. Naming: `yolo11s` (ONNX), `yolo11s_trt` (TRT).

#### 2.4 DALI module (`dali.py`)

`DaliPipelineConfig` (with `pad_value=114`, `normalize_div=255.0`), `generate_dali_pipeline_script()`, `build_dali_model_config()`, `build_ensemble_config()`, `build_ensemble_directory()`.

Ensemble structure per model: `yolo11s/` (inference) + `yolo11s_dali_preprocess/` (DALI backend) + `yolo11s_dali_ensemble/` (ensemble scheduler).

#### 2.5 CLI refactoring

Subcommand CLI: `triton-model-tools export --format onnx|tensorrt|all --dali-ensemble --models ...`

### Tests

- ExportFormat, platform/filename mappings
- Generalized build_model_directory (.onnx and .plan)
- DALI pipeline script: `ast.parse()` validity + verify letterbox ops present
- Ensemble config: verify `platform: "ensemble"`, step mappings
- All existing tests pass

### Verification

```bash
uv run pytest && uv run pre-commit run --all-files
```

---

## Phase 3 — Connectors + Engine Loop + Inference Client (parallel tracks)

**Goal**: Build both the app engine (Redis I/O, queues, processor dispatch) and the tools inference client. After this: (a) connector does Redis→processor→Redis passthrough, (b) tools can send images to Triton directly.

### 3A — App: Connectors, Queues, Engine (parallel track)

#### 3A.1 Consumer group management (`connectors/consumer_group.py`)

`ConsumerGroupManager`: `ensure_group_exists()`, `read_pending_newest_first()`, `read_new()`, `acknowledge()`.

#### 3A.2 Redis connectors (`connectors/redis_stream.py`)

`RedisStreamInputConnector` (thread, XREADGROUP, wraps in Envelope, puts on queue).
`RedisStreamOutputConnector` (thread, reads queue, XADD).

Built-in `RedisStreamSource[T]` (push-based, runs XREADGROUP in own thread, calls `emit(event)`) and `RedisStreamSink[T]` wrappers.

#### 3A.3 Queue bus (`engine/queue_bus.py`)

`live_queue`, `pending_queue`, `output_queue`, `error_queue` per stream.

#### 3A.4 Pipeline registry (`engine/registry.py`)

`PipelineRegistry`: inspects `__call__` type hints via `typing.get_type_hints()`. `build_pipeline()` wires Source→Processor→Sink by type matching. Validates no orphans, no cycles, no missing consumers.

#### 3A.5 Application skeleton (`engine/app.py`)

`Application.from_config()` and `run()`. Main loop: drain queues → unwrap Envelope → dispatch `__call__` → wrap result → route to output/next processor. Graceful shutdown.

#### 3A.6 Newest-first recovery

Read pending on startup, sort newest-first, dual-queue reader.

### 3B — Tools: Inference Client + Folder Inference (parallel track)

#### 3B.1 Detection models (`detection.py`)

`Detection`, `ImageDetectionResult`, `FolderInferenceResult`.

#### 3B.2 Inference client (`inference.py`)

`TritonInferenceClient` with `infer_raw()` / `infer_dali()`.

**CRITICAL: `YoloPreprocessor` must replicate ultralytics letterbox exactly.**

```python
class YoloPreprocessor:
    """Replicates ultralytics.data.augment.LetterBox preprocessing.

    Steps:
    1. Resize maintaining aspect ratio (scale to fit, not stretch)
    2. Pad with (114, 114, 114) to target_size (letterbox)
    3. BGR → RGB
    4. HWC → CHW
    5. /255.0
    6. float32
    7. batch dim

    Returns (tensor, scale_factor, pad_offset) for postprocessing coord rescaling.
    """
```

`YoloPostprocessor`: handles both end2end (yolo26s, already NMS'd) and standard (yolo11s/12s, need NMS) formats. Must account for letterbox offset/scale when converting back to original image coordinates.

#### 3B.3 Folder inference (`folder_inference.py`)

`run_folder_inference()` + `infer` CLI subcommand.

### Cross-Validation Test (CRITICAL)

Load real image, preprocess with `YoloPreprocessor`, also preprocess with `ultralytics.data.augment.LetterBox` + ultralytics pipeline, assert tensors numerically identical (`np.allclose`). This validates preprocessing replication.

### Tests

3A: Mock Redis, consumer groups, connectors, registry `__call__` wiring, Application passthrough
3B: YoloPreprocessor shape/dtype/letterbox, mock Triton, folder inference

### Verification

```bash
uv run pytest && uv run pre-commit run --all-files
```

---

## Phase 4 — Inference Worker Pool + Tools as E2E Test Driver

**Goal**: Full connector pipeline (Redis → preprocess → Triton → Redis). Tools package drives E2E tests through Redis.

### 4A — App: Inference Worker Pool

#### 4A.1 Worker pool (`inference/worker.py`)

`InferenceWorkerPool`: N processes, each with own `tritonclient.grpc.InferenceServerClient`. Communication via `mp.Queue`. The runtime layer intercepts `InferenceInput` events (from TritonInferenceProcessor in the pipeline) and routes through the pool.

#### 4A.2 Retry policy (`inference/retry.py`)

3 attempts, exponential backoff (0.1s, 0.4s, 1.6s).

#### 4A.3 Model version tracker (`inference/model_registry.py`)

Per-worker, polls `get_model_repository_index()`.

#### 4A.4 Image preprocessing (`preprocessing/`)

`TurboJPEGDecoder`, `CV2AvifDecoder`, `DecoderDispatcher`, `ParallelDecoderPool`.

**CRITICAL**: Connector's preprocessing MUST share the same letterbox algorithm as `YoloPreprocessor` from tools. Three implementations must produce identical results:
1. `YoloPreprocessor` in tools (client-side, for `infer_raw`)
2. DALI pipeline `model.py` (server-side, for ensemble)
3. Connector's `ImagePreprocessorProtocol.to_raw()` (in-pipeline)

#### 4A.5 Wire inference into Application

Update main loop: when pipeline hits `TritonInferenceProcessor`, route `InferenceInput` through `mp.request_queue`. Re-inject `InferenceOutput` back into pipeline from `mp.result_queue`.

### 4B — Tools as E2E Test Driver

#### 4B.1 E2E test harness (`e2e_test.py`)

```python
class ConnectorE2ETestHarness:
    """Drives E2E tests: push images → Redis → connector → Triton → Redis → collect results."""
    def push_test_images(self, image_paths: list[Path]) -> list[str]: ...
    def collect_results(self, expected_count: int, timeout_s: float = 30.0) -> list[ImageDetectionResult]: ...
    def run_through_connector(self, image_dir: Path) -> FolderInferenceResult: ...
```

#### 4B.2 CLI: `e2e-test` subcommand

```bash
triton-model-tools e2e-test --image-dir ./test_images --model yolo11s \
  --redis-url redis://redis:6379 --input-stream camera_images --output-stream detection_results
```

### Tests

- Worker pool integration with identity model
- Retry policy
- E2E: tools pushes image to Redis → connector processes → tools reads result
- Connector preprocessing matches tools preprocessing (same image, same results)

### Verification

```bash
uv run pytest && uv run pre-commit run --all-files
```

Integration: push image via tools → connector → verify result on output stream.

---

## Phase 5 — Benchmarking + Performance Comparison Stage 1

**Goal**: Benchmarking framework + first comparison: raw `ultralytics.predict()` vs direct Triton.

### Deliverables

#### 5.1 COCO downloader (`dataset.py`)

`download_coco_val()` — direct HTTP, no fiftyone dependency.

#### 5.2 Benchmark engine (`benchmark.py`)

`BenchmarkConfig`, `BenchmarkResult`, `BenchmarkSuite`. Functions: `run_benchmark()` (Triton via tools), `run_ultralytics_benchmark()` (raw `model.predict()` baseline), `run_benchmark_matrix()`.

#### 5.3 perf_analyzer integration (`perf_analyzer.py`)

Shells out to `perf_analyzer` in the SDK container via `docker exec`. Parses CSV output.

#### 5.4 CLI: `download-coco`, `benchmark` subcommands

#### 5.5 Performance Comparison Checkpoint 1

| Method | Description |
|---|---|
| `ultralytics.predict()` | Baseline (built-in preprocessing) |
| Triton ONNX raw | Tools client-side letterbox → Triton ONNX |
| Triton TRT raw | Tools client-side letterbox → Triton TensorRT |
| perf_analyzer ONNX | Triton's own benchmark (synthetic, peak throughput) |
| perf_analyzer TRT | Triton's own benchmark (synthetic, peak throughput) |

### Tests

- BenchmarkConfig/Result models
- run_benchmark with mocked client
- download_coco with mocked HTTP
- run_ultralytics_benchmark with mocked ultralytics
- BenchmarkSuite JSON/CSV serialization

### Verification

```bash
uv run pytest && uv run pre-commit run --all-files
```

---

## Phase 6 — Prediction Comparison + Metrics + Performance Stage 2

**Goal**: Detection accuracy comparison, app metrics/errors, second performance checkpoint adding DALI and connector.

### 6A — App: Metrics, Errors, Observability

- `MetricBuffer`, `MetricEmitter` (periodic XADD to metrics stream)
- Transit time tracking via `Envelope.track_state()` at each pipeline stage
- `TritonMetricsCollector` (polls Triton stats via gRPC)
- `ErrorCollector`, `ErrorHandler` (drain error queue, log, aggregate)
- Wire into Application main loop

### 6B — Tools: Prediction Comparison

#### 6B.1 Comparison module (`comparison.py`)

`compute_iou()`, `match_detections()`, `ComparisonMetrics`, `ComparisonSummary`, `compare_inference()`, `compare_backends()`.

**DALI validation gate**: Comparison between DALI ensemble and raw-mode results validates DALI preprocessing correctness. Low IoU = DALI letterbox is wrong.

#### 6B.2 FiftyOne integration (`fiftyone_export.py`)

Optional dependency (`[fiftyone]` extra). Guarded import with clear error message.

#### 6B.3 CLI: `compare`, `fiftyone-app` subcommands

#### 6B.4 Performance Comparison Checkpoint 2

| Method | Description |
|---|---|
| `ultralytics.predict()` | Baseline |
| Triton ONNX raw | Client-side letterbox |
| Triton TRT raw | Client-side letterbox |
| Triton ONNX DALI ensemble | Server-side DALI letterbox |
| Triton TRT DALI ensemble | Server-side DALI letterbox |
| Connector ONNX | Redis → connector → Triton (via E2E harness) |
| Connector TRT | Redis → connector → Triton (via E2E harness) |
| perf_analyzer (all backends) | Synthetic peak throughput |

Key accuracy metrics per method: precision/recall vs ultralytics baseline, avg IoU, class mismatch rate, false positives, missed detections.

### Tests

6A: MetricBuffer concurrent, MetricEmitter mock Redis, TritonMetricsCollector mock, ErrorCollector/Handler
6B: compute_iou edge cases, match_detections, compare_inference with mocks, FiftyOne export with mocks

### Verification

```bash
uv run pytest && uv run pre-commit run --all-files
```

---

## Phase 7 — Aggregation, Polish, Full Performance Matrix

**Goal**: Business metrics aggregation, user-facing API cleanup, complete performance matrix.

### 7A — App: Aggregation + Polish

#### 7A.1 Package: `triton_metrics_aggregator`

`AggregationRule`, `TimeWindowAggregator[TEvent]`. Built-in rules: `count_by`, `sum_by`, `avg_by`.

#### 7A.2 Observer as Sink

Register `TimeWindowAggregator` as a `Sink` — framework calls `__call__` on each event.

#### 7A.3 API cleanup

Public exports in `__init__.py`. Entry point with `@hydra.main`.

#### 7A.4 Example application

`examples/camera_inference/` — event models, processors using `__call__`, config YAML.

### 7B — Full Performance Matrix

3 models × 6 pipelines + 3 ultralytics baselines = 21 benchmark rows:

| Model | Backend | Preprocess | Pipeline |
|---|---|---|---|
| yolo26s/11s/12s | ONNX | raw letterbox | Direct Triton |
| yolo26s/11s/12s | ONNX | DALI ensemble | Direct Triton |
| yolo26s/11s/12s | TRT | raw letterbox | Direct Triton |
| yolo26s/11s/12s | TRT | DALI ensemble | Direct Triton |
| yolo26s/11s/12s | ONNX | connector preprocess | Redis→Connector→Triton |
| yolo26s/11s/12s | TRT | connector preprocess | Redis→Connector→Triton |
| yolo26s/11s/12s | ultralytics | built-in | `model.predict()` |

CLI: `triton-model-tools full-benchmark --image-dir ./data/val2017 --models yolo11s yolo26s yolo12s --output full_benchmark.json`

### Tests

- Aggregation unit tests
- Full pipeline integration
- Example app smoke test
- Benchmark CLI smoke test (mocked)

### Verification

```bash
uv run pytest && uv run pre-commit run --all-files
```

---

## Phase Dependency Graph

```
Phase 0 (Infrastructure)
    │
    v
Phase 1 (Foundation: shared models, config, protocols, __call__)
    │
    ├──► Phase 2 (Export: TRT + DALI)
    │        │
    │        v
    │    Phase 3B (Tools: inference client)───► Phase 5 (Benchmark Stage 1)
    │        │                                      │
    │        │                                      v
    │        │                                  Phase 6B (Comparison Stage 2)
    │        │                                      │
    │        │                                      v
    │        │                                  Phase 7B (Full matrix)
    │
    └──► Phase 3A (App: connectors, queues, engine)
             │
             v
         Phase 4A (App: inference workers)
             │
             ├──► Phase 4B (Tools E2E harness — needs both 3B + 4A)
             │        │
             │        v (feeds Phase 6B and 7B connector benchmarks)
             │
             └──► Phase 6A (App: metrics, errors)
                      │
                      v
                  Phase 7A (App: aggregation, polish)
```

**Parallelism**: 3A ‖ 3B, 5 starts after 3B (no connector needed), 6A ‖ 6B, 7A ‖ 7B. Phase 4B is the integration point requiring both tracks.

---

## Critical Files

| File | Phase | Purpose |
|---|---|---|
| `.devcontainer/docker-compose.yml` | 0 | Triton 26.02, SDK container |
| `packages/triton_shared_models/` | 1 | Shared Triton types |
| `packages/triton_redis_connector/triton_redis_connector/protocols.py` | 1 | `Source`, `Processor`, `Sink` protocols |
| `packages/triton_redis_connector/triton_redis_connector/engine/context.py` | 1 | `EventContext`, `current_context()` |
| `packages/triton_redis_connector/triton_redis_connector/engine/events.py` | 1 | `Event`, `InferenceInput`, `InferenceOutput` |
| `packages/triton_redis_connector/triton_redis_connector/engine/registry.py` | 3A | `PipelineRegistry` (`__call__` type inspection) |
| `packages/triton_redis_connector/triton_redis_connector/engine/app.py` | 3A | Main loop with envelope unwrap/wrap |
| `packages/triton_model_tools/triton_model_tools/dali.py` | 2 | DALI pipeline with correct letterbox |
| `packages/triton_model_tools/triton_model_tools/inference.py` | 3B | `YoloPreprocessor` (letterbox), `TritonInferenceClient` |
| `packages/triton_model_tools/triton_model_tools/e2e_test.py` | 4B | `ConnectorE2ETestHarness` |
| `packages/triton_model_tools/triton_model_tools/benchmark.py` | 5 | Benchmarking engine |
| `packages/triton_model_tools/triton_model_tools/comparison.py` | 6B | Prediction comparison |

---

## User Code Example (After All Phases)

```python
# events.py
class CameraImageEvent(BaseModel):
    camera_id: str
    image_data: bytes

class DetectionResult(BaseModel):
    camera_id: str
    detections: list[BoundingBox]

# processors.py
class CameraPreprocessor:
    def __call__(self, event: CameraImageEvent) -> InferenceInput:
        decoded = self._decoder.decode(event.image_data)
        tensor = self._preprocessor.letterbox(decoded)
        return InferenceInput(model_name="yolo11s", inputs={"images": tensor.tobytes()}, ...)

class DetectionPostprocessor:
    def __call__(self, event: InferenceOutput) -> DetectionResult:
        raw = np.frombuffer(event.outputs["output0"], dtype=np.float32)
        detections = self._parse_yolo_output(raw)
        return DetectionResult(camera_id="...", detections=detections)

# Optional custom source (push-based, runs in own thread):
class CameraSource:
    def __call__(self, emit: Callable[[CameraImageEvent], None]) -> None:
        """Runs in dedicated thread. Pushes events via emit()."""
        while not self._stop.is_set():
            frame = self._camera.read()
            emit(CameraImageEvent(camera_id="cam1", image_data=frame))

# config.yaml
sources:
  - type: redis_stream                    # built-in push-based source
    stream_name: camera_images
    event_type: myapp.events.CameraImageEvent
sinks:
  - type: redis_stream
    stream_name: detection_results
    event_type: myapp.events.DetectionResult
processors:
  - myapp.processors.CameraPreprocessor
  - triton_redis_connector.TritonInferenceProcessor
  - myapp.processors.DetectionPostprocessor

# main.py
app = Application.from_config()
app.run()
```

The user writes: 2 event models, 2 processors, 1 YAML config. Everything else (Redis I/O, queues, threading, inference workers, metrics, error handling, recovery) is handled by the framework. Sources are push-based — they run in their own thread and call `emit(event)` when data arrives.

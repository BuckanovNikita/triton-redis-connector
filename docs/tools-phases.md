# Model Tools — Implementation Phases

This document defines the implementation plan for `triton_model_tools` — the package
responsible for model export, Triton inference, benchmarking, and validation.

Builds on the existing codebase: ONNX export pipeline, TritonModelConfig, YOLO_MODELS
registry, Triton model repository builder.

Reference: [design.md](design.md) for the overall system architecture.

---

## Current State

```
triton_model_tools/
  models.py       — TritonDataType, TritonModelConfig, YoloModelVariant, YOLO_MODELS
  exporter.py     — ONNX-only export (ultralytics → ONNX → Triton repo)
  repository.py   — build_model_directory() (hardcoded to model.onnx)
  __main__.py     — single CLI command: triton-model-export
```

Limitations:
- Only ONNX export (no TensorRT)
- No DALI ensemble support
- No inference tools
- No benchmarking
- No prediction comparison
- `repository.py` hardcodes `model.onnx` filename
- `TritonModelConfig.platform` defaults to `"onnxruntime_onnx"`

---

## Target State After All Phases

```
triton_model_tools/
  models.py            — + ExportFormat enum, platform/filename mappings
  exporter.py          — ONNX + TensorRT export
  repository.py        — generalized model dir builder (any format)
  dali.py              — DALI pipeline generation + ensemble config
  detection.py         — Detection, ImageDetectionResult, FolderInferenceResult
  inference.py         — TritonInferenceClient, YoloPreprocessor, YoloPostprocessor
  folder_inference.py  — folder → Triton → results orchestrator
  benchmark.py         — benchmarking engine (Triton vs ultralytics)
  dataset.py           — COCO val2017 downloader
  comparison.py        — Triton vs ultralytics prediction comparison
  __main__.py          — subcommand CLI (export, infer, benchmark, compare, download-coco)
```

### CLI Commands

```bash
triton-model-tools export    [--format onnx|tensorrt|all] [--dali-ensemble] [--models ...]
triton-model-tools infer     [--image-dir DIR] [--model M] [--mode raw|dali] [--output FILE]
triton-model-tools benchmark [--image-dir DIR] [--models ...] [--formats ...] [--modes ...]
triton-model-tools compare   [--image-dir DIR] [--triton-model M] [--ultralytics-model M.pt]
triton-model-tools download-coco [--target-dir DIR] [--max-images N]
```

### Full Testing Matrix

Every model tested through all combinations:

| Model | Backend | API Mode | Operations |
|---|---|---|---|
| yolo26s | ONNX | raw | infer, benchmark, compare |
| yolo26s | ONNX | dali | infer, benchmark, compare |
| yolo26s | TensorRT | raw | infer, benchmark, compare |
| yolo26s | TensorRT | dali | infer, benchmark, compare |
| yolo11s | ONNX | raw | infer, benchmark, compare |
| yolo11s | ONNX | dali | infer, benchmark, compare |
| yolo11s | TensorRT | raw | infer, benchmark, compare |
| yolo11s | TensorRT | dali | infer, benchmark, compare |
| yolo12s | ONNX | raw | infer, benchmark, compare |
| yolo12s | ONNX | dali | infer, benchmark, compare |
| yolo12s | TensorRT | raw | infer, benchmark, compare |
| yolo12s | TensorRT | dali | infer, benchmark, compare |

---

## Phase 1 — Generalize Export + TensorRT Support

**Goal**: Refactor the ONNX-only pipeline into a backend-agnostic export system,
then add TensorRT export. After this phase, both ONNX and TensorRT models can be
exported to the Triton model repository.

### 1.1 New types in `models.py`

```python
class ExportFormat(StrEnum):
    ONNX = "onnx"
    TENSORRT = "engine"

EXPORT_FORMAT_TO_PLATFORM: dict[ExportFormat, str] = {
    ExportFormat.ONNX: "onnxruntime_onnx",
    ExportFormat.TENSORRT: "tensorrt_plan",
}

EXPORT_FORMAT_TO_FILENAME: dict[ExportFormat, str] = {
    ExportFormat.ONNX: "model.onnx",
    ExportFormat.TENSORRT: "model.plan",
}
```

Extend `YoloModelVariant` with `export_formats: tuple[ExportFormat, ...] = (ExportFormat.ONNX,)`.

### 1.2 Generalize `repository.py`

Rename parameter: `onnx_source: Path` → `model_source: Path`, add `model_filename: str`.
Copies `model_source` to `version_dir / model_filename` instead of hardcoded `model.onnx`.

### 1.3 TensorRT export in `exporter.py`

Strategy: ultralytics `model.export(format="engine")` internally goes through ONNX,
so we can inspect the intermediate ONNX for tensor specs, then use them for the
TensorRT config.pbtxt with `platform: "tensorrt_plan"`.

```python
def _export_onnx(variant: YoloModelVariant, repository_root: Path) -> Path: ...
def _export_tensorrt(variant: YoloModelVariant, repository_root: Path) -> Path: ...
```

Add `export_format: ExportFormat` parameter to `export_model()` and `export_all_models()`.

**Model naming**: `yolo11s` (ONNX), `yolo11s_trt` (TensorRT). Both coexist in repository.

### 1.4 Update `__main__.py`

Restructure to subcommand CLI (`argparse` subparsers). First subcommand: `export`.

```bash
triton-model-tools export --format tensorrt --models yolo11s
triton-model-tools export --format all       # exports both ONNX + TensorRT
```

Keep `triton-model-export` as backward-compatible entry point.

### Tests

- `ExportFormat` enum, platform/filename mappings
- Generalized `build_model_directory` with `.onnx` and `.plan` filenames
- Export format routing (mock ultralytics YOLO to avoid real downloads in CI)
- All existing tests pass after refactor

### Dependencies

None new. TensorRT export handled by ultralytics (TensorRT available in Triton container).

---

## Phase 2 — DALI Ensemble Model Generation

**Goal**: Generate Triton ensemble configs that chain DALI preprocessing with YOLO
inference. A single CLI command produces three model directories per ensemble.

### Ensemble Structure

For model `yolo11s` with ONNX backend:

```
model_repository/
  yolo11s/                       # Inference model (from Phase 1)
    config.pbtxt                 # platform: "onnxruntime_onnx"
    1/model.onnx

  yolo11s_dali_preprocess/       # DALI preprocessing
    config.pbtxt                 # backend: "dali"
    1/model.py                   # Generated DALI pipeline script

  yolo11s_dali_ensemble/         # Ensemble scheduler
    config.pbtxt                 # platform: "ensemble"
    1/                           # Empty version dir (required by Triton)
```

### 2.1 New module: `dali.py`

```python
class DaliPipelineConfig(BaseModel):
    target_width: int = 640
    target_height: int = 640
    mean: tuple[float, float, float] = (0.0, 0.0, 0.0)
    std: tuple[float, float, float] = (1.0, 1.0, 1.0)
    output_dtype: str = "FLOAT"
```

Key functions:

- `generate_dali_pipeline_script(config: DaliPipelineConfig) -> str`
  Returns Python source for `model.py` (the DALI pipeline class that runs inside
  Triton's DALI backend). Uses `nvidia.dali.fn.decoders.image`, `fn.resize`,
  `fn.crop_mirror_normalize`.

- `build_dali_model_config(name: str, max_batch_size: int) -> str`
  Returns config.pbtxt for DALI backend:
  ```
  backend: "dali"
  input [ { name: "DALI_INPUT_0" data_type: TYPE_UINT8 dims: [ -1 ] } ]
  output [ { name: "DALI_OUTPUT_0" data_type: TYPE_FP32 dims: [ 3, 640, 640 ] } ]
  ```

- `build_ensemble_config(ensemble_name, dali_model_name, inference_model_name, ...) -> str`
  Returns ensemble config.pbtxt with `platform: "ensemble"` and
  `ensemble_scheduling` section mapping DALI output → inference input.

- `build_ensemble_directory(repository_root, variant, export_format, dali_config) -> Path`
  Creates all three directories (inference, dali preprocess, ensemble).

### 2.2 New types in `models.py`

```python
class TritonEnsembleStep(BaseModel):
    model_name: str
    model_version: int = -1
    input_map: dict[str, str]
    output_map: dict[str, str]
```

### 2.3 Repository builder additions

- `build_dali_model_directory(repository_root, model_name, config_pbtxt, pipeline_script) -> Path`
- `build_ensemble_model_directory(repository_root, ensemble_name, config_pbtxt) -> Path`

### 2.4 CLI: `--dali-ensemble` flag

```bash
triton-model-tools export --format onnx --dali-ensemble --models yolo11s
# Creates: yolo11s/, yolo11s_dali_preprocess/, yolo11s_dali_ensemble/
```

### Tests

- `generate_dali_pipeline_script` → validate with `ast.parse()` (valid Python)
- `build_dali_model_config` → verify pbtxt contains `backend: "dali"`, correct I/O specs
- `build_ensemble_config` → verify `platform: "ensemble"`, step mappings
- `build_ensemble_directory` → verify all three directories created with correct structure
- Use `tmp_path` fixtures for all directory tests

### Dependencies

None. DALI runs inside Triton container. `model.py` is a generated file placed in
the repository — we only generate the string, not execute it.

---

## Phase 3 — Inference Client + Folder Inference Tool

**Goal**: Build a Triton inference client and a CLI tool to run inference on image
folders. Supports both raw (client-side decode) and DALI ensemble (raw bytes) APIs.

### 3.1 New module: `detection.py`

Pydantic models for detection results:

```python
class Detection(BaseModel):
    bbox: tuple[float, float, float, float]    # x1, y1, x2, y2
    confidence: float
    class_id: int
    class_name: str = ""

class ImageDetectionResult(BaseModel):
    image_path: str
    detections: list[Detection]
    inference_time_ms: float

class FolderInferenceResult(BaseModel):
    results: list[ImageDetectionResult]
    total_images: int
    total_time_ms: float
    avg_time_ms: float

    def to_json(self, path: Path) -> None: ...
```

### 3.2 New module: `inference.py`

Triton client abstraction + pre/postprocessing:

```python
class TritonInferenceClient:
    def __init__(self, url: str = "localhost:8001") -> None:
        self._client = tritonclient.grpc.InferenceServerClient(url=url)

    def infer_raw(
        self, model_name: str, images: list[npt.NDArray[np.float32]],
    ) -> list[npt.NDArray[np.float32]]:
        """Raw API: client-side preprocessed float32 tensors."""

    def infer_dali(
        self, model_name: str, jpeg_bytes_list: list[bytes],
    ) -> list[npt.NDArray[np.float32]]:
        """DALI ensemble API: raw JPEG bytes as 1D uint8 arrays."""

    def is_model_ready(self, model_name: str) -> bool: ...
    def get_model_metadata(self, model_name: str) -> dict[str, Any]: ...


class YoloPreprocessor:
    def preprocess(self, image_path: Path, imgsz: int = 640) -> npt.NDArray[np.float32]:
        """Load image, resize to imgsz x imgsz, normalize [0,1], CHW layout."""


class YoloPostprocessor:
    def postprocess(
        self,
        raw_output: npt.NDArray[np.float32],
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ) -> list[Detection]:
        """Parse YOLO output → list of Detection.

        Handles both end2end (yolo26s, already NMS'd) and
        standard (yolo11s/12s, need NMS) output formats.
        """
```

### 3.3 New module: `folder_inference.py`

```python
def run_folder_inference(
    image_dir: Path,
    model_name: str,
    triton_url: str = "localhost:8001",
    mode: Literal["raw", "dali"] = "raw",
    batch_size: int = 1,
    conf_threshold: float = 0.25,
    output_path: Path | None = None,
) -> FolderInferenceResult:
    """Scan image_dir, send to Triton, collect detection results.

    mode="raw":  preprocess client-side, use infer_raw()
    mode="dali": read raw bytes, use infer_dali() on ensemble model
    """
```

### 3.4 CLI subcommand: `infer`

```bash
triton-model-tools infer \
  --image-dir ./test_images \
  --model yolo11s \
  --mode raw \
  --batch-size 4 \
  --output results.json
```

### Tests

- `YoloPreprocessor`: synthetic image → verify shape `(1, 3, 640, 640)`, dtype float32,
  range [0, 1]
- `YoloPostprocessor`: synthetic YOLO output arrays (both end2end and standard format)
- `Detection` / `ImageDetectionResult` serialization round-trip
- `TritonInferenceClient`: mock `tritonclient.grpc.InferenceServerClient`
- `run_folder_inference`: mock client, temp dir with small images, verify result structure
- Integration tests (mark `@pytest.mark.integration`): real Triton with identity/YOLO model

### Dependencies

None new. Uses existing: `tritonclient[all]`, `numpy`, `Pillow`.

---

## Phase 4 — Benchmarking + COCO Dataset

**Goal**: Benchmarking framework measuring throughput and latency across all
model/backend/preprocessing combinations. Includes COCO validation set download.

### 4.1 New module: `dataset.py`

```python
def download_coco_val(target_dir: Path, max_images: int | None = None) -> Path:
    """Download COCO 2017 val images (~1GB, 5000 images).

    Direct HTTP download from images.cocodataset.org/zips/val2017.zip.
    Skips if already downloaded. Returns path to image directory.
    """

def download_coco_annotations(target_dir: Path) -> Path:
    """Download instances_val2017.json for mAP calculation."""
```

Rationale for direct download over fiftyone: fiftyone pulls in MongoDB and ~1GB of
transitive dependencies. A single `urllib.request.urlretrieve` call is simpler and
matches the project's minimalist dependency approach.

### 4.2 New module: `benchmark.py`

```python
class BenchmarkConfig(BaseModel):
    model_name: str
    backend: ExportFormat
    mode: Literal["raw", "dali"]
    batch_size: int = 1
    num_warmup: int = 10
    num_iterations: int = 100
    triton_url: str = "localhost:8001"


class BenchmarkResult(BaseModel):
    config: BenchmarkConfig
    throughput_img_per_sec: float
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    total_images: int
    total_time_sec: float


class BenchmarkSuite(BaseModel):
    results: list[BenchmarkResult]
    timestamp: str

    def to_json(self, path: Path) -> None: ...
    def to_csv(self, path: Path) -> None: ...
```

Key functions:

- `run_benchmark(config, image_dir) -> BenchmarkResult`
  Warmup → timed iterations → compute percentile stats.

- `run_ultralytics_benchmark(model_name, image_dir, ...) -> BenchmarkResult`
  Direct `model.predict()` baseline for comparison against Triton.

- `run_benchmark_matrix(models, formats, modes, image_dir, batch_sizes) -> BenchmarkSuite`
  Runs all combinations, collects results into a single report.

### 4.3 CLI subcommands

```bash
triton-model-tools download-coco --target-dir ./data --max-images 1000

triton-model-tools benchmark \
  --image-dir ./data/val2017 \
  --models yolo11s yolo26s \
  --formats onnx tensorrt \
  --modes raw dali \
  --batch-sizes 1 4 8 \
  --output benchmark.json
```

### Tests

- `BenchmarkConfig` / `BenchmarkResult` pydantic models
- `run_benchmark` with mocked `TritonInferenceClient`: verify warmup + measurement,
  correct statistics from known latency arrays
- `download_coco_val` with mocked `urllib.request.urlretrieve`: verify directory
  structure, skip-if-exists logic
- `run_ultralytics_benchmark` with mocked ultralytics YOLO
- `BenchmarkSuite` JSON/CSV serialization

### Dependencies

None new. Uses `urllib.request` (stdlib) for COCO download.

---

## Phase 5 — Prediction Comparison Tool

**Goal**: Numerically compare Triton inference outputs against ultralytics native
`model.predict()`. Measures detection accuracy drift across backends and
preprocessing modes.

### 5.1 New module: `comparison.py`

```python
def compute_iou(
    box_a: tuple[float, float, float, float],
    box_b: tuple[float, float, float, float],
) -> float:
    """Standard IoU between two bounding boxes."""


def match_detections(
    ground_truth: list[Detection],
    predictions: list[Detection],
    iou_threshold: float = 0.5,
) -> tuple[list[tuple[Detection, Detection]], list[Detection], list[Detection]]:
    """Greedy IoU matching.

    Returns: (matched_pairs, unmatched_gt, unmatched_pred)
    """


class ComparisonMetrics(BaseModel):
    image_path: str
    num_gt_detections: int
    num_pred_detections: int
    num_matched: int
    num_missed: int
    num_false_positives: int
    avg_iou: float
    avg_confidence_delta: float
    class_mismatches: int


class ComparisonSummary(BaseModel):
    total_images: int
    total_gt_detections: int
    total_pred_detections: int
    precision: float
    recall: float
    avg_iou_matched: float
    avg_confidence_delta: float
    class_mismatch_rate: float
    per_image: list[ComparisonMetrics]
    mismatched_images: list[str]

    def to_json(self, path: Path) -> None: ...


def compare_inference(
    image_dir: Path,
    triton_model_name: str,
    ultralytics_model: str,
    triton_url: str = "localhost:8001",
    mode: Literal["raw", "dali"] = "raw",
    iou_threshold: float = 0.5,
    conf_threshold: float = 0.25,
) -> ComparisonSummary:
    """For each image: run ultralytics predict + Triton inference,
    match detections via IoU, compute metrics."""


def compare_backends(
    image_dir: Path,
    model_variants: list[tuple[str, ExportFormat, Literal["raw", "dali"]]],
    ultralytics_model: str,
    triton_url: str = "localhost:8001",
) -> dict[str, ComparisonSummary]:
    """Compare multiple backend/mode combinations against ultralytics baseline."""
```

### 5.2 CLI subcommand: `compare`

```bash
triton-model-tools compare \
  --image-dir ./data/val2017 \
  --triton-model yolo11s \
  --ultralytics-model yolo11s.pt \
  --mode raw \
  --iou-threshold 0.5 \
  --output comparison.json
```

### Reuses from Phase 3

- `TritonInferenceClient` (inference.py)
- `YoloPreprocessor`, `YoloPostprocessor` (inference.py)
- `Detection`, `ImageDetectionResult` (detection.py)

### Tests

- `compute_iou`: complete overlap (1.0), no overlap (0.0), known partial overlap
- `match_detections`: perfect match, partial, none, class mismatches
- `ComparisonMetrics` / `ComparisonSummary` from known detection sets
- `compare_inference` with mocked ultralytics + mocked Triton: known outputs,
  verify comparison metrics
- Edge cases: empty detections, single detection, all matching, none matching

### Dependencies

None new.

---

## Phase Summary

| Phase | Scope | Key Output |
|---|---|---|
| **1** | Generalize export + TensorRT | `export --format tensorrt` works |
| **2** | DALI ensemble generation | `export --dali-ensemble` creates 3-model ensemble |
| **3** | Inference client + folder tool | `infer --image-dir DIR --mode raw\|dali` |
| **4** | Benchmarking + COCO download | `benchmark` + `download-coco` commands |
| **5** | Prediction comparison | `compare` Triton vs ultralytics |

### Dependency Graph

```
Phase 1 ──► Phase 2
               │
               ▼
           Phase 3 ──► Phase 4 (benchmark)
               │
               └─────► Phase 5 (comparison)
```

Phases 4 and 5 are independent of each other, both depend on Phase 3.

### File Layout After All Phases

```
packages/triton_model_tools/
  pyproject.toml
  triton_model_tools/
    __init__.py
    __main__.py          # subcommand CLI (export, infer, benchmark, compare, download-coco)
    py.typed
    models.py            # + ExportFormat, platform/filename mappings, TritonEnsembleStep
    exporter.py          # ONNX + TensorRT export
    repository.py        # generalized model dir builder
    dali.py              # DALI pipeline + ensemble config generation
    detection.py         # Detection result models
    inference.py         # Triton client + YOLO pre/postprocessor
    folder_inference.py  # folder → Triton → results
    benchmark.py         # benchmarking engine
    dataset.py           # COCO val2017 downloader
    comparison.py        # prediction comparison tool
  tests/
    test_models.py           # updated
    test_repository.py       # updated
    test_dali.py
    test_detection.py
    test_inference.py
    test_folder_inference.py
    test_benchmark.py
    test_dataset.py
    test_comparison.py
```

### No New Dependencies

All phases use existing dependencies (`ultralytics`, `tritonclient[all]`, `pydantic`,
`loguru`, `numpy`, `Pillow`) plus Python stdlib (`urllib.request`, `ast`, `csv`,
`json`, `time`, `statistics`). No new packages needed.

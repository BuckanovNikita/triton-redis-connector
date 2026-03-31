"""Benchmark engine for comparing Triton and Ultralytics inference performance."""

from __future__ import annotations

import csv
import platform
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from triton_model_tools.benchmark_stats import MILLISECONDS_PER_SECOND, compute_statistics
from triton_model_tools.inference import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_IOU_THRESHOLD,
    TritonInferenceClient,
    YoloPostprocessor,
    YoloPreprocessor,
)

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np
    import numpy.typing as npt

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"})


class BenchmarkConfig(BaseModel):
    """Configuration for a single benchmark run."""

    model_config = ConfigDict(frozen=True)

    model_name: str
    backend: Literal["onnx", "tensorrt"] = "onnx"
    preprocess_mode: Literal["raw", "dali"] = "raw"
    triton_url: str = "localhost:8001"
    target_size: int = 640
    num_images: int = 100
    warmup_runs: int = 5
    end2end: bool = False
    conf_threshold: float = Field(default=DEFAULT_CONFIDENCE_THRESHOLD, ge=0.0, le=1.0)
    iou_threshold: float = Field(default=DEFAULT_IOU_THRESHOLD, ge=0.0, le=1.0)


class BenchmarkResult(BaseModel):
    """Results from a single benchmark run."""

    model_config = ConfigDict(frozen=True)

    config: BenchmarkConfig
    total_time_s: float = Field(ge=0.0)
    avg_latency_ms: float = Field(ge=0.0)
    throughput_fps: float = Field(ge=0.0)
    p50_latency_ms: float = Field(ge=0.0)
    p95_latency_ms: float = Field(ge=0.0)
    p99_latency_ms: float = Field(ge=0.0)
    num_images: int = Field(ge=0)


class BenchmarkSuite(BaseModel):
    """Collection of benchmark results with metadata."""

    model_config = ConfigDict(frozen=True)

    results: list[BenchmarkResult]
    timestamp: str = Field(default_factory=lambda: datetime.now(tz=UTC).isoformat())
    python_version: str = Field(default_factory=platform.python_version)
    platform_info: str = Field(default_factory=platform.platform)

    def to_json(self) -> str:
        """Serialize suite to JSON string."""
        return self.model_dump_json(indent=2)

    def to_csv(self, path: Path) -> None:
        """Write results to CSV file.

        Args:
            path: Output CSV file path.

        """
        if not self.results:
            logger.warning("No results to write to CSV")
            return

        fieldnames = [
            "model_name",
            "backend",
            "preprocess_mode",
            "num_images",
            "total_time_s",
            "avg_latency_ms",
            "throughput_fps",
            "p50_latency_ms",
            "p95_latency_ms",
            "p99_latency_ms",
        ]

        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for result in self.results:
                writer.writerow(
                    {
                        "model_name": result.config.model_name,
                        "backend": result.config.backend,
                        "preprocess_mode": result.config.preprocess_mode,
                        "num_images": result.num_images,
                        "total_time_s": f"{result.total_time_s:.4f}",
                        "avg_latency_ms": f"{result.avg_latency_ms:.2f}",
                        "throughput_fps": f"{result.throughput_fps:.2f}",
                        "p50_latency_ms": f"{result.p50_latency_ms:.2f}",
                        "p95_latency_ms": f"{result.p95_latency_ms:.2f}",
                        "p99_latency_ms": f"{result.p99_latency_ms:.2f}",
                    }
                )

        logger.info("Benchmark results written to {}", path)


def _collect_image_paths(image_dir: Path, max_count: int) -> list[Path]:
    paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not paths:
        msg = f"No image files found in {image_dir}"
        raise FileNotFoundError(msg)
    return paths[:max_count]


def _load_image(path: Path) -> npt.NDArray[np.uint8]:
    import cv2  # noqa: PLC0415  # pyright: ignore[reportMissingTypeStubs]

    img: npt.NDArray[np.uint8] | None = cv2.imread(str(path))  # type: ignore[assignment]  # pyright: ignore[reportUnknownMemberType,reportAssignmentType]
    if img is None:
        msg = f"Failed to load image: {path}"
        raise FileNotFoundError(msg)
    return img


def run_benchmark(config: BenchmarkConfig, image_dir: Path) -> BenchmarkResult:
    """Run Triton inference benchmark.

    Args:
        config: Benchmark configuration.
        image_dir: Directory containing images.

    Returns:
        BenchmarkResult with timing statistics.

    """
    image_paths = _collect_image_paths(image_dir, config.num_images)
    logger.info("Running benchmark: model={} backend={} images={}", config.model_name, config.backend, len(image_paths))

    client = TritonInferenceClient(url=config.triton_url)
    preprocessor = YoloPreprocessor(target_size=config.target_size)
    postprocessor = YoloPostprocessor(end2end=config.end2end)

    output_names = ["output0"]

    if config.warmup_runs > 0:
        logger.info("Running {} warmup iterations", config.warmup_runs)
        warmup_image = _load_image(image_paths[0])
        warmup_tensor, _, _ = preprocessor.preprocess(warmup_image)
        for _ in range(config.warmup_runs):
            client.infer_raw(config.model_name, warmup_tensor, output_names)

    latencies_ms: list[float] = []
    for image_path in image_paths:
        image = _load_image(image_path)

        start = time.perf_counter()
        tensor, scale, pad_offset = preprocessor.preprocess(image)
        outputs = client.infer_raw(config.model_name, tensor, output_names)
        postprocessor.postprocess(
            outputs,
            scale,
            pad_offset,
            conf_threshold=config.conf_threshold,
            iou_threshold=config.iou_threshold,
        )
        elapsed_ms = (time.perf_counter() - start) * MILLISECONDS_PER_SECOND
        latencies_ms.append(elapsed_ms)

    total_s, avg_ms, throughput, p50, p95, p99 = compute_statistics(latencies_ms)

    logger.info(
        "Benchmark complete: avg={:.2f}ms p50={:.2f}ms p95={:.2f}ms p99={:.2f}ms throughput={:.1f}fps",
        avg_ms,
        p50,
        p95,
        p99,
        throughput,
    )

    return BenchmarkResult(
        config=config,
        total_time_s=total_s,
        avg_latency_ms=avg_ms,
        throughput_fps=throughput,
        p50_latency_ms=p50,
        p95_latency_ms=p95,
        p99_latency_ms=p99,
        num_images=len(latencies_ms),
    )


def run_ultralytics_benchmark(
    model_path: str,
    image_dir: Path,
    num_images: int,
) -> BenchmarkResult:
    """Run ultralytics model.predict() baseline benchmark.

    Args:
        model_path: Path or name of ultralytics model (e.g. 'yolo11s.pt').
        image_dir: Directory containing images.
        num_images: Maximum number of images to process.

    Returns:
        BenchmarkResult with timing statistics.

    """
    image_paths = _collect_image_paths(image_dir, num_images)
    logger.info("Running ultralytics baseline benchmark: model={} images={}", model_path, len(image_paths))

    from ultralytics import YOLO  # type: ignore[attr-defined]  # noqa: PLC0415

    model: Any = YOLO(model_path)  # pyright: ignore[reportUnknownVariableType]

    latencies_ms: list[float] = []
    for image_path in image_paths:
        start = time.perf_counter()
        model.predict(str(image_path), verbose=False)  # pyright: ignore[reportUnknownMemberType]
        elapsed_ms = (time.perf_counter() - start) * MILLISECONDS_PER_SECOND
        latencies_ms.append(elapsed_ms)

    total_s, avg_ms, throughput, p50, p95, p99 = compute_statistics(latencies_ms)

    logger.info(
        "Ultralytics benchmark complete: avg={:.2f}ms p50={:.2f}ms throughput={:.1f}fps",
        avg_ms,
        p50,
        throughput,
    )

    config = BenchmarkConfig(model_name=model_path, backend="onnx", preprocess_mode="raw", num_images=num_images)

    return BenchmarkResult(
        config=config,
        total_time_s=total_s,
        avg_latency_ms=avg_ms,
        throughput_fps=throughput,
        p50_latency_ms=p50,
        p95_latency_ms=p95,
        p99_latency_ms=p99,
        num_images=len(latencies_ms),
    )


def run_benchmark_matrix(configs: list[BenchmarkConfig], image_dir: Path) -> BenchmarkSuite:
    """Run benchmark for all configurations.

    Args:
        configs: List of benchmark configurations to run.
        image_dir: Directory containing images.

    Returns:
        BenchmarkSuite with all results and metadata.

    """
    logger.info("Running benchmark matrix with {} configurations", len(configs))
    results: list[BenchmarkResult] = []

    for i, config in enumerate(configs):
        logger.info("Running config {}/{}: {} ({})", i + 1, len(configs), config.model_name, config.backend)
        result = run_benchmark(config, image_dir)
        results.append(result)

    suite = BenchmarkSuite(results=results)
    logger.info("Benchmark matrix complete: {} configurations", len(results))
    return suite


def build_full_matrix(
    models: list[str],
    triton_url: str = "localhost:8001",
    target_size: int = 640,
    num_images: int = 100,
    warmup_runs: int = 5,
) -> list[BenchmarkConfig]:
    """Generate all BenchmarkConfig combinations for a full performance matrix.

    Creates configs for each model across all backend/preprocess combinations:
    - ONNX + raw letterbox
    - ONNX + DALI ensemble
    - TensorRT + raw letterbox
    - TensorRT + DALI ensemble

    Args:
        models: List of model names (e.g. ["yolo11s", "yolo26s", "yolo12s"]).
        triton_url: Triton gRPC URL.
        target_size: Model input size.
        num_images: Number of images per benchmark run.
        warmup_runs: Number of warmup iterations per run.

    Returns:
        List of BenchmarkConfig for all combinations.

    """
    backends: list[Literal["onnx", "tensorrt"]] = ["onnx", "tensorrt"]
    preprocess_modes: list[Literal["raw", "dali"]] = ["raw", "dali"]

    configs: list[BenchmarkConfig] = [
        BenchmarkConfig(
            model_name=model_name,
            backend=backend,
            preprocess_mode=preprocess_mode,
            triton_url=triton_url,
            target_size=target_size,
            num_images=num_images,
            warmup_runs=warmup_runs,
        )
        for model_name in models
        for backend in backends
        for preprocess_mode in preprocess_modes
    ]

    logger.info(
        "Built full benchmark matrix: {} models x {} configs = {} total",
        len(models),
        len(backends) * len(preprocess_modes),
        len(configs),
    )
    return configs


def save_benchmark_output(suite: BenchmarkSuite, output_path: Path) -> None:
    """Save benchmark suite to file (CSV if .csv extension, JSON otherwise).

    Args:
        suite: Benchmark suite to save.
        output_path: Output file path.

    """
    if output_path.suffix == ".csv":
        suite.to_csv(output_path)
    else:
        output_path.write_text(suite.to_json())
        logger.info("Benchmark results written to {}", output_path)

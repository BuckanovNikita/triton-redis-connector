"""Benchmark engine for comparing Triton and Ultralytics inference performance."""

from __future__ import annotations

import csv
import platform
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field
from ultralytics import YOLO  # type: ignore[attr-defined]

from triton_model_tools.benchmarks.memory import MemoryMonitor
from triton_model_tools.benchmarks.stats import MILLISECONDS_PER_SECOND, compute_statistics
from triton_model_tools.export.exporter import TENSORRT_NAME_SUFFIX
from triton_model_tools.inference.client import TritonInferenceClient
from triton_model_tools.inference.detection_io import collect_image_paths, load_image
from triton_model_tools.inference.postprocessing import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_IOU_THRESHOLD,
    YoloPostprocessor,
)
from triton_model_tools.inference.preprocessing import YoloPreprocessor

if TYPE_CHECKING:
    from pathlib import Path


class BenchmarkConfig(BaseModel):
    """Configuration for a single benchmark run."""

    model_config = ConfigDict(frozen=True)

    model_name: str
    backend: Literal["onnx", "tensorrt"] = "onnx"
    preprocess_mode: Literal["raw", "dali"] = "raw"
    triton_url: str = "localhost:8001"
    target_size: int | tuple[int, int] = 640
    num_images: int = 100
    warmup_runs: int = 5
    end2end: bool = False
    half: bool = False
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
    peak_vram_mb: float | None = None
    peak_ram_mb: float | None = None


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
            "peak_vram_mb",
            "peak_ram_mb",
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
                        "peak_vram_mb": f"{result.peak_vram_mb:.1f}" if result.peak_vram_mb is not None else "",
                        "peak_ram_mb": f"{result.peak_ram_mb:.1f}" if result.peak_ram_mb is not None else "",
                    }
                )

        logger.info("Benchmark results written to {}", path)


def run_benchmark(config: BenchmarkConfig, image_dir: Path) -> BenchmarkResult:
    """Run Triton inference benchmark.

    Args:
        config: Benchmark configuration.
        image_dir: Directory containing images.

    Returns:
        BenchmarkResult with timing statistics.

    """
    image_paths = collect_image_paths(image_dir, max_count=config.num_images)

    triton_model_name = config.model_name
    if config.backend == "tensorrt" and not triton_model_name.endswith(TENSORRT_NAME_SUFFIX):
        triton_model_name = f"{triton_model_name}{TENSORRT_NAME_SUFFIX}"

    use_fp16 = config.half or config.backend == "tensorrt"
    preprocess_dtype: type[np.floating[Any]] = np.float16 if use_fp16 else np.float32

    logger.info(
        "Running benchmark: model={} triton_name={} backend={} half={} images={}",
        config.model_name,
        triton_model_name,
        config.backend,
        use_fp16,
        len(image_paths),
    )

    client = TritonInferenceClient(url=config.triton_url)
    preprocessor = YoloPreprocessor(target_size=config.target_size, dtype=preprocess_dtype)
    postprocessor = YoloPostprocessor(end2end=config.end2end)

    output_names = ["output0"]

    if config.warmup_runs > 0:
        logger.info("Running {} warmup iterations", config.warmup_runs)
        warmup_image = load_image(image_paths[0])
        warmup_tensor, _, _ = preprocessor.preprocess(warmup_image)
        for _ in range(config.warmup_runs):
            client.infer_raw(triton_model_name, warmup_tensor, output_names)

    latencies_ms: list[float] = []
    with MemoryMonitor() as mem:
        for image_path in image_paths:
            image = load_image(image_path)

            start = time.perf_counter()
            tensor, scale, pad_offset = preprocessor.preprocess(image)
            outputs = client.infer_raw(triton_model_name, tensor, output_names)
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

    if mem.peak_vram_mb is not None:
        logger.info("Peak VRAM: {:.1f} MB", mem.peak_vram_mb)
    if mem.peak_ram_mb is not None:
        logger.info("Peak RAM: {:.1f} MB", mem.peak_ram_mb)

    return BenchmarkResult(
        config=config,
        total_time_s=total_s,
        avg_latency_ms=avg_ms,
        throughput_fps=throughput,
        p50_latency_ms=p50,
        p95_latency_ms=p95,
        p99_latency_ms=p99,
        num_images=len(latencies_ms),
        peak_vram_mb=mem.peak_vram_mb,
        peak_ram_mb=mem.peak_ram_mb,
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
    image_paths = collect_image_paths(image_dir, max_count=num_images)
    logger.info("Running ultralytics baseline benchmark: model={} images={}", model_path, len(image_paths))

    model: Any = YOLO(model_path)

    latencies_ms: list[float] = []
    for image_path in image_paths:
        start = time.perf_counter()
        model.predict(str(image_path), verbose=False)
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
    target_size: int | tuple[int, int] = 640,
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


class BatchSizeBenchmarkConfig(BaseModel):
    """Configuration for batch-size sweep benchmark using ultralytics YOLO.predict()."""

    model_config = ConfigDict(frozen=True)

    model_path: str = "yolo11s.pt"
    batch_sizes: list[int] = Field(default=[1, 2, 4, 8, 16, 32])
    num_images: int = Field(default=100, ge=1)
    target_size: int | tuple[int, int] = 640
    warmup_runs: int = Field(default=5, ge=0)


class BatchSizeBenchmarkResult(BaseModel):
    """Results from a batch-size sweep benchmark."""

    model_config = ConfigDict(frozen=True)

    config: BatchSizeBenchmarkConfig
    per_batch_size: list[BenchmarkResult]


def run_batch_size_benchmark(config: BatchSizeBenchmarkConfig, image_dir: Path) -> BatchSizeBenchmarkResult:
    """Benchmark ultralytics YOLO.predict() across multiple batch sizes.

    Loads the model once, then for each batch_size groups images into batches,
    times model.predict() on each batch, and computes throughput statistics.

    Args:
        config: Batch-size benchmark configuration.
        image_dir: Directory containing images.

    Returns:
        BatchSizeBenchmarkResult with one BenchmarkResult per batch_size.

    """
    image_paths = collect_image_paths(image_dir, max_count=config.num_images)
    logger.info(
        "Batch-size benchmark: model={} images={} batch_sizes={}",
        config.model_path,
        len(image_paths),
        config.batch_sizes,
    )

    model: Any = YOLO(config.model_path)
    image_path_strs = [str(p) for p in image_paths]

    if config.warmup_runs > 0:
        logger.info("Running {} warmup iterations", config.warmup_runs)
        for _ in range(config.warmup_runs):
            model.predict(image_path_strs[0], verbose=False)

    per_batch_size: list[BenchmarkResult] = []
    for batch_size in config.batch_sizes:
        batches = [image_path_strs[i : i + batch_size] for i in range(0, len(image_path_strs), batch_size)]

        latencies_ms: list[float] = []
        for batch in batches:
            start = time.perf_counter()
            model.predict(batch, verbose=False)
            elapsed_ms = (time.perf_counter() - start) * MILLISECONDS_PER_SECOND
            per_image_ms = elapsed_ms / len(batch)
            latencies_ms.extend([per_image_ms] * len(batch))

        total_s, avg_ms, throughput, p50, p95, p99 = compute_statistics(latencies_ms)

        bench_config = BenchmarkConfig(
            model_name=config.model_path,
            backend="onnx",
            preprocess_mode="raw",
            target_size=config.target_size,
            num_images=len(latencies_ms),
            warmup_runs=config.warmup_runs,
        )

        result = BenchmarkResult(
            config=bench_config,
            total_time_s=total_s,
            avg_latency_ms=avg_ms,
            throughput_fps=throughput,
            p50_latency_ms=p50,
            p95_latency_ms=p95,
            p99_latency_ms=p99,
            num_images=len(latencies_ms),
        )
        per_batch_size.append(result)

        logger.info(
            "batch_size={}: avg={:.2f}ms p50={:.2f}ms throughput={:.1f}fps",
            batch_size,
            avg_ms,
            p50,
            throughput,
        )

    return BatchSizeBenchmarkResult(config=config, per_batch_size=per_batch_size)


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

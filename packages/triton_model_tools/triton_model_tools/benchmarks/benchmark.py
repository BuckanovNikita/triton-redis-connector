"""Benchmark engine for Triton inference performance measurement."""

import csv
import platform
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field
from tqdm import tqdm
from triton_shared_models.detection.postprocessing import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_IOU_THRESHOLD,
    YoloPostprocessor,
)
from triton_shared_models.detection.preprocessing import LetterBoxPreprocessor

from triton_model_tools.benchmarks.stats import compute_statistics
from triton_model_tools.export.exporter import TENSORRT_NAME_SUFFIX
from triton_model_tools.inference.client import TritonInferenceClient
from triton_model_tools.inference.detection_io import collect_image_paths, load_image
from triton_model_tools.models.constants import MILLISECONDS_PER_SECOND


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
    min_images: int = Field(default=100, ge=0)


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


def run_benchmark(config: BenchmarkConfig, image_dir: Path) -> BenchmarkResult:
    """Run Triton inference benchmark.

    Args:
        config: Benchmark configuration.
        image_dir: Directory containing images.

    Returns:
        BenchmarkResult with timing statistics.

    """
    image_paths = collect_image_paths(image_dir, max_count=config.num_images)

    if config.min_images > 0 and len(image_paths) < config.min_images:
        logger.warning(
            "Only {} images available, recommended minimum is {} for reliable benchmarks",
            len(image_paths),
            config.min_images,
        )

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
    preprocessor = LetterBoxPreprocessor(target_size=config.target_size, dtype=preprocess_dtype)
    postprocessor = YoloPostprocessor(end2end=config.end2end)

    output_names = ["output0"]

    if config.warmup_runs > 0:
        logger.info("Running {} warmup iterations", config.warmup_runs)
        warmup_image = load_image(image_paths[0])
        warmup_tensor, _, _ = preprocessor.preprocess(warmup_image)
        for _ in range(config.warmup_runs):
            client.infer_raw(triton_model_name, warmup_tensor, output_names)

    latencies_ms: list[float] = []
    for image_path in tqdm(image_paths, desc="Benchmark", unit="img"):
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


def save_benchmark_output(suite: BenchmarkSuite, output_path: Path) -> None:
    """Save benchmark suite to file (CSV if .csv extension, JSON otherwise).

    Args:
        suite: Benchmark suite to save.
        output_path: Output file path.

    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == ".csv":
        suite.to_csv(output_path)
    else:
        output_path.write_text(suite.to_json())
        logger.info("Benchmark results written to {}", output_path)

"""End-to-end performance measurement of the Redis detection pipeline."""

from __future__ import annotations

import time
import uuid
from typing import TYPE_CHECKING, Any

import redis as redis_lib
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field
from tqdm import tqdm

from triton_model_tools.benchmarks.stats import compute_statistics
from triton_model_tools.inference.detection_io import collect_image_paths
from triton_model_tools.models.constants import MILLISECONDS_PER_SECOND
from triton_model_tools.testing.e2e_test import CONSUMER_NAME, POLL_INTERVAL_S, ensure_consumer_group

if TYPE_CHECKING:
    from pathlib import Path

    import redis

BACKENDS: list[str] = ["onnx", "tensorrt"]
PREPROCESS_MODES: list[str] = ["dali", "cv2"]


class PipelineBenchmarkConfig(BaseModel):
    """Configuration for pipeline E2E benchmark."""

    model_config = ConfigDict(frozen=True)

    redis_url: str = "redis://redis:6379"
    input_stream: str = "camera_images"
    output_stream: str = "detection_results"
    model_name: str = "yolo11s"
    consumer_group: str = "pipeline-bench"
    num_iterations: int = Field(default=3, ge=1)
    warmup_runs: int = Field(default=1, ge=0)
    timeout_s: float = Field(default=60.0, gt=0.0)


class PipelineBenchmarkResult(BaseModel):
    """Results from a pipeline E2E benchmark run."""

    model_config = ConfigDict(frozen=True)

    config: PipelineBenchmarkConfig
    total_time_s: float = Field(ge=0.0)
    avg_latency_ms: float = Field(ge=0.0)
    throughput_fps: float = Field(ge=0.0)
    p50_latency_ms: float = Field(ge=0.0)
    p95_latency_ms: float = Field(ge=0.0)
    p99_latency_ms: float = Field(ge=0.0)
    num_images: int = Field(ge=0)
    num_iterations: int = Field(ge=0)


def _push_images(
    redis_client: redis.Redis[bytes],
    input_stream: str,
    image_paths: list[Path],
    model_name: str,
) -> dict[str, float]:
    """Push images to Redis and record push timestamps. Returns {image_name: push_time}."""
    push_times: dict[str, float] = {}
    for image_path in tqdm(image_paths, desc="Pushing images", unit="img"):
        image_bytes = image_path.read_bytes()
        push_times[image_path.name] = time.monotonic()
        redis_client.xadd(
            name=input_stream,
            fields={
                b"data": image_bytes,
                b"request_id": uuid.uuid4().hex.encode(),
                b"model_name": model_name.encode(),
                b"image_name": image_path.name.encode(),
            },
        )
    return push_times


def _collect_with_timestamps(
    redis_client: redis.Redis[bytes],
    output_stream: str,
    consumer_group: str,
    expected_count: int,
    timeout_s: float,
) -> list[float]:
    """Collect results and return per-result collection timestamps."""
    collect_times: list[float] = []
    deadline = time.monotonic() + timeout_s
    block_ms = int(POLL_INTERVAL_S * MILLISECONDS_PER_SECOND)

    while len(collect_times) < expected_count and time.monotonic() < deadline:
        raw_messages: Any = redis_client.xreadgroup(
            groupname=consumer_group,
            consumername=CONSUMER_NAME,
            streams={output_stream: ">"},
            count=expected_count - len(collect_times),
            block=block_ms,
        )

        if not raw_messages:
            continue

        for _stream_name, entries in raw_messages:
            for msg_id, _fields in entries:
                collect_times.append(time.monotonic())
                redis_client.xack(  # type: ignore[no-untyped-call]
                    output_stream,
                    consumer_group,
                    msg_id,
                )

    return collect_times


def run_pipeline_benchmark(
    config: PipelineBenchmarkConfig,
    image_dir: Path,
) -> PipelineBenchmarkResult:
    """Measure E2E pipeline performance over multiple iterations.

    For each iteration:
    1. Push all images to Redis input stream
    2. Collect all results from output stream
    3. Measure per-batch latency (total push-to-last-collect time)

    Args:
        config: Pipeline benchmark configuration.
        image_dir: Directory containing images.

    Returns:
        PipelineBenchmarkResult with latency statistics.

    """
    image_paths = collect_image_paths(image_dir)
    logger.info(
        "Pipeline benchmark: model={} images={} iterations={}",
        config.model_name,
        len(image_paths),
        config.num_iterations,
    )

    redis_client: redis.Redis[bytes] = redis_lib.Redis.from_url(
        config.redis_url,
        decode_responses=False,
    )
    ensure_consumer_group(redis_client, config.output_stream, config.consumer_group)

    if config.warmup_runs > 0:
        logger.info("Running {} warmup iterations", config.warmup_runs)
        for _ in range(config.warmup_runs):
            _push_images(redis_client, config.input_stream, image_paths, config.model_name)
            _collect_with_timestamps(
                redis_client,
                config.output_stream,
                config.consumer_group,
                len(image_paths),
                config.timeout_s,
            )

    latencies_ms: list[float] = []
    for iteration in tqdm(range(config.num_iterations), desc="Iterations", unit="iter"):
        batch_start = time.monotonic()
        _push_images(redis_client, config.input_stream, image_paths, config.model_name)
        collect_times = _collect_with_timestamps(
            redis_client,
            config.output_stream,
            config.consumer_group,
            len(image_paths),
            config.timeout_s,
        )

        if collect_times:
            batch_latency_ms = (collect_times[-1] - batch_start) * MILLISECONDS_PER_SECOND
            per_image_ms = batch_latency_ms / len(collect_times)
            latencies_ms.extend([per_image_ms] * len(collect_times))
            logger.info(
                "Iteration {}/{}: {:.1f}ms total, {:.1f}ms/image ({} results)",
                iteration + 1,
                config.num_iterations,
                batch_latency_ms,
                per_image_ms,
                len(collect_times),
            )
        else:
            logger.warning("Iteration {}/{}: no results collected", iteration + 1, config.num_iterations)

    if not latencies_ms:
        return PipelineBenchmarkResult(
            config=config,
            total_time_s=0.0,
            avg_latency_ms=0.0,
            throughput_fps=0.0,
            p50_latency_ms=0.0,
            p95_latency_ms=0.0,
            p99_latency_ms=0.0,
            num_images=0,
            num_iterations=config.num_iterations,
        )

    total_s, avg_ms, throughput, p50, p95, p99 = compute_statistics(latencies_ms)

    logger.info(
        "Pipeline benchmark complete: avg={:.2f}ms p50={:.2f}ms p95={:.2f}ms throughput={:.1f}fps",
        avg_ms,
        p50,
        p95,
        throughput,
    )

    return PipelineBenchmarkResult(
        config=config,
        total_time_s=total_s,
        avg_latency_ms=avg_ms,
        throughput_fps=throughput,
        p50_latency_ms=p50,
        p95_latency_ms=p95,
        p99_latency_ms=p99,
        num_images=len(latencies_ms),
        num_iterations=config.num_iterations,
    )


def build_pipeline_matrix(  # noqa: PLR0913
    models: list[str],
    *,
    backends: list[str] | None = None,
    preprocess_modes: list[str] | None = None,
    image_sizes: list[int] | None = None,
    redis_url: str = "redis://redis:6379",
    input_stream: str = "camera_images",
    output_stream: str = "detection_results",
    num_iterations: int = 3,
    warmup_runs: int = 1,
    timeout_s: float = 60.0,
) -> list[PipelineBenchmarkConfig]:
    """Generate PipelineBenchmarkConfig for all combinations of parameters.

    Args:
        models: Model names to benchmark.
        backends: Backend types (default: onnx, tensorrt).
        preprocess_modes: Preprocessing modes (default: dali, cv2).
        image_sizes: Image sizes (not used in PipelineBenchmarkConfig directly,
            included for matrix description in model_name).
        redis_url: Redis connection URL.
        input_stream: Redis input stream name.
        output_stream: Redis output stream name.
        num_iterations: Iterations per benchmark run.
        warmup_runs: Warmup iterations per run.
        timeout_s: Timeout in seconds per iteration.

    Returns:
        List of PipelineBenchmarkConfig for all combinations.

    """
    effective_backends = backends if backends is not None else BACKENDS
    effective_preprocess = preprocess_modes if preprocess_modes is not None else PREPROCESS_MODES
    effective_sizes = image_sizes if image_sizes is not None else [640]

    configs: list[PipelineBenchmarkConfig] = []
    for model_name in models:
        for backend in effective_backends:
            for preprocess in effective_preprocess:
                for size in effective_sizes:
                    label = f"{model_name}_{backend}_{preprocess}_{size}"
                    configs.append(
                        PipelineBenchmarkConfig(
                            redis_url=redis_url,
                            input_stream=input_stream,
                            output_stream=output_stream,
                            model_name=label,
                            num_iterations=num_iterations,
                            warmup_runs=warmup_runs,
                            timeout_s=timeout_s,
                        )
                    )

    logger.info(
        "Built pipeline benchmark matrix: {} models x {} backends x {} preprocess x {} sizes = {} total",
        len(models),
        len(effective_backends),
        len(effective_preprocess),
        len(effective_sizes),
        len(configs),
    )
    return configs


def run_pipeline_benchmark_matrix(
    configs: list[PipelineBenchmarkConfig],
    image_dir: Path,
) -> list[PipelineBenchmarkResult]:
    """Run pipeline benchmark for all configurations.

    Args:
        configs: List of pipeline benchmark configurations.
        image_dir: Directory containing images.

    Returns:
        List of PipelineBenchmarkResult for each configuration.

    """
    logger.info("Running pipeline benchmark matrix with {} configurations", len(configs))
    results: list[PipelineBenchmarkResult] = []

    for i, config in enumerate(configs):
        logger.info("Running config {}/{}: {}", i + 1, len(configs), config.model_name)
        result = run_pipeline_benchmark(config, image_dir)
        results.append(result)

    logger.info("Pipeline benchmark matrix complete: {} configurations", len(results))
    return results

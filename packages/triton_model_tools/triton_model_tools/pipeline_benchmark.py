"""End-to-end performance measurement of the Redis detection pipeline."""

from __future__ import annotations

import time
import uuid
from typing import TYPE_CHECKING, Any

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from triton_model_tools.benchmark_stats import MILLISECONDS_PER_SECOND, compute_statistics
from triton_model_tools.e2e_test import CONSUMER_NAME, IMAGE_EXTENSIONS, POLL_INTERVAL_S, _ensure_consumer_group

if TYPE_CHECKING:
    from pathlib import Path

    import redis


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


def _collect_image_paths(image_dir: Path) -> list[Path]:
    paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not paths:
        msg = f"No image files found in {image_dir}"
        raise FileNotFoundError(msg)
    return paths


def _push_images(
    redis_client: redis.Redis[bytes],
    input_stream: str,
    image_paths: list[Path],
    model_name: str,
) -> dict[str, float]:
    """Push images to Redis and record push timestamps. Returns {image_name: push_time}."""
    push_times: dict[str, float] = {}
    for image_path in image_paths:
        image_bytes = image_path.read_bytes()
        push_times[image_path.name] = time.monotonic()
        redis_client.xadd(  # pyright: ignore[reportUnknownMemberType]
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
        raw_messages: Any = redis_client.xreadgroup(  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
            groupname=consumer_group,
            consumername=CONSUMER_NAME,
            streams={output_stream: ">"},
            count=expected_count - len(collect_times),
            block=block_ms,
        )

        if not raw_messages:
            continue

        for _stream_name, entries in raw_messages:  # pyright: ignore[reportUnknownVariableType]
            for msg_id, _fields in entries:  # pyright: ignore[reportUnknownVariableType]
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
    import redis as redis_lib  # noqa: PLC0415

    image_paths = _collect_image_paths(image_dir)
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
    _ensure_consumer_group(redis_client, config.output_stream, config.consumer_group)

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
    for iteration in range(config.num_iterations):
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

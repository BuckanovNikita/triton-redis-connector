"""Measure E2E performance of the Redis detection pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    import argparse

    from triton_model_tools.cli._types import SubParsers


def build_parser(subparsers: SubParsers) -> None:
    """Register the ``pipeline-benchmark`` subcommand."""
    pb_parser = subparsers.add_parser(
        "pipeline-benchmark",
        help="Measure E2E performance of the Redis detection pipeline",
    )
    pb_parser.add_argument(
        "--image-dir",
        type=Path,
        required=True,
        help="Directory containing images to benchmark with",
    )
    pb_parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
        help="Triton model name (e.g. yolo11s)",
    )
    pb_parser.add_argument(
        "--redis-url",
        type=str,
        default="redis://redis:6379",
        help="Redis URL (default: redis://redis:6379)",
    )
    pb_parser.add_argument(
        "--input-stream",
        type=str,
        default="camera_images",
        help="Redis input stream name (default: camera_images)",
    )
    pb_parser.add_argument(
        "--output-stream",
        type=str,
        default="detection_results",
        help="Redis output stream name (default: detection_results)",
    )
    pb_parser.add_argument(
        "--num-iterations",
        type=int,
        default=3,
        help="Number of benchmark iterations (default: 3)",
    )
    pb_parser.add_argument(
        "--warmup-runs",
        type=int,
        default=1,
        help="Number of warmup iterations (default: 1)",
    )
    pb_parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Timeout in seconds for collecting results per iteration (default: 60.0)",
    )
    pb_parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output file path for results JSON",
    )
    pb_parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    """Execute the ``pipeline-benchmark`` subcommand."""
    from triton_model_tools.benchmarks.pipeline_benchmark import (  # noqa: PLC0415
        PipelineBenchmarkConfig,
        run_pipeline_benchmark,
    )

    image_dir: Path = args.image_dir.resolve()
    model_name: str = args.model
    redis_url: str = args.redis_url
    input_stream: str = args.input_stream
    output_stream: str = args.output_stream
    num_iterations: int = args.num_iterations
    warmup_runs: int = args.warmup_runs
    timeout_s: float = args.timeout
    output_path: Path | None = args.output

    logger.info("Pipeline E2E Benchmark")
    logger.info("Image dir: {}", image_dir)
    logger.info("Model: {}", model_name)
    logger.info("Redis: {}", redis_url)

    config = PipelineBenchmarkConfig(
        redis_url=redis_url,
        input_stream=input_stream,
        output_stream=output_stream,
        model_name=model_name,
        num_iterations=num_iterations,
        warmup_runs=warmup_runs,
        timeout_s=timeout_s,
    )

    result = run_pipeline_benchmark(config, image_dir)

    logger.info(
        "Results: {} images, avg={:.2f}ms, throughput={:.1f}fps",
        result.num_images,
        result.avg_latency_ms,
        result.throughput_fps,
    )
    logger.info(
        "Latencies: p50={:.2f}ms p95={:.2f}ms p99={:.2f}ms",
        result.p50_latency_ms,
        result.p95_latency_ms,
        result.p99_latency_ms,
    )

    if output_path is not None:
        import json  # noqa: PLC0415

        resolved_output = output_path.resolve()
        resolved_output.parent.mkdir(parents=True, exist_ok=True)
        with resolved_output.open("w") as f:
            json.dump(result.model_dump(), f, indent=2)
        logger.info("Benchmark results saved to {}", resolved_output)

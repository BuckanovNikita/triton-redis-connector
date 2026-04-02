"""Run inference benchmark on all models discovered in a Triton model repository."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    import argparse

    from triton_model_tools.cli._types import SubParsers

MIN_DIMS_FOR_SIZE = 4


def build_parser(subparsers: SubParsers) -> None:
    """Register the ``benchmark-all`` subcommand."""
    ba_parser = subparsers.add_parser(
        "benchmark-all",
        help="Run inference benchmark on all models in a Triton model repository",
    )
    ba_parser.add_argument(
        "--image-dir",
        type=Path,
        required=True,
        help="Directory containing images to benchmark with",
    )
    ba_parser.add_argument(
        "--repository-root",
        type=Path,
        required=True,
        help="Path to Triton model repository",
    )
    ba_parser.add_argument(
        "--triton-url",
        type=str,
        default="localhost:8001",
        help="Triton gRPC URL (default: localhost:8001)",
    )
    ba_parser.add_argument(
        "--num-images",
        type=int,
        default=100,
        help="Number of images to benchmark per model (default: 100)",
    )
    ba_parser.add_argument(
        "--warmup-runs",
        type=int,
        default=5,
        help="Number of warmup iterations per model (default: 5)",
    )
    ba_parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output file path (JSON or CSV based on extension)",
    )
    ba_parser.set_defaults(func=run)


def _target_size_from_dims(input_dims: list[int]) -> int | tuple[int, int]:
    if input_dims and len(input_dims) >= MIN_DIMS_FOR_SIZE:
        h, w = input_dims[-2], input_dims[-1]
        return h if h == w else (h, w)
    return 640


def run(args: argparse.Namespace) -> None:
    """Execute the ``benchmark-all`` subcommand."""
    from triton_model_tools.benchmarks.benchmark import (  # noqa: PLC0415
        BenchmarkConfig,
        BenchmarkResult,
        BenchmarkSuite,
        run_benchmark,
        save_benchmark_output,
    )
    from triton_model_tools.models import discover_models  # noqa: PLC0415

    image_dir: Path = args.image_dir.resolve()
    repository_root: Path = args.repository_root.resolve()
    triton_url: str = args.triton_url
    num_images: int = args.num_images
    warmup_runs: int = args.warmup_runs
    output_path: Path | None = args.output

    logger.info("Benchmark All Models")
    logger.info("Image dir: {}", image_dir)
    logger.info("Repository root: {}", repository_root)

    discovered = discover_models(repository_root)
    if not discovered:
        logger.warning("No models discovered in {}", repository_root)
        return

    logger.info("Discovered {} model(s), starting benchmarks", len(discovered))

    results: list[BenchmarkResult] = []
    for model in discovered:
        target_size = _target_size_from_dims(model.input_dims)
        logger.info(
            "Benchmarking model: {} (end2end={}, target_size={})",
            model.name,
            model.is_end2end,
            target_size,
        )

        config = BenchmarkConfig(
            model_name=model.name,
            triton_url=triton_url,
            target_size=target_size,
            num_images=num_images,
            warmup_runs=warmup_runs,
            end2end=model.is_end2end,
        )

        result = run_benchmark(config, image_dir)
        results.append(result)

    suite = BenchmarkSuite(results=results)

    logger.info("=" * 80)
    logger.info("Benchmark Summary ({} models)", len(results))
    logger.info(
        "{:<30s} {:>10s} {:>12s} {:>10s} {:>10s} {:>10s}",
        "Model",
        "Avg (ms)",
        "Thput (fps)",
        "p50 (ms)",
        "p95 (ms)",
        "p99 (ms)",
    )
    logger.info("-" * 80)
    for result in results:
        logger.info(
            "{:<30s} {:>10.2f} {:>12.1f} {:>10.2f} {:>10.2f} {:>10.2f}",
            result.config.model_name,
            result.avg_latency_ms,
            result.throughput_fps,
            result.p50_latency_ms,
            result.p95_latency_ms,
            result.p99_latency_ms,
        )
    logger.info("=" * 80)

    if output_path is not None:
        save_benchmark_output(suite, output_path.resolve())

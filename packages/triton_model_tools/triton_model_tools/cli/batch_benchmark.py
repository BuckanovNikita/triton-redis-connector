"""Run batch-size sweep benchmark using ultralytics YOLO.predict()."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    import argparse

    from triton_model_tools.cli._types import SubParsers

from triton_model_tools.cli._types import parse_image_size


def build_parser(subparsers: SubParsers) -> None:
    """Register the ``batch-benchmark`` subcommand."""
    bb_parser = subparsers.add_parser(
        "batch-benchmark",
        help="Run batch-size sweep benchmark using ultralytics YOLO.predict()",
    )
    bb_parser.add_argument(
        "--image-dir",
        type=Path,
        required=True,
        help="Directory containing images to benchmark with",
    )
    bb_parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="yolo11s.pt",
        help="Path or name of ultralytics model (default: yolo11s.pt)",
    )
    bb_parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=[1, 2, 4, 8, 16, 32],
        help="Batch sizes to test (default: 1 2 4 8 16 32)",
    )
    bb_parser.add_argument(
        "--num-images",
        type=int,
        default=100,
        help="Number of images to use (default: 100)",
    )
    bb_parser.add_argument(
        "--target-size",
        type=parse_image_size,
        default=640,
        help="Model input size: int or HxW (e.g. 640 or 960x600)",
    )
    bb_parser.add_argument(
        "--warmup-runs",
        type=int,
        default=5,
        help="Number of warmup iterations (default: 5)",
    )
    bb_parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output file path for results JSON",
    )
    bb_parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    """Execute the ``batch-benchmark`` subcommand."""
    from triton_model_tools.benchmarks.benchmark import (  # noqa: PLC0415
        BatchSizeBenchmarkConfig,
        run_batch_size_benchmark,
    )

    image_dir: Path = args.image_dir.resolve()
    model_path: str = args.model
    batch_sizes: list[int] = args.batch_sizes
    num_images: int = args.num_images
    target_size: int | tuple[int, int] = args.target_size
    warmup_runs: int = args.warmup_runs
    output_path: Path | None = args.output

    logger.info("Batch-Size Sweep Benchmark")
    logger.info("Image dir: {}", image_dir)
    logger.info("Model: {}", model_path)
    logger.info("Batch sizes: {}", batch_sizes)

    config = BatchSizeBenchmarkConfig(
        model_path=model_path,
        batch_sizes=batch_sizes,
        num_images=num_images,
        target_size=target_size,
        warmup_runs=warmup_runs,
    )

    result = run_batch_size_benchmark(config, image_dir)

    for bs_result in result.per_batch_size:
        logger.info(
            "batch_size={}: avg={:.2f}ms, throughput={:.1f}fps, p50={:.2f}ms, p95={:.2f}ms",
            bs_result.config.num_images,
            bs_result.avg_latency_ms,
            bs_result.throughput_fps,
            bs_result.p50_latency_ms,
            bs_result.p95_latency_ms,
        )

    if output_path is not None:
        import json  # noqa: PLC0415

        resolved_output = output_path.resolve()
        resolved_output.parent.mkdir(parents=True, exist_ok=True)
        with resolved_output.open("w") as f:
            json.dump(result.model_dump(), f, indent=2)
        logger.info("Batch benchmark results saved to {}", resolved_output)

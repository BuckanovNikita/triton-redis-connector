"""Run full performance matrix across models and pipeline configs."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    import argparse

    from triton_model_tools.cli._types import SubParsers

from triton_model_tools.cli._types import parse_image_size


def build_parser(subparsers: SubParsers) -> None:
    """Register the ``full-benchmark`` subcommand."""
    fb_parser = subparsers.add_parser(
        "full-benchmark",
        help="Run full performance matrix across models and pipeline configs",
    )
    fb_parser.add_argument(
        "--image-dir",
        type=Path,
        required=True,
        help="Directory containing images to benchmark with",
    )
    fb_parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Model names to benchmark (e.g. yolo11s yolo26s yolo12s)",
    )
    fb_parser.add_argument(
        "--repository-root",
        type=Path,
        default=None,
        help="Auto-discover models from Triton repository instead of --models",
    )
    fb_parser.add_argument(
        "--triton-url",
        type=str,
        default="localhost:8001",
        help="Triton gRPC URL (default: localhost:8001)",
    )
    fb_parser.add_argument(
        "--target-size",
        type=parse_image_size,
        default=640,
        help="Model input size: int or HxW (e.g. 640 or 960x600)",
    )
    fb_parser.add_argument(
        "--num-images",
        type=int,
        default=100,
        help="Number of images per benchmark run (default: 100)",
    )
    fb_parser.add_argument(
        "--warmup-runs",
        type=int,
        default=5,
        help="Number of warmup iterations per run (default: 5)",
    )
    fb_parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Output file path for results (JSON or CSV)",
    )
    fb_parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    """Execute the ``full-benchmark`` subcommand."""
    from triton_model_tools.benchmarks.benchmark import (  # noqa: PLC0415
        build_full_matrix,
        run_benchmark_matrix,
        save_benchmark_output,
    )
    from triton_model_tools.models import discover_models  # noqa: PLC0415

    image_dir: Path = args.image_dir.resolve()
    repository_root: Path | None = args.repository_root
    models: list[str] | None = args.models

    if models is None and repository_root is not None:
        discovered = discover_models(repository_root)
        models = [m.name for m in discovered]
        if not models:
            msg = f"No models discovered in {repository_root}"
            raise ValueError(msg)
        logger.info("Auto-discovered models: {}", ", ".join(models))

    if models is None:
        msg = "Either --models or --repository-root must be specified"
        raise ValueError(msg)

    triton_url: str = args.triton_url
    target_size: int | tuple[int, int] = args.target_size
    num_images: int = args.num_images
    warmup_runs: int = args.warmup_runs
    output_path: Path = args.output.resolve()

    logger.info("Full Performance Matrix Benchmark")
    logger.info("Image dir: {}", image_dir)
    logger.info("Models: {}", ", ".join(models))
    logger.info("Output: {}", output_path)

    configs = build_full_matrix(
        models=models,
        triton_url=triton_url,
        target_size=target_size,
        num_images=num_images,
        warmup_runs=warmup_runs,
    )

    suite = run_benchmark_matrix(configs, image_dir)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_benchmark_output(suite, output_path)

    logger.info("Full benchmark complete: {} configurations", len(suite.results))
    for result in suite.results:
        logger.info(
            "  {} ({}+{}): avg={:.2f}ms throughput={:.1f}fps",
            result.config.model_name,
            result.config.backend,
            result.config.preprocess_mode,
            result.avg_latency_ms,
            result.throughput_fps,
        )

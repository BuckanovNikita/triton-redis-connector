"""Run inference benchmark on images via Triton."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    import argparse

    from triton_model_tools.cli._types import SubParsers

from triton_model_tools.cli._types import parse_image_size


def build_parser(subparsers: SubParsers) -> None:
    """Register the ``benchmark`` subcommand."""
    bench_parser = subparsers.add_parser(
        "benchmark",
        help="Run inference benchmark on images via Triton",
    )
    bench_parser.add_argument(
        "--image-dir",
        type=Path,
        required=True,
        help="Directory containing images to benchmark with",
    )
    bench_parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
        help="Triton model name (e.g. yolo11s)",
    )
    bench_parser.add_argument(
        "--backend",
        choices=["onnx", "tensorrt"],
        default="onnx",
        help="Model backend (default: onnx)",
    )
    bench_parser.add_argument(
        "--triton-url",
        type=str,
        default="localhost:8001",
        help="Triton gRPC URL (default: localhost:8001)",
    )
    bench_parser.add_argument(
        "--target-size",
        type=parse_image_size,
        default=640,
        help="Model input size: int or HxW (e.g. 640 or 960x600)",
    )
    bench_parser.add_argument(
        "--num-images",
        type=int,
        default=100,
        help="Number of images to benchmark (default: 100)",
    )
    bench_parser.add_argument(
        "--warmup-runs",
        type=int,
        default=5,
        help="Number of warmup iterations (default: 5)",
    )
    bench_parser.add_argument(
        "--repository-root",
        type=Path,
        default=None,
        help="Triton model repository root for auto-detecting model properties",
    )
    bench_parser.add_argument(
        "--end2end",
        action="store_true",
        default=False,
        help="Model uses end2end NMS output format",
    )
    bench_parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output file path (JSON or CSV based on extension)",
    )
    bench_parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    """Execute the ``benchmark`` subcommand."""
    from triton_model_tools.benchmarks.benchmark import (  # noqa: PLC0415
        BenchmarkConfig,
        BenchmarkSuite,
        run_benchmark,
        save_benchmark_output,
    )
    from triton_model_tools.models import discover_models  # noqa: PLC0415

    image_dir: Path = args.image_dir.resolve()
    model_name: str = args.model
    backend: str = args.backend
    triton_url: str = args.triton_url
    target_size: int | tuple[int, int] = args.target_size
    num_images: int = args.num_images
    warmup_runs: int = args.warmup_runs
    end2end: bool = args.end2end
    output_path: Path | None = args.output
    repository_root: Path | None = args.repository_root

    if repository_root is not None:
        discovered = discover_models(repository_root)
        match = next((m for m in discovered if m.name == model_name), None)
        if match is not None:
            end2end = match.is_end2end
            if match.input_dims and len(match.input_dims) >= 4:
                h, w = match.input_dims[-2], match.input_dims[-1]
                target_size = h if h == w else (h, w)
            logger.info("Auto-detected from repository: end2end={}, target_size={}", end2end, target_size)

    logger.info("Triton Benchmark")
    logger.info("Image dir: {}", image_dir)
    logger.info("Model: {} (backend: {})", model_name, backend)

    config = BenchmarkConfig(
        model_name=model_name,
        backend=backend,  # type: ignore[arg-type]
        triton_url=triton_url,
        target_size=target_size,
        num_images=num_images,
        warmup_runs=warmup_runs,
        end2end=end2end,
    )

    result = run_benchmark(config, image_dir)
    suite = BenchmarkSuite(results=[result])

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
        save_benchmark_output(suite, output_path.resolve())

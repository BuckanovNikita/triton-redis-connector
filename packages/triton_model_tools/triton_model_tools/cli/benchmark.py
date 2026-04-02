"""Run inference benchmark on images via Triton or E2E via Redis pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    import argparse

    from triton_model_tools.cli._types import SubParsers

from triton_model_tools.cli._types import parse_image_size

MIN_DIMS_FOR_SIZE = 4


def build_parser(subparsers: SubParsers) -> None:
    """Register the ``benchmark`` subcommand."""
    bench_parser = subparsers.add_parser(
        "benchmark",
        help="Run inference benchmark on images via Triton or E2E via Redis pipeline",
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

    pipeline_group = bench_parser.add_argument_group("pipeline benchmark options")
    pipeline_group.add_argument(
        "--pipeline",
        action="store_true",
        default=False,
        help="Run E2E benchmark via Redis pipeline instead of direct Triton inference",
    )
    pipeline_group.add_argument(
        "--redis-url",
        type=str,
        default="redis://redis:6379",
        help="Redis URL for pipeline mode (default: redis://redis:6379)",
    )
    pipeline_group.add_argument(
        "--input-stream",
        type=str,
        default="camera_images",
        help="Redis input stream name for pipeline mode (default: camera_images)",
    )
    pipeline_group.add_argument(
        "--output-stream",
        type=str,
        default="detection_results",
        help="Redis output stream name for pipeline mode (default: detection_results)",
    )
    pipeline_group.add_argument(
        "--num-iterations",
        type=int,
        default=3,
        help="Number of benchmark iterations for pipeline mode (default: 3)",
    )
    pipeline_group.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Timeout in seconds for collecting results per iteration (default: 60.0)",
    )
    pipeline_group.add_argument(
        "--full-matrix",
        action="store_true",
        default=False,
        help="Run full benchmark matrix across backends and preprocess modes (pipeline only)",
    )
    pipeline_group.add_argument(
        "--preprocess-mode",
        nargs="+",
        default=None,
        help="Preprocess modes for matrix mode (default: dali cv2)",
    )
    pipeline_group.add_argument(
        "--models",
        type=str,
        action="append",
        default=None,
        help="Additional model names for matrix mode (repeatable)",
    )
    pipeline_group.add_argument(
        "--all",
        action="store_true",
        default=False,
        dest="all_models",
        help="Discover all models from --repository-root for matrix mode",
    )
    bench_parser.set_defaults(func=run)


def _collect_matrix_models(args: argparse.Namespace) -> list[str]:
    from triton_model_tools.models import discover_models  # noqa: PLC0415

    model_names: list[str] = [args.model]

    extra_models: list[str] | None = args.models
    if extra_models:
        model_names.extend(extra_models)

    all_models: bool = args.all_models
    repository_root: Path | None = args.repository_root

    if all_models:
        if repository_root is None:
            msg = "--repository-root is required when using --all"
            raise ValueError(msg)
        discovered = discover_models(repository_root.resolve())
        discovered_names = [m.name for m in discovered]
        model_names.extend(n for n in discovered_names if n not in model_names)

    return model_names


def _save_results_json(results: object, output_path: Path) -> None:
    import json  # noqa: PLC0415

    resolved_output = output_path.resolve()
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    with resolved_output.open("w") as f:
        json.dump(results, f, indent=2)
    logger.info("Pipeline benchmark results saved to {}", resolved_output)


def _run_pipeline_benchmark(args: argparse.Namespace) -> None:
    from triton_model_tools.benchmarks.pipeline_benchmark import (  # noqa: PLC0415
        PipelineBenchmarkConfig,
        build_pipeline_matrix,
        run_pipeline_benchmark,
        run_pipeline_benchmark_matrix,
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
    full_matrix: bool = args.full_matrix
    backends: list[str] | None = [args.backend] if args.backend else None
    preprocess_modes: list[str] | None = args.preprocess_mode
    all_models: bool = args.all_models

    logger.info("Pipeline E2E Benchmark")
    logger.info("Image dir: {}", image_dir)
    logger.info("Model: {}", model_name)
    logger.info("Redis: {}", redis_url)

    if full_matrix or all_models:
        matrix_models = _collect_matrix_models(args) if all_models else [model_name]
        configs = build_pipeline_matrix(
            models=matrix_models,
            backends=backends,
            preprocess_modes=preprocess_modes,
            redis_url=redis_url,
            input_stream=input_stream,
            output_stream=output_stream,
            num_iterations=num_iterations,
            warmup_runs=warmup_runs,
            timeout_s=timeout_s,
        )
        results = run_pipeline_benchmark_matrix(configs, image_dir)

        for result in results:
            logger.info(
                "{}: avg={:.2f}ms, throughput={:.1f}fps, p50={:.2f}ms",
                result.config.model_name,
                result.avg_latency_ms,
                result.throughput_fps,
                result.p50_latency_ms,
            )

        if output_path is not None:
            _save_results_json([r.model_dump() for r in results], output_path)
        return

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
        _save_results_json(result.model_dump(), output_path)


def _run_triton_benchmark(args: argparse.Namespace) -> None:
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
            if match.input_dims and len(match.input_dims) >= MIN_DIMS_FOR_SIZE:
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


def run(args: argparse.Namespace) -> None:
    """Execute the ``benchmark`` subcommand."""
    pipeline: bool = args.pipeline
    if pipeline:
        _run_pipeline_benchmark(args)
    else:
        _run_triton_benchmark(args)

"""Run Triton perf_analyzer on one or more models."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    import argparse

    from triton_model_tools.cli._types import SubParsers


def build_parser(subparsers: SubParsers) -> None:
    """Register the ``perf-analyze`` subcommand."""
    pa_parser = subparsers.add_parser(
        "perf-analyze",
        help="Run Triton perf_analyzer on one or more models",
    )
    pa_parser.add_argument(
        "-m",
        "--model",
        type=str,
        action="append",
        default=None,
        help="Model name to benchmark (repeatable, e.g. -m yolo11s -m yolo11m)",
    )
    pa_parser.add_argument(
        "--all",
        action="store_true",
        default=False,
        dest="all_models",
        help="Discover and benchmark all models from --repository-root",
    )
    pa_parser.add_argument(
        "--repository-root",
        type=Path,
        default=None,
        help="Path to Triton model repository (required with --all)",
    )
    pa_parser.add_argument(
        "--concurrency",
        type=str,
        default="1,4,8",
        help="Comma-separated concurrency levels (default: 1,4,8)",
    )
    pa_parser.add_argument(
        "--url",
        type=str,
        default="localhost:8001",
        help="Triton server URL (default: localhost:8001)",
    )
    pa_parser.add_argument(
        "--protocol",
        choices=["grpc", "http"],
        default="grpc",
        help="Communication protocol (default: grpc)",
    )
    pa_parser.add_argument(
        "--measurement-interval",
        type=int,
        default=5000,
        help="Measurement interval in ms (default: 5000)",
    )
    pa_parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output JSON file path for results",
    )
    pa_parser.set_defaults(func=run)


def _collect_model_names(args: argparse.Namespace) -> list[str]:
    from triton_model_tools.models import discover_models  # noqa: PLC0415

    model_names: list[str] = args.model or []
    all_models: bool = args.all_models
    repository_root: Path | None = args.repository_root

    if all_models:
        if repository_root is None:
            logger.error("--repository-root is required when using --all")
            sys.exit(1)
        discovered = discover_models(repository_root.resolve())
        model_names.extend(m.name for m in discovered)

    if not model_names:
        logger.error("No models specified. Use -m MODEL or --all with --repository-root")
        sys.exit(1)

    return model_names


def _parse_concurrency_levels(concurrency_str: str) -> list[int]:
    return [int(c.strip()) for c in concurrency_str.split(",") if c.strip()]


def run(args: argparse.Namespace) -> None:
    """Execute the ``perf-analyze`` subcommand."""
    from triton_model_tools.benchmarks.perf_analyzer import (  # noqa: PLC0415
        PerfAnalyzerConfig,
        is_perf_analyzer_available,
        run_perf_analyzer,
    )

    if not is_perf_analyzer_available():
        logger.error(
            "perf_analyzer binary not found on PATH. "
            "Install the Triton SDK or run inside the Triton SDK container "
            "(nvcr.io/nvidia/tritonserver:*-py3-sdk)."
        )
        sys.exit(1)

    model_names = _collect_model_names(args)
    concurrency_levels = _parse_concurrency_levels(args.concurrency)
    url: str = args.url
    protocol: str = args.protocol
    measurement_interval: int = args.measurement_interval
    output_path: Path | None = args.output

    logger.info("Perf Analyzer Benchmark")
    logger.info("Models: {}", model_names)
    logger.info("Concurrency levels: {}", concurrency_levels)
    logger.info("Protocol: {} URL: {}", protocol, url)

    all_results: list[dict[str, object]] = []
    for model_name in model_names:
        for concurrency in concurrency_levels:
            config = PerfAnalyzerConfig(
                model_name=model_name,
                protocol=protocol,  # type: ignore[arg-type]
                url=url,
                concurrency=concurrency,
                measurement_interval_ms=measurement_interval,
            )

            logger.info("Running perf_analyzer: model={} concurrency={}", model_name, concurrency)
            result = run_perf_analyzer(config)

            logger.info(
                "  p50={:.2f}ms p95={:.2f}ms p99={:.2f}ms throughput={:.1f} infer/s",
                result.p50_latency_ms,
                result.p95_latency_ms,
                result.p99_latency_ms,
                result.throughput_infer_per_sec,
            )

            all_results.append(
                {
                    "model_name": model_name,
                    **result.model_dump(),
                }
            )

    if output_path is not None:
        import json  # noqa: PLC0415

        resolved_output = output_path.resolve()
        resolved_output.parent.mkdir(parents=True, exist_ok=True)
        with resolved_output.open("w") as f:
            json.dump(all_results, f, indent=2)
        logger.info("Results saved to {}", resolved_output)

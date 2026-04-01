"""CLI entry point for triton-model-tools: subcommand-based model management."""

from __future__ import annotations

import argparse
import sys

from loguru import logger

from triton_model_tools.cli import (
    benchmark,
    compare_folders,
    compare_visual,
    convert,
    download_coco,
    e2e_test,
    infer,
    local_infer,
    pipeline_benchmark,
    pipeline_predict,
    populate_stream,
    predict,
    validate,
)


def main() -> None:
    """Entry point for triton-model-tools CLI."""
    parser = argparse.ArgumentParser(
        prog="triton-model-tools",
        description="Tools for managing YOLO models on NVIDIA Triton Inference Server",
    )
    subparsers = parser.add_subparsers(dest="command")

    convert.build_parser(subparsers)
    infer.build_parser(subparsers)
    local_infer.build_parser(subparsers)
    download_coco.build_parser(subparsers)
    benchmark.build_parser(subparsers)
    e2e_test.build_parser(subparsers)
    predict.build_parser(subparsers)
    pipeline_predict.build_parser(subparsers)
    compare_folders.build_parser(subparsers)
    compare_visual.build_parser(subparsers)
    pipeline_benchmark.build_parser(subparsers)
    populate_stream.build_parser(subparsers)
    validate.build_parser(subparsers)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    try:
        args.func(args)
    except KeyboardInterrupt:
        logger.warning("Operation interrupted by user")
        sys.exit(130)
    except ValueError as exc:
        logger.error("{}", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()

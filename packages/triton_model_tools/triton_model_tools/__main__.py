"""CLI entry point for triton-model-tools: subcommand-based model management."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loguru import logger

from triton_model_tools.models import YOLO_MODELS

AVAILABLE_MODEL_NAMES = ", ".join(v.name for v in YOLO_MODELS)


def _build_export_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],  # pyright: ignore[reportPrivateUsage]
) -> None:
    export_parser = subparsers.add_parser(
        "export",
        help="Export YOLO models for NVIDIA Triton Inference Server",
    )
    export_parser.add_argument(
        "-r",
        "--repository-root",
        type=Path,
        required=True,
        help="Path to Triton model repository",
    )
    export_parser.add_argument(
        "-m",
        "--models",
        nargs="+",
        default=None,
        help="Model names to export (default: all). Choices: " + AVAILABLE_MODEL_NAMES,
    )
    export_parser.add_argument(
        "-f",
        "--format",
        choices=["onnx", "tensorrt", "all"],
        default="onnx",
        dest="export_format",
        help="Export format (default: onnx)",
    )
    export_parser.add_argument(
        "--dali-ensemble",
        "--dali",
        action="store_true",
        default=False,
        help="Also generate DALI preprocess + ensemble models",
    )
    export_parser.add_argument(
        "--nms",
        action="store_true",
        default=False,
        help="Also generate NMS postprocess model for non-end2end models",
    )
    export_parser.set_defaults(func=_run_export)


def _run_export(args: argparse.Namespace) -> None:
    from triton_model_tools.exporter import export_all_models  # noqa: PLC0415

    repository_root: Path = args.repository_root.resolve()
    model_names: list[str] | None = args.models
    export_format: str = args.export_format
    generate_dali: bool = args.dali_ensemble
    generate_nms: bool = args.nms

    logger.info("Triton Model Export Tool")
    logger.info("Repository root: {}", repository_root)
    logger.info("Export format: {}", export_format)

    if model_names:
        logger.info("Selected models: {}", ", ".join(model_names))
    else:
        logger.info("Exporting all models")

    if generate_dali:
        logger.info("DALI ensemble generation enabled")

    if generate_nms:
        logger.info("NMS postprocess generation enabled")

    exported_dirs = export_all_models(
        repository_root=repository_root,
        model_names=model_names,
        fmt=export_format,  # type: ignore[arg-type]
    )

    if generate_dali:
        from triton_model_tools.dali import (  # noqa: PLC0415
            DaliPipelineConfig,
            build_ensemble_directory,
        )

        dali_config = DaliPipelineConfig()
        for model_dir in exported_dirs:
            build_ensemble_directory(
                repository_root=repository_root,
                model_name=model_dir.name,
                config=dali_config,
            )

    if generate_nms:
        from triton_model_tools.nms import (  # noqa: PLC0415
            NmsModelConfig,
            build_nms_model_directory,
        )

        build_nms_model_directory(
            repository_root=repository_root,
            config=NmsModelConfig(),
        )


def main() -> None:
    """Entry point for triton-model-tools CLI."""
    parser = argparse.ArgumentParser(
        prog="triton-model-tools",
        description="Tools for managing YOLO models on NVIDIA Triton Inference Server",
    )
    subparsers = parser.add_subparsers(dest="command")

    _build_export_parser(subparsers)

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

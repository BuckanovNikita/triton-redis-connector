"""CLI entry point for triton-model-tools: export YOLO models for Triton."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loguru import logger

from triton_model_tools.exporter import export_all_models
from triton_model_tools.models import YOLO_MODELS


def main() -> None:
    """Export Ultralytics YOLO models to ONNX and build Triton model repository."""
    parser = argparse.ArgumentParser(
        description="Export YOLO models for NVIDIA Triton Inference Server",
    )
    parser.add_argument(
        "-r",
        "--repository-root",
        type=Path,
        default=Path("model_repository"),
        help="Path to Triton model repository (default: ./model_repository)",
    )
    parser.add_argument(
        "-m",
        "--models",
        nargs="+",
        default=None,
        help="Model names to export (default: all). Choices: " + ", ".join(v.name for v in YOLO_MODELS),
    )

    args = parser.parse_args()

    logger.info("Triton Model Export Tool")
    logger.info("Repository root: {}", args.repository_root)
    if args.models:
        logger.info("Selected models: {}", ", ".join(args.models))
    else:
        logger.info("Exporting all models")

    try:
        export_all_models(
            repository_root=args.repository_root.resolve(),
            model_names=args.models,
        )
    except KeyboardInterrupt:
        logger.warning("Export interrupted by user")
        sys.exit(130)
    except ValueError as exc:
        logger.error("{}", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()

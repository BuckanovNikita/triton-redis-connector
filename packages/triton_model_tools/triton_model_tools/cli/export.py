"""Export YOLO models for NVIDIA Triton Inference Server."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from triton_model_tools.models import YOLO_MODELS

if TYPE_CHECKING:
    import argparse

    from triton_model_tools.cli._types import SubParsers

AVAILABLE_MODEL_NAMES = ", ".join(v.name for v in YOLO_MODELS)


def build_parser(subparsers: SubParsers) -> None:
    """Register the ``export`` subcommand."""
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
        "--models-config",
        type=Path,
        default=None,
        help="Path to YAML file defining model variants (default: built-in models)",
    )
    export_parser.add_argument(
        "--half",
        "--fp16",
        action="store_true",
        default=False,
        help="Export models in FP16 precision (default: FP32 for ONNX, FP16 for TensorRT)",
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
    export_parser.add_argument(
        "--rescale",
        action="store_true",
        default=False,
        help="Also generate rescale postprocess model to map detections back to original image coordinates",
    )
    export_parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    """Execute the ``export`` subcommand."""
    from triton_model_tools.export.exporter import export_all_models  # noqa: PLC0415
    from triton_model_tools.models import YOLO_MODELS, load_export_config  # noqa: PLC0415

    repository_root: Path = args.repository_root.resolve()
    model_names: list[str] | None = args.models
    export_format: str = args.export_format
    half: bool = args.half
    generate_dali: bool = args.dali_ensemble
    generate_nms: bool = args.nms
    generate_rescale: bool = args.rescale
    models_config: Path | None = args.models_config

    logger.info("Triton Model Export Tool")
    logger.info("Repository root: {}", repository_root)
    logger.info("Export format: {}", export_format)
    logger.info("FP16 (half): {}", half)

    if models_config is not None:
        logger.info("Models config: {}", models_config)

    if model_names:
        logger.info("Selected models: {}", ", ".join(model_names))
    else:
        logger.info("Exporting all models")

    if generate_dali:
        logger.info("DALI ensemble generation enabled")

    if generate_nms:
        logger.info("NMS postprocess generation enabled")

    if generate_rescale:
        logger.info("Rescale postprocess generation enabled")
    else:
        logger.warning("No rescale postprocessing configured. Detection coordinates will be in letterbox space.")

    exported_dirs = export_all_models(
        repository_root=repository_root,
        model_names=model_names,
        fmt=export_format,  # type: ignore[arg-type]
        models_config=models_config,
        half=half,
    )

    nms_model_name: str | None = None
    if generate_nms:
        from triton_model_tools.export.nms import (  # noqa: PLC0415
            NMS_MODEL_NAME,
            NmsModelConfig,
            build_nms_model_directory,
        )

        build_nms_model_directory(
            repository_root=repository_root,
            config=NmsModelConfig(),
        )
        nms_model_name = NMS_MODEL_NAME

    rescale_model_name: str | None = None
    if generate_rescale:
        from triton_model_tools.export.rescale import (  # noqa: PLC0415
            RESCALE_MODEL_NAME,
            build_rescale_model_directory,
        )

        build_rescale_model_directory(repository_root=repository_root)
        rescale_model_name = RESCALE_MODEL_NAME

    if generate_dali:
        from triton_model_tools.export.dali import (  # noqa: PLC0415
            DaliPipelineConfig,
            build_ensemble_directory,
        )

        variants = load_export_config(models_config) if models_config is not None else YOLO_MODELS
        end2end_names = {v.name for v in variants if v.end2end_export}
        dali_config = DaliPipelineConfig()
        for model_dir in exported_dirs:
            effective_nms = None if model_dir.name in end2end_names else nms_model_name
            build_ensemble_directory(
                repository_root=repository_root,
                model_name=model_dir.name,
                config=dali_config,
                nms_model_name=effective_nms,
                rescale_model_name=rescale_model_name,
            )

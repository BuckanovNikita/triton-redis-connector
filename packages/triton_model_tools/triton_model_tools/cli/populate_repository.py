"""Export all models, generate DALI/NMS/rescale, and validate the repository."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    import argparse

    from triton_model_tools.cli._types import SubParsers


def build_parser(subparsers: SubParsers) -> None:
    """Register the ``populate-repository`` subcommand."""
    pr_parser = subparsers.add_parser(
        "populate-repository",
        help="Export all models, generate DALI/NMS/rescale, and validate",
    )
    pr_parser.add_argument(
        "-r",
        "--repository-root",
        type=Path,
        required=True,
        help="Path to Triton model repository",
    )
    pr_parser.add_argument(
        "-f",
        "--format",
        choices=["onnx", "tensorrt", "all"],
        default="onnx",
        dest="export_format",
        help="Export format (default: onnx)",
    )
    pr_parser.add_argument(
        "--models-config",
        type=Path,
        default=None,
        help="Path to YAML file defining model variants",
    )
    pr_parser.add_argument(
        "--half",
        "--fp16",
        action="store_true",
        default=False,
        help="Export models in FP16 precision",
    )
    pr_parser.add_argument(
        "--dali",
        action="store_true",
        default=False,
        help="Generate DALI preprocess + ensemble models",
    )
    pr_parser.add_argument(
        "--nms",
        action="store_true",
        default=False,
        help="Generate NMS postprocess model",
    )
    pr_parser.add_argument(
        "--rescale",
        action="store_true",
        default=False,
        help="Generate rescale postprocess model",
    )
    pr_parser.add_argument(
        "--validate-only",
        action="store_true",
        default=False,
        help="Only validate existing repository without exporting",
    )
    pr_parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    """Execute the ``populate-repository`` subcommand."""
    from triton_model_tools.export.validate import validate_repository  # noqa: PLC0415

    repository_root: Path = args.repository_root.resolve()
    validate_only: bool = args.validate_only

    if validate_only:
        logger.info("Validating repository: {}", repository_root)
        results = validate_repository(repository_root)
        _report_validation(results)
        return

    from triton_model_tools.export.exporter import export_all_models  # noqa: PLC0415
    from triton_model_tools.models import YOLO_MODELS, load_export_config  # noqa: PLC0415

    export_format: str = args.export_format
    half: bool = args.half
    generate_dali: bool = args.dali
    generate_nms: bool = args.nms
    generate_rescale: bool = args.rescale
    models_config: Path | None = args.models_config

    logger.info("Populating repository: {}", repository_root)
    logger.info("Format: {}, FP16: {}", export_format, half)

    exported_dirs = export_all_models(
        repository_root=repository_root,
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

        build_nms_model_directory(repository_root=repository_root, config=NmsModelConfig())
        nms_model_name = NMS_MODEL_NAME
        logger.info("Generated NMS model: {}", nms_model_name)

    rescale_model_name: str | None = None
    if generate_rescale:
        from triton_model_tools.export.rescale import (  # noqa: PLC0415
            RESCALE_MODEL_NAME,
            build_rescale_model_directory,
        )

        build_rescale_model_directory(repository_root=repository_root)
        rescale_model_name = RESCALE_MODEL_NAME
        logger.info("Generated rescale model: {}", rescale_model_name)

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
        logger.info("Generated DALI ensembles for {} models", len(exported_dirs))

    logger.info("Validating repository after population")
    results = validate_repository(repository_root)
    _report_validation(results)


def _report_validation(results: dict[str, list[str]]) -> None:
    """Log validation results summary."""
    has_errors = False
    for errors in results.values():
        if errors:
            has_errors = True
            for error in errors:
                logger.error("{}", error)

    if has_errors:
        logger.error("Repository validation failed")
    else:
        logger.info("Repository validation passed: all {} models valid", len(results))

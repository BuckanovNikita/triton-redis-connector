"""Convert YOLO .pt weights to Triton format with optional local verification."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from triton_model_tools.cli._types import parse_image_size

if TYPE_CHECKING:
    import argparse

    from triton_model_tools.cli._types import SubParsers


def build_parser(subparsers: SubParsers) -> None:
    """Register the ``convert`` subcommand."""
    convert_parser = subparsers.add_parser(
        "convert",
        help="Convert YOLO .pt model to Triton format with optional local verification",
    )
    convert_parser.add_argument(
        "-w",
        "--weights",
        type=Path,
        required=True,
        help="Path to YOLO .pt weights file",
    )
    convert_parser.add_argument(
        "-r",
        "--repository-root",
        type=Path,
        required=True,
        help="Path to Triton model repository",
    )
    convert_parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Model name in Triton repository (default: derived from weights filename)",
    )
    convert_parser.add_argument(
        "--imgsz",
        "--target-size",
        type=parse_image_size,
        default=640,
        help="Input size: int or HxW (e.g. 640 or 960x608). Default: 640",
    )
    convert_parser.add_argument(
        "--no-rect",
        action="store_true",
        default=False,
        help="Disable rectangular inference (force square input). Default: rect=True",
    )
    convert_parser.add_argument(
        "--end2end",
        action="store_true",
        default=False,
        help="Export with end-to-end NMS",
    )
    convert_parser.add_argument(
        "-f",
        "--format",
        choices=["onnx", "tensorrt", "all"],
        default="onnx",
        dest="export_format",
        help="Export format (default: onnx)",
    )
    convert_parser.add_argument(
        "--half",
        "--fp16",
        action="store_true",
        default=False,
        help="Export models in FP16 precision (default: FP32 for ONNX, FP16 for TensorRT)",
    )
    convert_parser.add_argument(
        "--no-dynamic-batching",
        action="store_true",
        default=False,
        help="Disable Triton dynamic batching (enabled by default)",
    )
    convert_parser.add_argument(
        "--max-batch-size",
        type=int,
        default=8,
        help="Maximum batch size for dynamic batching (default: 8)",
    )
    convert_parser.add_argument(
        "--preferred-batch-sizes",
        type=str,
        default=None,
        help="Comma-separated preferred batch sizes for dynamic batching (default: 1,2,4,8)",
    )
    convert_parser.add_argument(
        "--dali-ensemble",
        "--dali",
        action="store_true",
        default=False,
        help="Also generate DALI preprocess + ensemble models",
    )
    convert_parser.add_argument(
        "--nms",
        action="store_true",
        default=False,
        help="Also generate NMS postprocess model for non-end2end models",
    )
    convert_parser.add_argument(
        "--rescale",
        action="store_true",
        default=False,
        help="Also generate rescale postprocess model",
    )
    convert_parser.add_argument(
        "--trt-optimize",
        action="store_true",
        default=False,
        help="Enable TensorRT execution accelerator for ONNX models in Triton config",
    )
    convert_parser.add_argument(
        "--image-dir",
        type=Path,
        default=None,
        help="Image folder for local ultralytics predict verification",
    )
    convert_parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save predict txt results (default: {repository}/.verify/{name})",
    )
    convert_parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.25,
        help="Confidence threshold for predict verification (default: 0.25)",
    )
    convert_parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.45,
        help="IoU threshold for predict verification (default: 0.45)",
    )
    convert_parser.set_defaults(func=run)


def _generate_postprocess_models(  # noqa: PLR0913
    repository_root: Path,
    exported_dirs: list[Path],
    *,
    generate_nms: bool,
    generate_rescale: bool,
    generate_dali: bool,
    end2end: bool,
    target_size: tuple[int, int],
) -> None:
    """Generate NMS, rescale, and DALI ensemble models after export."""
    nms_model_name: str | None = None
    if generate_nms:
        from triton_model_tools.export.nms import (  # noqa: PLC0415
            NMS_MODEL_NAME,
            NmsModelConfig,
            build_nms_model_directory,
        )

        build_nms_model_directory(repository_root=repository_root, config=NmsModelConfig())
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

        dali_config = DaliPipelineConfig(target_size=target_size)
        for model_dir in exported_dirs:
            effective_nms = None if end2end else nms_model_name
            build_ensemble_directory(
                repository_root=repository_root,
                model_name=model_dir.name,
                config=dali_config,
                nms_model_name=effective_nms,
                rescale_model_name=rescale_model_name,
            )


def run(args: argparse.Namespace) -> None:
    """Execute the ``convert`` subcommand."""
    from triton_model_tools.export.converter import convert_model  # noqa: PLC0415

    weights_path: Path = args.weights.resolve()
    repository_root: Path = args.repository_root.resolve()
    imgsz: int | tuple[int, int] = args.imgsz
    rect: bool = not args.no_rect
    end2end: bool = args.end2end
    model_name = args.name if args.name is not None else weights_path.stem

    logger.info(
        "Converting {} -> {} (imgsz={}, rect={}, format={})",
        weights_path.name,
        model_name,
        imgsz,
        rect,
        args.export_format,
    )

    preferred_batch_size: list[int] | None = None
    if args.preferred_batch_sizes is not None:
        preferred_batch_size = [int(x.strip()) for x in args.preferred_batch_sizes.split(",")]

    exported_dirs = convert_model(
        weights_path=weights_path,
        repository_root=repository_root,
        name=args.name,
        imgsz=imgsz,
        rect=rect,
        fmt=args.export_format,
        half=args.half,
        end2end=end2end,
        enable_dynamic_batching=not args.no_dynamic_batching,
        max_batch_size=args.max_batch_size,
        preferred_batch_size=preferred_batch_size,
        trt_acceleration=args.trt_optimize,
    )

    resolved_imgsz = (imgsz, imgsz) if isinstance(imgsz, int) else imgsz
    _generate_postprocess_models(
        repository_root,
        exported_dirs,
        generate_nms=args.nms,
        generate_rescale=args.rescale,
        generate_dali=args.dali_ensemble,
        end2end=end2end,
        target_size=resolved_imgsz,
    )

    if args.image_dir is not None:
        from triton_model_tools.inference.predict import PredictConfig, run_ultralytics_predict  # noqa: PLC0415

        output_dir: Path = (
            args.output_dir.resolve() if args.output_dir is not None else repository_root / ".verify" / model_name
        )

        logger.info("Running local predict verification: {} -> {}", args.image_dir, output_dir)

        result = run_ultralytics_predict(
            args.image_dir.resolve(),
            output_dir,
            PredictConfig(
                model_path=str(weights_path),
                target_size=imgsz,
                conf_threshold=args.conf_threshold,
                iou_threshold=args.iou_threshold,
            ),
        )

        logger.info(
            "Predict verification: {} images, {:.2f}s total, {:.1f}ms avg",
            len(result.results),
            result.total_time_s,
            result.avg_time_ms,
        )

    logger.info("Convert complete: {} model director(ies) in {}", len(exported_dirs), repository_root)

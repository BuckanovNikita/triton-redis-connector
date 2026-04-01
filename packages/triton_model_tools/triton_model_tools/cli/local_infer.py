"""Run local ONNX inference on a folder of images without Triton."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    import argparse

    from triton_model_tools.cli._types import SubParsers

from triton_model_tools.cli._types import parse_image_size


def build_parser(subparsers: SubParsers) -> None:
    """Register the ``local-infer`` subcommand."""
    parser = subparsers.add_parser(
        "local-infer",
        help="Run YOLO inference on images using a local ONNX model (no Triton required)",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to an ONNX model file",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        required=True,
        help="Directory containing images to run inference on",
    )
    parser.add_argument(
        "--imgsz",
        type=parse_image_size,
        default=640,
        help="Model input size: int or HxW (e.g. 640 or 960x600)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="Confidence threshold (default: 0.25)",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="IoU threshold for NMS (default: 0.45)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save detection txt files (optional)",
    )
    parser.add_argument(
        "--end2end",
        action="store_true",
        default=False,
        help="Force end2end NMS output format (auto-detected if not set)",
    )
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    """Execute the ``local-infer`` subcommand."""
    from triton_model_tools.inference.local_folder_inference import run_local_folder_inference  # noqa: PLC0415

    model_path: Path = args.model_path.resolve()
    image_dir: Path = args.image_dir.resolve()
    target_size: int | tuple[int, int] = args.imgsz
    conf_threshold: float = args.confidence
    iou_threshold: float = args.iou
    output_dir: Path | None = args.output_dir
    force_end2end: bool = args.end2end

    logger.info("ONNX Local Inference")
    logger.info("Model: {}", model_path)
    logger.info("Image dir: {}", image_dir)
    logger.info("Input size: {}", target_size)

    result = run_local_folder_inference(
        model_path=model_path,
        image_dir=image_dir,
        target_size=target_size,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        end2end=force_end2end,
    )

    logger.info("Results: {} images processed", len(result.results))
    logger.info("Total time: {:.2f}s", result.total_time_s)
    logger.info("Average time: {:.1f}ms per image", result.avg_time_ms)

    total_detections = sum(len(r.detections) for r in result.results)
    logger.info("Total detections: {}", total_detections)

    if output_dir is not None:
        from triton_model_tools.inference.detection_io import save_detections_to_txt  # noqa: PLC0415

        resolved_output_dir = output_dir.resolve()
        resolved_output_dir.mkdir(parents=True, exist_ok=True)
        for img_result in result.results:
            stem = Path(img_result.image_path).stem
            save_detections_to_txt(img_result.detections, resolved_output_dir / f"{stem}.txt")
        logger.info("Saved detection txt files to {}", resolved_output_dir)

    for img_result in result.results:
        logger.info(
            "  {}: {} detections ({:.1f}ms)",
            img_result.image_path,
            len(img_result.detections),
            img_result.inference_time_ms,
        )
        for det in img_result.detections:
            logger.info(
                "    class={} conf={:.3f} bbox=({:.0f},{:.0f},{:.0f},{:.0f})",
                det.class_id,
                det.confidence,
                det.x1,
                det.y1,
                det.x2,
                det.y2,
            )

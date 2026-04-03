"""Run YOLO inference on a folder of images via Triton or local ONNX."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    import argparse

    from triton_model_tools.cli._types import SubParsers

from triton_model_tools.cli._types import parse_image_size


def build_parser(subparsers: SubParsers) -> None:
    """Register the ``infer`` subcommand."""
    infer_parser = subparsers.add_parser(
        "infer",
        help="Run YOLO inference on a folder of images via Triton or local ONNX",
    )
    infer_parser.add_argument(
        "--image-dir",
        type=Path,
        required=True,
        help="Directory containing images to run inference on",
    )
    infer_parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
        help="Triton model name (e.g. yolo11s) or ONNX model path with --local",
    )
    infer_parser.add_argument(
        "--local",
        action="store_true",
        default=False,
        help="Use local ONNX runtime instead of Triton (--model is path to .onnx file)",
    )
    infer_parser.add_argument(
        "--triton-url",
        type=str,
        default="localhost:8001",
        help="Triton gRPC URL (default: localhost:8001)",
    )
    infer_parser.add_argument(
        "--target-size",
        type=parse_image_size,
        default=640,
        help="Model input size: int or HxW (e.g. 640 or 960x600)",
    )
    infer_parser.add_argument(
        "--repository-root",
        type=Path,
        default=None,
        help="Triton model repository root for auto-detecting model properties",
    )
    infer_parser.add_argument(
        "--end2end",
        action="store_true",
        default=False,
        help="Model uses end2end NMS output format",
    )
    infer_parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.25,
        help="Confidence threshold (default: 0.25)",
    )
    infer_parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.45,
        help="IoU threshold for NMS (default: 0.45)",
    )
    infer_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("e2e_results/infer"),
        help="Directory to save detection txt files (default: e2e_results/infer/)",
    )
    infer_parser.set_defaults(func=run)


def _run_local(args: argparse.Namespace) -> None:
    from triton_model_tools.inference.local_folder_inference import run_local_folder_inference

    model_path = Path(args.model).resolve()
    image_dir: Path = args.image_dir.resolve()
    target_size: int | tuple[int, int] = args.target_size
    conf_threshold: float = args.conf_threshold
    iou_threshold: float = args.iou_threshold
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
        from triton_model_tools.inference.detection_io import save_detections_to_txt

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


def _run_triton(args: argparse.Namespace) -> None:
    from triton_model_tools.inference.folder_inference import run_folder_inference
    from triton_model_tools.models import discover_models

    image_dir: Path = args.image_dir.resolve()
    model_name: str = args.model
    triton_url: str = args.triton_url
    target_size: int | tuple[int, int] = args.target_size
    end2end: bool = args.end2end
    conf_threshold: float = args.conf_threshold
    iou_threshold: float = args.iou_threshold
    repository_root: Path | None = args.repository_root
    output_dir: Path | None = args.output_dir

    if repository_root is not None:
        discovered = discover_models(repository_root)
        match = next((m for m in discovered if m.name == model_name), None)
        if match is not None:
            end2end = match.is_end2end
            if match.input_dims and len(match.input_dims) >= 4:
                h, w = match.input_dims[-2], match.input_dims[-1]
                target_size = h if h == w else (h, w)
            logger.info("Auto-detected from repository: end2end={}, target_size={}", end2end, target_size)

    logger.info("Triton Folder Inference")
    logger.info("Image dir: {}", image_dir)
    logger.info("Model: {}", model_name)
    logger.info("Triton URL: {}", triton_url)

    result = run_folder_inference(
        image_dir=image_dir,
        model_name=model_name,
        triton_url=triton_url,
        target_size=target_size,
        end2end=end2end,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
    )

    logger.info("Results: {} images processed", len(result.results))
    logger.info("Total time: {:.2f}s", result.total_time_s)
    logger.info("Average time: {:.1f}ms per image", result.avg_time_ms)

    if output_dir is not None:
        from triton_model_tools.inference.detection_io import save_detections_to_txt

        resolved_output_dir = output_dir.resolve()
        resolved_output_dir.mkdir(parents=True, exist_ok=True)
        for img_result in result.results:
            stem = Path(img_result.image_path).stem
            save_detections_to_txt(img_result.detections, resolved_output_dir / f"{stem}.txt")

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


def run(args: argparse.Namespace) -> None:
    """Execute the ``infer`` subcommand."""
    local: bool = args.local
    if local:
        _run_local(args)
    else:
        _run_triton(args)

"""Compare detection txt results from two folders (baseline vs candidate)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    import argparse

    from triton_model_tools.cli._types import SubParsers


def build_parser(subparsers: SubParsers) -> None:
    """Register the ``compare-folders`` subcommand."""
    cf_parser = subparsers.add_parser(
        "compare-folders",
        help="Compare detection txt results from two folders (baseline vs candidate)",
    )
    cf_parser.add_argument(
        "--baseline-dir",
        type=Path,
        required=True,
        help="Directory with baseline detection txt files",
    )
    cf_parser.add_argument(
        "--candidate-dir",
        type=Path,
        required=True,
        help="Directory with candidate detection txt files",
    )
    cf_parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
        help="Model name for detection results",
    )
    cf_parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold for detection matching (default: 0.5)",
    )
    cf_parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output path for comparison results JSON",
    )
    cf_parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    """Execute the ``compare-folders`` subcommand."""
    import json

    from triton_model_tools.benchmarks.comparison import compare_inference
    from triton_model_tools.inference.detection_io import load_folder_detections

    baseline_dir: Path = args.baseline_dir.resolve()
    candidate_dir: Path = args.candidate_dir.resolve()
    model_name: str = args.model
    iou_threshold: float = args.iou_threshold
    output_path: Path | None = args.output

    logger.info("Folder Comparison")
    logger.info("Baseline: {}", baseline_dir)
    logger.info("Candidate: {}", candidate_dir)
    logger.info("Model: {}", model_name)

    baseline = load_folder_detections(baseline_dir, model_name)
    candidate = load_folder_detections(candidate_dir, model_name)
    summary = compare_inference(baseline, candidate, iou_threshold)

    logger.info("Overall precision: {:.4f}", summary.overall.precision)
    logger.info("Overall recall: {:.4f}", summary.overall.recall)
    logger.info("Overall avg IoU: {:.4f}", summary.overall.avg_iou)
    logger.info("Class mismatch rate: {:.4f}", summary.overall.class_mismatch_rate)
    logger.info("False positives: {}", summary.overall.false_positive_count)
    logger.info("Missed detections: {}", summary.overall.missed_count)

    if output_path is not None:
        resolved_output = output_path.resolve()
        resolved_output.parent.mkdir(parents=True, exist_ok=True)
        with resolved_output.open("w") as f:
            json.dump(summary.model_dump(), f, indent=2)
        logger.info("Comparison results saved to {}", resolved_output)

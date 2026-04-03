"""Visual comparison of local vs Triton detection results using FiftyOne."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    import argparse

    from triton_model_tools.cli._types import SubParsers
    from triton_model_tools.models.detection import ImageDetectionResult


def build_parser(subparsers: SubParsers) -> None:
    """Register the ``compare-visual`` subcommand."""
    cv_parser = subparsers.add_parser(
        "compare-visual",
        help="Visual drift comparison between local and Triton detection results via FiftyOne",
    )
    cv_parser.add_argument(
        "--local-dir",
        type=Path,
        required=True,
        help="Directory with local inference detection txt files (YOLO format)",
    )
    cv_parser.add_argument(
        "--triton-dir",
        type=Path,
        required=True,
        help="Directory with Triton inference detection txt files (YOLO format)",
    )
    cv_parser.add_argument(
        "--image-dir",
        type=Path,
        required=True,
        help="Directory containing the source images",
    )
    cv_parser.add_argument(
        "--dataset-name",
        type=str,
        default="local-vs-triton",
        help="FiftyOne dataset name (default: local-vs-triton)",
    )
    cv_parser.add_argument(
        "--launch",
        action="store_true",
        default=False,
        help="Launch the FiftyOne app after creating the dataset",
    )
    cv_parser.set_defaults(func=run)


def _build_results_with_image_paths(
    txt_dir: Path,
    image_dir: Path,
    model_name: str,
) -> list[ImageDetectionResult]:
    """Load txt detections and resolve image paths from the image directory.

    Each txt file is matched to an image by stem name. If no matching image is
    found, the txt file is skipped with a warning.
    """
    from triton_model_tools.inference.detection_io import (
        IMAGE_EXTENSIONS,
        TXT_EXTENSION,
        load_detections_from_txt,
    )

    image_map: dict[str, Path] = {}
    for p in image_dir.iterdir():
        if p.suffix.lower() in IMAGE_EXTENSIONS:
            image_map[p.stem] = p

    txt_files = sorted(txt_dir.glob(f"*{TXT_EXTENSION}"))
    if not txt_files:
        msg = f"No txt files found in {txt_dir}"
        raise FileNotFoundError(msg)

    results: list[ImageDetectionResult] = []
    skipped = 0
    for txt_path in txt_files:
        stem = txt_path.stem
        image_path = image_map.get(stem)
        if image_path is None:
            logger.warning("No matching image for {}, skipping", txt_path.name)
            skipped += 1
            continue
        result = load_detections_from_txt(
            txt_path,
            image_path=str(image_path),
            model_name=model_name,
        )
        results.append(result)

    logger.info(
        "Loaded {} detection files from {} ({} skipped, no matching image)",
        len(results),
        txt_dir,
        skipped,
    )
    return results


def run(args: argparse.Namespace) -> None:
    """Execute the ``compare-visual`` subcommand."""
    try:
        import fiftyone  # noqa: F401
    except ImportError:
        logger.error(
            "fiftyone is required for compare-visual. Install it with: pip install fiftyone",
        )
        raise SystemExit(1)  # noqa: B904

    from triton_model_tools.benchmarks.fiftyone_export import (
        export_comparison_to_fiftyone,
    )

    local_dir: Path = args.local_dir.resolve()
    triton_dir: Path = args.triton_dir.resolve()
    image_dir: Path = args.image_dir.resolve()
    dataset_name: str = args.dataset_name
    launch: bool = args.launch

    logger.info("Visual Comparison (FiftyOne)")
    logger.info("Local results: {}", local_dir)
    logger.info("Triton results: {}", triton_dir)
    logger.info("Image directory: {}", image_dir)
    logger.info("Dataset name: {}", dataset_name)

    local_results = _build_results_with_image_paths(local_dir, image_dir, model_name="local")
    triton_results = _build_results_with_image_paths(triton_dir, image_dir, model_name="triton")

    export_comparison_to_fiftyone(local_results, triton_results, dataset_name)

    if launch:
        import fiftyone as fo

        dataset: Any = fo.load_dataset(dataset_name)
        session: Any = fo.launch_app(dataset)
        session.wait()
    else:
        logger.info(
            "Dataset created. Launch FiftyOne app with: fiftyone app launch --dataset {}",
            dataset_name,
        )

"""Download COCO val2017 images for benchmarking."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    import argparse

    from triton_model_tools.cli._types import SubParsers


def build_parser(subparsers: SubParsers) -> None:
    """Register the ``download-coco`` subcommand."""
    download_parser = subparsers.add_parser(
        "download-coco",
        help="Download COCO val2017 images for benchmarking",
    )
    download_parser.add_argument(
        "--target-dir",
        type=Path,
        default=Path("data"),
        help="Target directory for COCO dataset (default: ./data)",
    )
    download_parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Limit number of extracted images (for quick testing)",
    )
    download_parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    """Execute the ``download-coco`` subcommand."""
    from triton_model_tools.benchmarks.dataset import download_coco_val  # noqa: PLC0415

    target_dir: Path = args.target_dir.resolve()
    max_images: int | None = args.max_images

    logger.info("COCO val2017 Downloader")
    logger.info("Target directory: {}", target_dir)
    if max_images is not None:
        logger.info("Max images: {}", max_images)

    image_dir = download_coco_val(target_dir, max_images=max_images)
    logger.info("Images available at: {}", image_dir)

"""COCO val2017 dataset downloader for benchmarking."""

from __future__ import annotations

import shutil
import tempfile
import urllib.request
import zipfile
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from pathlib import Path

COCO_VAL2017_URL = "https://images.cocodataset.org/zips/val2017.zip"
COCO_VAL2017_DIR_NAME = "val2017"
IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png"})
DOWNLOAD_CHUNK_SIZE = 8192
PROGRESS_LOG_INTERVAL_MB = 50


def _count_images(directory: Path) -> int:
    return sum(1 for p in directory.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


def _download_with_progress(url: str, dest_path: Path) -> None:
    logger.info("Downloading {} to {}", url, dest_path)
    with urllib.request.urlopen(url) as response, dest_path.open("wb") as out_file:  # noqa: S310
        content_length = response.headers.get("Content-Length")
        total_size = int(content_length) if content_length else None
        downloaded = 0
        last_logged_mb = 0

        while True:
            chunk = response.read(DOWNLOAD_CHUNK_SIZE)
            if not chunk:
                break
            out_file.write(chunk)
            downloaded += len(chunk)
            current_mb = downloaded // (1024 * 1024)

            if current_mb >= last_logged_mb + PROGRESS_LOG_INTERVAL_MB:
                last_logged_mb = current_mb
                if total_size:
                    pct = downloaded * 100 // total_size
                    logger.info("Downloaded {}MB / {}MB ({}%)", current_mb, total_size // (1024 * 1024), pct)
                else:
                    logger.info("Downloaded {}MB", current_mb)

    logger.info("Download complete: {} bytes", downloaded)


def download_coco_val(target_dir: Path, *, max_images: int | None = None) -> Path:
    """Download COCO val2017 images for benchmarking.

    Args:
        target_dir: Parent directory where val2017/ will be created.
        max_images: If set, only extract this many images (for quick testing).

    Returns:
        Path to the extracted image directory (target_dir/val2017/).

    """
    image_dir = target_dir / COCO_VAL2017_DIR_NAME

    if image_dir.is_dir():
        existing_count = _count_images(image_dir)
        if existing_count > 0:
            logger.info(
                "COCO val2017 already exists at {} ({} images), skipping",
                image_dir,
                existing_count,
            )
            return image_dir

    target_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_dir_str:
        tmp_dir = target_dir.__class__(tmp_dir_str)
        zip_path = tmp_dir / "val2017.zip"

        _download_with_progress(COCO_VAL2017_URL, zip_path)

        logger.info("Extracting archive to {}", target_dir)
        with zipfile.ZipFile(zip_path) as zf:
            members = [m for m in zf.namelist() if m.startswith(COCO_VAL2017_DIR_NAME + "/") and not m.endswith("/")]

            if max_images is not None:
                members = members[:max_images]
                logger.info("Limiting extraction to {} images", max_images)

            image_dir.mkdir(parents=True, exist_ok=True)
            extracted_count = 0
            for member in members:
                filename = member.split("/", 1)[1] if "/" in member else member
                if not filename:
                    continue
                source = zf.open(member)
                dest = image_dir / filename
                with source, dest.open("wb") as dest_file:
                    shutil.copyfileobj(source, dest_file)
                extracted_count += 1

            logger.info("Extracted {} images to {}", extracted_count, image_dir)

    return image_dir

"""Preprocessing verification utilities for YOLO and classification models."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import cv2
from pydantic import BaseModel, ConfigDict
from triton_shared_models.classification.preprocessing import CenterCropPreprocessor
from triton_shared_models.detection.preprocessing import LetterBoxPreprocessor

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np


class PreprocessingResult(BaseModel):
    """Result of running preprocessing on a single image."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    image_path: str
    tensor_shape: tuple[int, ...]
    scale: float
    pad_offset: tuple[int, int]


def _load_image(path: Path) -> np.ndarray[Any, np.dtype[np.uint8]]:
    image = cv2.imread(str(path))
    if image is None:
        msg = f"Failed to load image: {path}"
        raise FileNotFoundError(msg)
    return image  # type: ignore[return-value]


def verify_yolo_preprocessing(
    images: list[Path],
    target_size: int = 640,
) -> list[PreprocessingResult]:
    """Run LetterBoxPreprocessor on each image and return shape/scale/pad metadata.

    Args:
        images: Paths to BGR images (JPEG/PNG).
        target_size: Target square size for letterbox resize.

    Returns:
        One PreprocessingResult per image with tensor shape, scale, and pad offset.

    """
    preprocessor = LetterBoxPreprocessor(target_size=target_size)
    results: list[PreprocessingResult] = []

    for path in images:
        image = _load_image(path)
        tensor, scale, pad_offset = preprocessor.preprocess(image)

        results.append(
            PreprocessingResult(
                image_path=str(path),
                tensor_shape=tensor.shape,
                scale=scale,
                pad_offset=pad_offset,
            ),
        )

    return results


def verify_classification_preprocessing(
    images: list[Path],
    crop_size: int = 224,
) -> list[PreprocessingResult]:
    """Run CenterCropPreprocessor on each image and return shape metadata.

    Args:
        images: Paths to BGR images (JPEG/PNG).
        crop_size: Target crop size for center crop.

    Returns:
        One PreprocessingResult per image. Scale is 0.0 and pad_offset is (0, 0)
        since classification preprocessing does not use letterbox padding.

    """
    preprocessor = CenterCropPreprocessor(crop_size=crop_size)
    results: list[PreprocessingResult] = []

    for path in images:
        image = _load_image(path)
        tensor = preprocessor.preprocess(image)

        results.append(
            PreprocessingResult(
                image_path=str(path),
                tensor_shape=tensor.shape,
                scale=0.0,
                pad_offset=(0, 0),
            ),
        )

    return results

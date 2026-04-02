"""Backward-compatible re-export from detection subpackage."""

from __future__ import annotations

from triton_shared_models.detection.preprocessing import (
    LETTERBOX_PAD_VALUE,
    YOLO_NORMALIZE_DIVISOR,
    LetterBoxPreprocessor,
)

__all__ = ["LETTERBOX_PAD_VALUE", "YOLO_NORMALIZE_DIVISOR", "LetterBoxPreprocessor"]

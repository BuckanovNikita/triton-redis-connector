"""Object detection types and preprocessing."""

from __future__ import annotations

from triton_shared_models.detection.models import Detection
from triton_shared_models.detection.preprocessing import (
    LETTERBOX_PAD_VALUE,
    YOLO_NORMALIZE_DIVISOR,
    LetterBoxPreprocessor,
)

__all__ = ["Detection", "LETTERBOX_PAD_VALUE", "YOLO_NORMALIZE_DIVISOR", "LetterBoxPreprocessor"]

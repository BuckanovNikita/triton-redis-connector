"""YOLO letterbox preprocessing for Triton inference."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt

LETTERBOX_PAD_VALUE = 114
YOLO_NORMALIZE_DIVISOR = 255.0


class YoloPreprocessor:
    """Replicates ultralytics LetterBox preprocessing.

    Steps:
    1. Resize maintaining aspect ratio (scale to fit, not stretch)
    2. Pad with (114, 114, 114) to target_size (letterbox)
    3. BGR -> RGB
    4. HWC -> CHW
    5. /255.0
    6. cast to dtype (float32 or float16)
    7. batch dim

    """

    def __init__(
        self,
        target_size: int | tuple[int, int] = 640,
        dtype: type[np.floating[Any]] = np.float32,
    ) -> None:
        """Initialize with the model's expected input size and output dtype."""
        if isinstance(target_size, int):
            self._target_h = target_size
            self._target_w = target_size
        else:
            self._target_h, self._target_w = target_size
        self._dtype = dtype

    def preprocess(
        self,
        image: npt.NDArray[np.uint8],
    ) -> tuple[npt.NDArray[np.floating[Any]], float, tuple[int, int]]:
        """Preprocess a BGR image using letterbox method.

        Args:
            image: BGR image as HWC uint8 array.

        Returns:
            Tuple of (tensor, scale_factor, (pad_left, pad_top)).

        """
        orig_h, orig_w = image.shape[:2]
        scale = min(self._target_h / orig_h, self._target_w / orig_w)

        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)

        resized: npt.NDArray[np.uint8] = cv2.resize(  # type: ignore[assignment]
            image,
            (new_w, new_h),
            interpolation=cv2.INTER_LINEAR,
        )

        pad_w = self._target_w - new_w
        pad_h = self._target_h - new_h
        pad_left = pad_w // 2
        pad_top = pad_h // 2
        pad_right = pad_w - pad_left
        pad_bottom = pad_h - pad_top

        padded: npt.NDArray[np.uint8] = cv2.copyMakeBorder(  # type: ignore[assignment]
            resized,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_CONSTANT,
            value=(LETTERBOX_PAD_VALUE, LETTERBOX_PAD_VALUE, LETTERBOX_PAD_VALUE),
        )

        rgb = padded[:, :, ::-1]
        chw = np.transpose(rgb, (2, 0, 1))
        normalized: npt.NDArray[np.floating[Any]] = chw.astype(self._dtype) / YOLO_NORMALIZE_DIVISOR
        batched: npt.NDArray[np.floating[Any]] = np.expand_dims(normalized, axis=0)

        return batched, scale, (pad_left, pad_top)

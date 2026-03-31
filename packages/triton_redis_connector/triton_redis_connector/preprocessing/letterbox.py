"""LetterBox preprocessor replicating ultralytics preprocessing exactly.

This implementation MUST produce identical results to YoloPreprocessor
from triton_model_tools/inference.py. Both share the same algorithm:
1. Resize maintaining aspect ratio
2. Pad with (114, 114, 114) to target_size
3. BGR -> RGB
4. HWC -> CHW
5. /255.0
6. float32
7. Batch dim
"""

from __future__ import annotations

import cv2
import numpy as np
import numpy.typing as npt

LETTERBOX_PAD_VALUE = 114
NORMALIZE_DIVISOR = 255.0


class LetterBoxPreprocessor:
    """Transforms a decoded BGR image into a model-ready float32 tensor.

    Implements ImagePreprocessorProtocol.
    """

    def preprocess(
        self,
        image: npt.NDArray[np.uint8],
        target_size: tuple[int, int],
    ) -> tuple[npt.NDArray[np.float32], float, tuple[int, int]]:
        """Preprocess image using the letterbox method.

        Args:
            image: BGR image as HWC uint8 array.
            target_size: (height, width) of the model input.

        Returns:
            Tuple of (NCHW float32 tensor, scale_factor, (pad_left, pad_top)).

        """
        target_h, target_w = target_size
        orig_h, orig_w = image.shape[:2]

        scale = min(target_h / orig_h, target_w / orig_w)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)

        resized: npt.NDArray[np.uint8] = cv2.resize(  # type: ignore[assignment]
            image,
            (new_w, new_h),
            interpolation=cv2.INTER_LINEAR,
        )

        pad_w = target_w - new_w
        pad_h = target_h - new_h
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
        normalized = chw.astype(np.float32) / NORMALIZE_DIVISOR
        batched: npt.NDArray[np.float32] = np.expand_dims(normalized, axis=0)

        return batched, scale, (pad_left, pad_top)

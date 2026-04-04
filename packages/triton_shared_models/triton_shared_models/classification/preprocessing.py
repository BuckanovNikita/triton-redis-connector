"""Classification preprocessing: resize short edge + center crop + ImageNet normalize."""

from typing import Any

import cv2
import numpy as np
import numpy.typing as npt

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SHORT_EDGE_SIZE = 256
CROP_SIZE = 224


class CenterCropPreprocessor:
    """ImageNet-style preprocessing for classification models.

    Steps:
    1. Resize short edge to 256 maintaining aspect ratio
    2. Center crop 224x224
    3. BGR -> RGB
    4. HWC -> CHW
    5. /255.0
    6. ImageNet normalize (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    7. Cast to dtype
    8. Add batch dim

    """

    def __init__(
        self,
        crop_size: int = CROP_SIZE,
        short_edge: int = SHORT_EDGE_SIZE,
        dtype: type[np.floating[Any]] = np.float32,
    ) -> None:
        """Initialize with crop size, short edge resize target, and output dtype."""
        self._crop_size = crop_size
        self._short_edge = short_edge
        self._dtype = dtype

    def preprocess(
        self,
        image: npt.NDArray[np.uint8],
    ) -> npt.NDArray[np.floating[Any]]:
        """Preprocess a BGR image using ImageNet-style center crop.

        Args:
            image: BGR image as HWC uint8 array.

        Returns:
            NCHW float tensor with ImageNet normalization applied.

        """
        orig_h, orig_w = image.shape[:2]
        if orig_h < orig_w:
            new_h = self._short_edge
            new_w = int(orig_w * self._short_edge / orig_h)
        else:
            new_w = self._short_edge
            new_h = int(orig_h * self._short_edge / orig_w)

        resized: npt.NDArray[np.uint8] = cv2.resize(  # type: ignore[assignment]
            image,
            (new_w, new_h),
            interpolation=cv2.INTER_LINEAR,
        )

        start_h = (new_h - self._crop_size) // 2
        start_w = (new_w - self._crop_size) // 2
        cropped = resized[start_h : start_h + self._crop_size, start_w : start_w + self._crop_size]

        rgb = cropped[:, :, ::-1]
        chw = np.transpose(rgb, (2, 0, 1))
        normalized = chw.astype(self._dtype) / 255.0

        mean = np.array(IMAGENET_MEAN, dtype=self._dtype).reshape(3, 1, 1)
        std = np.array(IMAGENET_STD, dtype=self._dtype).reshape(3, 1, 1)
        standardized = (normalized - mean) / std

        return np.expand_dims(standardized, axis=0)

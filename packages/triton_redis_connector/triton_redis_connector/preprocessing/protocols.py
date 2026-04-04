"""Preprocessing protocols for image decoding and tensor preparation.

These protocols define the contract for image decoders and preprocessors.
Multiple implementations (OpenCV, TurboJPEG) share the same interface.
"""

from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt


@runtime_checkable
class ImageBytesProtocol(Protocol):
    """Provides raw image bytes for decoding."""

    def to_bytes(self) -> bytes:
        """Return raw image bytes (e.g. JPEG, PNG encoded)."""
        ...


@runtime_checkable
class RawImageProtocol(Protocol):
    """Provides a preprocessed float32 tensor ready for inference."""

    def to_raw_image(self) -> npt.NDArray[np.float32]:
        """Return a preprocessed float32 tensor."""
        ...


@runtime_checkable
class ImageDecoderProtocol(Protocol):
    """Decodes raw image bytes into a numpy BGR HWC array.

    Implementations must release the GIL during decode to allow
    concurrent execution in thread pool processors.
    """

    def decode(self, raw: bytes) -> npt.NDArray[np.uint8]:
        """Decode raw image bytes into a BGR HWC uint8 numpy array.

        Raises ValueError if the image cannot be decoded.
        """
        ...


@runtime_checkable
class ImagePreprocessorProtocol(Protocol):
    """Transforms a decoded BGR image into a model-ready float32 tensor.

    Must replicate the ultralytics LetterBox preprocessing exactly:
    1. Resize maintaining aspect ratio
    2. Pad with (114, 114, 114) to target_size
    3. BGR -> RGB
    4. HWC -> CHW
    5. /255.0
    6. float32
    7. Batch dim

    Target size is provided at construction time.
    """

    def preprocess(
        self,
        image: npt.NDArray[np.uint8],
    ) -> tuple[npt.NDArray[np.float32], float, tuple[int, int]]:
        """Preprocess image to model-ready tensor.

        Returns:
            tensor: NCHW float32 tensor ready for inference.
            scale: Scale factor applied during resize.
            pad_offset: (pad_x, pad_y) pixel offsets from letterbox padding.

        """
        ...

"""ThreadPoolProcessor that decodes raw image bytes into BGR numpy arrays.

Wraps OpenCVDecoder for use as a pipeline graph node. Consumes raw bytes
and produces DecodedImage events for downstream preprocessing.
"""

from __future__ import annotations

import threading
import time

from loguru import logger

from triton_redis_connector.engine.events import DecodedImage
from triton_redis_connector.errors.models import PreprocessingError
from triton_redis_connector.preprocessing.decoder import OpenCVDecoder


class ImageDecodeProcessor:
    """ThreadPoolProcessor wrapping OpenCVDecoder.

    Decodes raw image bytes into BGR HWC uint8 numpy arrays.
    Returns PreprocessingError on decode failure instead of raising.
    """

    def __init__(self, decoder: OpenCVDecoder | None = None) -> None:  # noqa: D107
        self.stop = threading.Event()
        self._decoder = decoder or OpenCVDecoder()

    def __call__(self, event: bytes) -> DecodedImage | PreprocessingError | None:
        """Decode raw image bytes into a DecodedImage event."""
        try:
            image = self._decoder.decode(event)
            h, w = image.shape[:2]
            return DecodedImage(
                image=image,
                original_height=h,
                original_width=w,
            )
        except ValueError as exc:
            logger.warning("Image decode failed: {}", exc)
            return PreprocessingError(
                message_id="",
                decoder_type="opencv",
                error_type="DecodeError",
                error_message=str(exc),
                timestamp_ms=int(time.time() * 1000),
            )

"""Pure image decode processor.

Decodes raw image bytes into BGR numpy arrays. No threading primitives —
concurrency is handled by framework wrappers (ThreadedProcessor).
"""

from __future__ import annotations

import time

from loguru import logger

from triton_redis_connector.config.schema import DecoderType
from triton_redis_connector.engine.events import DecodedImage
from triton_redis_connector.errors.models import PreprocessingError
from triton_redis_connector.preprocessing.decoder import OpenCVDecoder, TurboJPEGDecoder


class ImageDecodeProcessor:
    """Processor that decodes raw image bytes into BGR HWC uint8 numpy arrays.

    Returns PreprocessingError on decode failure instead of raising.
    """

    def __init__(  # noqa: D107
        self,
        decoder: OpenCVDecoder | TurboJPEGDecoder | None = None,
        decoder_type: DecoderType = DecoderType.OPENCV,
    ) -> None:
        if decoder is not None:
            self._decoder = decoder
        elif decoder_type == DecoderType.TURBOJPEG:
            self._decoder = TurboJPEGDecoder()
        else:
            self._decoder = OpenCVDecoder()
        self._decoder_type = decoder_type

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
                decoder_type=self._decoder_type.value,
                error_type="DecodeError",
                error_message=str(exc),
                timestamp_ms=int(time.time() * 1000),
            )

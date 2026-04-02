"""Image decoders implementing ImageDecoderProtocol.

OpenCVDecoder supports all image formats via cv2.imdecode.
TurboJPEGDecoder uses PyTurboJPEG for faster JPEG-only decoding.

Both release the GIL during decode, enabling concurrent execution
in thread pool processors.
"""

from __future__ import annotations

import threading

import cv2
import numpy as np
import numpy.typing as npt
import turbojpeg

from triton_redis_connector.config.schema import DecoderType
from triton_redis_connector.preprocessing.protocols import ImageDecoderProtocol


class OpenCVDecoder(ImageDecoderProtocol):
    """Decodes raw image bytes into BGR HWC uint8 numpy array using OpenCV."""

    def decode(self, raw: bytes) -> npt.NDArray[np.uint8]:
        """Decode raw image bytes into a BGR HWC uint8 numpy array.

        Raises:
            ValueError: If the image cannot be decoded.

        """
        if not raw:
            msg = "Failed to decode image: empty input bytes"
            raise ValueError(msg)

        buf = np.frombuffer(raw, dtype=np.uint8)
        try:
            image: npt.NDArray[np.uint8] | None = cv2.imdecode(buf, cv2.IMREAD_COLOR)  # type: ignore[assignment]
        except cv2.error as exc:
            msg = f"Failed to decode image: {exc}"
            raise ValueError(msg) from exc

        if image is None:
            msg = "Failed to decode image: cv2.imdecode returned None"
            raise ValueError(msg)

        return image


class TurboJPEGDecoder(ImageDecoderProtocol):
    """Decodes JPEG bytes using PyTurboJPEG. Faster for JPEG-only workloads.

    Uses thread-local TurboJPEG instances for thread safety in ThreadPoolProcessor.
    """

    def __init__(self) -> None:  # noqa: D107
        self._local = threading.local()

    def _get_decoder(self) -> turbojpeg.TurboJPEG:
        if not hasattr(self._local, "decoder"):
            self._local.decoder = turbojpeg.TurboJPEG()  # type: ignore[no-untyped-call]
        return self._local.decoder  # type: ignore[no-any-return]

    def decode(self, raw: bytes) -> npt.NDArray[np.uint8]:
        """Decode JPEG bytes into a BGR HWC uint8 numpy array.

        Raises:
            ValueError: If the input is empty or not valid JPEG data.

        """
        if not raw:
            msg = "Failed to decode image: empty input bytes"
            raise ValueError(msg)

        decoder = self._get_decoder()
        try:
            image: npt.NDArray[np.uint8] | None = decoder.decode(  # type: ignore[no-untyped-call]
                raw, pixel_format=turbojpeg.TJPF_BGR
            )
        except OSError as exc:
            msg = f"Failed to decode image: {exc}"
            raise ValueError(msg) from exc

        if image is None:  # pyright: ignore[reportUnnecessaryComparison]
            msg = "Failed to decode image: TurboJPEG.decode returned None"
            raise ValueError(msg)

        return image


def create_decoder(decoder_type: DecoderType) -> ImageDecoderProtocol:
    """Create an image decoder based on the configured decoder type."""
    match decoder_type:
        case DecoderType.OPENCV | DecoderType.DALI:
            return OpenCVDecoder()
        case DecoderType.TURBOJPEG:
            return TurboJPEGDecoder()

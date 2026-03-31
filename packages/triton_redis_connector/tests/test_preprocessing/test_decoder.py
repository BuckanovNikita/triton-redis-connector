"""Tests for OpenCVDecoder with real image bytes."""

from __future__ import annotations

import numpy as np
import pytest
from triton_redis_connector.preprocessing.decoder import OpenCVDecoder
from triton_redis_connector.preprocessing.protocols import ImageDecoderProtocol


def _make_tiny_jpeg() -> bytes:
    """Create a minimal valid JPEG image (8x8 red square)."""
    import cv2

    image = np.full((8, 8, 3), (0, 0, 255), dtype=np.uint8)
    success, encoded = cv2.imencode(".jpg", image)
    assert success
    return bytes(encoded)


def _make_tiny_png() -> bytes:
    """Create a minimal valid PNG image (4x4 green square)."""
    import cv2

    image = np.full((4, 4, 3), (0, 255, 0), dtype=np.uint8)
    success, encoded = cv2.imencode(".png", image)
    assert success
    return bytes(encoded)


class TestOpenCVDecoder:
    def test_implements_protocol(self) -> None:
        decoder = OpenCVDecoder()
        assert isinstance(decoder, ImageDecoderProtocol)

    def test_decode_jpeg(self) -> None:
        decoder = OpenCVDecoder()
        jpeg_bytes = _make_tiny_jpeg()

        result = decoder.decode(jpeg_bytes)

        assert result.dtype == np.uint8
        assert result.ndim == 3
        assert result.shape[2] == 3
        assert result.shape[0] == 8
        assert result.shape[1] == 8

    def test_decode_png(self) -> None:
        decoder = OpenCVDecoder()
        png_bytes = _make_tiny_png()

        result = decoder.decode(png_bytes)

        assert result.dtype == np.uint8
        assert result.ndim == 3
        assert result.shape == (4, 4, 3)

    def test_decode_invalid_bytes_raises_value_error(self) -> None:
        decoder = OpenCVDecoder()

        with pytest.raises(ValueError, match="Failed to decode image"):
            decoder.decode(b"not a valid image")

    def test_decode_empty_bytes_raises_value_error(self) -> None:
        decoder = OpenCVDecoder()

        with pytest.raises(ValueError, match="Failed to decode image"):
            decoder.decode(b"")

    def test_decoded_image_is_bgr(self) -> None:
        decoder = OpenCVDecoder()
        import cv2

        red_image = np.full((4, 4, 3), (0, 0, 255), dtype=np.uint8)
        _, encoded = cv2.imencode(".png", red_image)
        result = decoder.decode(bytes(encoded))
        assert result[0, 0, 2] > 200
        assert result[0, 0, 0] < 50

"""Tests for OpenCVDecoder and TurboJPEGDecoder with real image bytes."""

from __future__ import annotations

import numpy as np
import pytest
import turbojpeg as _tjpeg  # pyright: ignore[reportMissingTypeStubs]
from triton_redis_connector.preprocessing.decoder import OpenCVDecoder, TurboJPEGDecoder


def _turbojpeg_available() -> bool:
    try:
        _tjpeg.TurboJPEG()
    except RuntimeError:
        return False
    return True


_skip_no_turbojpeg = pytest.mark.skipif(
    not _turbojpeg_available(),
    reason="libturbojpeg native library not installed",
)


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


@_skip_no_turbojpeg
class TestTurboJPEGDecoder:
    def test_decode_jpeg(self) -> None:
        decoder = TurboJPEGDecoder()
        jpeg_bytes = _make_tiny_jpeg()

        result = decoder.decode(jpeg_bytes)

        assert result.dtype == np.uint8
        assert result.ndim == 3
        assert result.shape[2] == 3
        assert result.shape[0] == 8
        assert result.shape[1] == 8

    def test_decode_empty_bytes_raises_value_error(self) -> None:
        decoder = TurboJPEGDecoder()

        with pytest.raises(ValueError, match="Failed to decode image"):
            decoder.decode(b"")

    def test_decode_invalid_bytes_raises_value_error(self) -> None:
        decoder = TurboJPEGDecoder()

        with pytest.raises(ValueError, match="Failed to decode image"):
            decoder.decode(b"not a valid jpeg")

    def test_decoded_image_is_bgr(self) -> None:
        decoder = TurboJPEGDecoder()
        import cv2

        red_image = np.full((16, 16, 3), (0, 0, 255), dtype=np.uint8)
        _, encoded = cv2.imencode(".jpg", red_image, [cv2.IMWRITE_JPEG_QUALITY, 100])
        result = decoder.decode(bytes(encoded))
        assert result[0, 0, 2] > 200
        assert result[0, 0, 0] < 50

    def test_thread_local_isolation(self) -> None:
        import threading

        decoder = TurboJPEGDecoder()
        jpeg_bytes = _make_tiny_jpeg()
        results: list[tuple[int, int]] = []
        errors: list[ValueError] = []

        def decode_in_thread() -> None:
            try:
                result = decoder.decode(jpeg_bytes)
                results.append(result.shape[:2])
            except ValueError as exc:
                errors.append(exc)

        threads = [threading.Thread(target=decode_in_thread) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(results) == 4
        assert all(shape == (8, 8) for shape in results)

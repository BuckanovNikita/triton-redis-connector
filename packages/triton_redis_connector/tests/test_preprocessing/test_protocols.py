"""Tests for preprocessing protocol isinstance checks."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from triton_redis_connector.preprocessing.protocols import (
    ImageDecoderProtocol,
    ImagePreprocessorProtocol,
)


class DummyDecoder:
    def decode(self, raw: bytes) -> npt.NDArray[np.uint8]:
        return np.zeros((480, 640, 3), dtype=np.uint8)


class DummyPreprocessor:
    def preprocess(
        self,
        image: npt.NDArray[np.uint8],
        target_size: tuple[int, int],
    ) -> tuple[npt.NDArray[np.float32], float, tuple[int, int]]:
        h, w = target_size
        tensor = np.zeros((1, 3, h, w), dtype=np.float32)
        return tensor, 1.0, (0, 0)


class NotADecoder:
    def unrelated(self) -> None:
        pass


def test_decoder_protocol_isinstance() -> None:
    decoder = DummyDecoder()
    assert isinstance(decoder, ImageDecoderProtocol)


def test_preprocessor_protocol_isinstance() -> None:
    preprocessor = DummyPreprocessor()
    assert isinstance(preprocessor, ImagePreprocessorProtocol)


def test_non_decoder_fails_isinstance() -> None:
    obj = NotADecoder()
    assert not isinstance(obj, ImageDecoderProtocol)
    assert not isinstance(obj, ImagePreprocessorProtocol)


def test_decoder_returns_correct_shape() -> None:
    decoder = DummyDecoder()
    result = decoder.decode(b"\xff\xd8\xff\xe0")
    assert result.shape == (480, 640, 3)
    assert result.dtype == np.uint8


def test_preprocessor_returns_correct_shape() -> None:
    preprocessor = DummyPreprocessor()
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    tensor, scale, pad = preprocessor.preprocess(image, (640, 640))
    assert tensor.shape == (1, 3, 640, 640)
    assert tensor.dtype == np.float32
    assert scale == 1.0
    assert pad == (0, 0)

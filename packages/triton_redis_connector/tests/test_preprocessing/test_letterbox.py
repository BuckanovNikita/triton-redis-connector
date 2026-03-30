"""Tests for LetterBoxPreprocessor: shape, dtype, padding, aspect ratio."""

from __future__ import annotations

import numpy as np
import pytest
from triton_redis_connector.preprocessing.letterbox import (
    LETTERBOX_PAD_VALUE,
    NORMALIZE_DIVISOR,
    LetterBoxPreprocessor,
)
from triton_redis_connector.preprocessing.protocols import ImagePreprocessorProtocol


class TestLetterBoxPreprocessor:
    def test_implements_protocol(self) -> None:
        preprocessor = LetterBoxPreprocessor()
        assert isinstance(preprocessor, ImagePreprocessorProtocol)

    def test_output_shape_square_target(self) -> None:
        preprocessor = LetterBoxPreprocessor()
        image = np.zeros((480, 640, 3), dtype=np.uint8)

        tensor, _scale, _pad_offset = preprocessor.preprocess(image, (640, 640))

        assert tensor.shape == (1, 3, 640, 640)

    def test_output_dtype_float32(self) -> None:
        preprocessor = LetterBoxPreprocessor()
        image = np.zeros((480, 640, 3), dtype=np.uint8)

        tensor, _, _ = preprocessor.preprocess(image, (640, 640))

        assert tensor.dtype == np.float32

    def test_values_normalized_to_0_1(self) -> None:
        preprocessor = LetterBoxPreprocessor()
        image = np.full((100, 100, 3), 255, dtype=np.uint8)

        tensor, _, _ = preprocessor.preprocess(image, (640, 640))

        assert tensor.max() <= 1.0
        assert tensor.min() >= 0.0

    def test_padding_values_normalized(self) -> None:
        preprocessor = LetterBoxPreprocessor()
        image = np.zeros((100, 200, 3), dtype=np.uint8)

        tensor, _, (_pad_left, _pad_top) = preprocessor.preprocess(image, (640, 640))

        expected_pad_value = LETTERBOX_PAD_VALUE / NORMALIZE_DIVISOR
        corner_value = tensor[0, 0, 0, 0]
        assert corner_value == pytest.approx(expected_pad_value, abs=0.01)

    def test_scale_factor_correct_landscape(self) -> None:
        preprocessor = LetterBoxPreprocessor()
        image = np.zeros((480, 640, 3), dtype=np.uint8)

        _, scale, _ = preprocessor.preprocess(image, (640, 640))

        expected_scale = min(640 / 480, 640 / 640)
        assert scale == pytest.approx(expected_scale)

    def test_scale_factor_correct_portrait(self) -> None:
        preprocessor = LetterBoxPreprocessor()
        image = np.zeros((800, 400, 3), dtype=np.uint8)

        _, scale, _ = preprocessor.preprocess(image, (640, 640))

        expected_scale = min(640 / 800, 640 / 400)
        assert scale == pytest.approx(expected_scale)

    def test_pad_offset_landscape(self) -> None:
        preprocessor = LetterBoxPreprocessor()
        image = np.zeros((480, 640, 3), dtype=np.uint8)

        _, scale, (pad_left, pad_top) = preprocessor.preprocess(image, (640, 640))

        new_w = int(640 * scale)
        new_h = int(480 * scale)
        expected_pad_left = (640 - new_w) // 2
        expected_pad_top = (640 - new_h) // 2

        assert pad_left == expected_pad_left
        assert pad_top == expected_pad_top

    def test_pad_offset_portrait(self) -> None:
        preprocessor = LetterBoxPreprocessor()
        image = np.zeros((800, 400, 3), dtype=np.uint8)

        _, scale, (pad_left, pad_top) = preprocessor.preprocess(image, (640, 640))

        new_w = int(400 * scale)
        new_h = int(800 * scale)
        expected_pad_left = (640 - new_w) // 2
        expected_pad_top = (640 - new_h) // 2

        assert pad_left == expected_pad_left
        assert pad_top == expected_pad_top

    def test_square_image_no_padding(self) -> None:
        preprocessor = LetterBoxPreprocessor()
        image = np.zeros((640, 640, 3), dtype=np.uint8)

        _, scale, (pad_left, pad_top) = preprocessor.preprocess(image, (640, 640))

        assert scale == pytest.approx(1.0)
        assert pad_left == 0
        assert pad_top == 0

    def test_small_square_image_scaled_up(self) -> None:
        preprocessor = LetterBoxPreprocessor()
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        tensor, scale, (pad_left, pad_top) = preprocessor.preprocess(image, (640, 640))

        assert tensor.shape == (1, 3, 640, 640)
        assert scale == pytest.approx(640 / 100)
        assert pad_left == 0
        assert pad_top == 0

    def test_bgr_to_rgb_conversion(self) -> None:
        preprocessor = LetterBoxPreprocessor()
        image = np.zeros((640, 640, 3), dtype=np.uint8)
        image[:, :, 0] = 255
        image[:, :, 1] = 0
        image[:, :, 2] = 0

        tensor, _, _ = preprocessor.preprocess(image, (640, 640))

        assert tensor[0, 2, 0, 0] == pytest.approx(1.0, abs=0.01)
        assert tensor[0, 0, 0, 0] == pytest.approx(0.0, abs=0.01)

    def test_hwc_to_chw_layout(self) -> None:
        preprocessor = LetterBoxPreprocessor()
        image = np.zeros((640, 640, 3), dtype=np.uint8)

        tensor, _, _ = preprocessor.preprocess(image, (640, 640))

        assert tensor.shape[1] == 3

    def test_various_input_sizes(self) -> None:
        preprocessor = LetterBoxPreprocessor()
        sizes = [(1920, 1080), (320, 240), (1, 1), (1000, 500), (500, 1000)]

        for h, w in sizes:
            image = np.zeros((h, w, 3), dtype=np.uint8)
            tensor, scale, (_pad_left, _pad_top) = preprocessor.preprocess(image, (640, 640))

            assert tensor.shape == (1, 3, 640, 640)
            assert tensor.dtype == np.float32
            assert scale > 0

    def test_non_square_target_size(self) -> None:
        preprocessor = LetterBoxPreprocessor()
        image = np.zeros((480, 640, 3), dtype=np.uint8)

        tensor, _scale, _ = preprocessor.preprocess(image, (416, 416))

        assert tensor.shape == (1, 3, 416, 416)

"""Tests for preprocessing verification utilities."""

from pathlib import Path

import cv2
import numpy as np
import pytest
from triton_model_tools.testing.preprocessing_verify import (
    PreprocessingResult,
    verify_classification_preprocessing,
    verify_yolo_preprocessing,
)
from triton_shared_models.classification.preprocessing import CenterCropPreprocessor
from triton_shared_models.detection.preprocessing import LetterBoxPreprocessor

IMAGENET_MIN = -2.2
IMAGENET_MAX = 2.7


class TestYoloPreprocessingVerify:
    def test_output_shape_matches_target(self, test_images_dir: Path) -> None:
        images = sorted(test_images_dir.glob("*.jpg"))
        results = verify_yolo_preprocessing(images, target_size=640)

        for result in results:
            assert result.tensor_shape == (1, 3, 640, 640)

    def test_value_range_zero_to_one(self, test_images_dir: Path) -> None:
        images = sorted(test_images_dir.glob("*.jpg"))[:2]
        preprocessor = LetterBoxPreprocessor(target_size=640)

        for img_path in images:
            image = cv2.imread(str(img_path))
            tensor, _, _ = preprocessor.preprocess(image)
            assert tensor.min() >= 0.0
            assert tensor.max() <= 1.0

    def test_batch_consistency(self, test_images_dir: Path) -> None:
        images = sorted(test_images_dir.glob("*.jpg"))[:1]
        preprocessor = LetterBoxPreprocessor(target_size=640)

        image = cv2.imread(str(images[0]))
        tensor_a, scale_a, pad_a = preprocessor.preprocess(image)
        tensor_b, scale_b, pad_b = preprocessor.preprocess(image)

        np.testing.assert_array_equal(tensor_a, tensor_b)
        assert scale_a == scale_b
        assert pad_a == pad_b

    @pytest.mark.parametrize(
        ("width", "height"),
        [
            (1920, 1080),
            (640, 480),
            (100, 100),
            (3840, 2160),
            (400, 800),
        ],
        ids=["1920x1080", "640x480", "100x100", "4k", "portrait"],
    )
    def test_different_input_sizes_produce_same_output_shape(
        self,
        tmp_path: Path,
        width: int,
        height: int,
    ) -> None:
        rng = np.random.default_rng(42)
        image = rng.integers(0, 255, (height, width, 3), dtype=np.uint8)
        img_path = tmp_path / f"test_{width}x{height}.jpg"
        cv2.imwrite(str(img_path), image)

        results = verify_yolo_preprocessing([img_path], target_size=640)

        assert len(results) == 1
        assert results[0].tensor_shape == (1, 3, 640, 640)

    def test_verify_returns_correct_result_count(self, test_images_dir: Path) -> None:
        images = sorted(test_images_dir.glob("*.jpg"))
        results = verify_yolo_preprocessing(images, target_size=640)

        assert len(results) == len(images)
        assert all(isinstance(r, PreprocessingResult) for r in results)

    def test_scale_factor_is_positive(self, test_images_dir: Path) -> None:
        images = sorted(test_images_dir.glob("*.jpg"))
        results = verify_yolo_preprocessing(images, target_size=640)

        for result in results:
            assert result.scale > 0.0

    def test_nonexistent_image_raises(self, tmp_path: Path) -> None:
        fake_path = tmp_path / "nonexistent.jpg"
        with pytest.raises(FileNotFoundError, match="Failed to load image"):
            verify_yolo_preprocessing([fake_path])


class TestClassificationPreprocessingVerify:
    def test_output_shape_matches_crop_size(self, test_images_dir: Path) -> None:
        images = sorted(test_images_dir.glob("*.jpg"))
        results = verify_classification_preprocessing(images, crop_size=224)

        for result in results:
            assert result.tensor_shape == (1, 3, 224, 224)

    def test_imagenet_normalized_value_range(self, test_images_dir: Path) -> None:
        images = sorted(test_images_dir.glob("*.jpg"))[:2]
        preprocessor = CenterCropPreprocessor(crop_size=224)

        for img_path in images:
            image = cv2.imread(str(img_path))
            tensor = preprocessor.preprocess(image)
            assert tensor.min() >= IMAGENET_MIN
            assert tensor.max() <= IMAGENET_MAX

    def test_classification_scale_and_pad_are_zero(self, test_images_dir: Path) -> None:
        images = sorted(test_images_dir.glob("*.jpg"))[:1]
        results = verify_classification_preprocessing(images)

        for result in results:
            assert result.scale == 0.0
            assert result.pad_offset == (0, 0)

    def test_classification_batch_consistency(self, test_images_dir: Path) -> None:
        images = sorted(test_images_dir.glob("*.jpg"))[:1]
        preprocessor = CenterCropPreprocessor(crop_size=224)

        image = cv2.imread(str(images[0]))
        tensor_a = preprocessor.preprocess(image)
        tensor_b = preprocessor.preprocess(image)

        np.testing.assert_array_equal(tensor_a, tensor_b)

    @pytest.mark.parametrize(
        ("width", "height"),
        [
            (1920, 1080),
            (640, 480),
            (300, 300),
            (400, 800),
        ],
        ids=["1920x1080", "640x480", "square", "portrait"],
    )
    def test_different_input_sizes_produce_same_output_shape(
        self,
        tmp_path: Path,
        width: int,
        height: int,
    ) -> None:
        rng = np.random.default_rng(42)
        image = rng.integers(0, 255, (height, width, 3), dtype=np.uint8)
        img_path = tmp_path / f"test_{width}x{height}.jpg"
        cv2.imwrite(str(img_path), image)

        results = verify_classification_preprocessing([img_path], crop_size=224)

        assert len(results) == 1
        assert results[0].tensor_shape == (1, 3, 224, 224)

    def test_nonexistent_image_raises(self, tmp_path: Path) -> None:
        fake_path = tmp_path / "nonexistent.jpg"
        with pytest.raises(FileNotFoundError, match="Failed to load image"):
            verify_classification_preprocessing([fake_path])

"""Tests for YOLO preprocessing, postprocessing, and Triton inference client."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from triton_model_tools.inference import (
    LETTERBOX_PAD_VALUE,
    YOLO_NORMALIZE_DIVISOR,
    TritonInferenceClient,
    YoloPostprocessor,
    YoloPreprocessor,
)
from triton_model_tools.inference.postprocessing import _nms_numpy


class TestYoloPreprocessor:
    def test_output_shape_and_dtype(self) -> None:
        preprocessor = YoloPreprocessor(target_size=640)
        image = np.random.default_rng(42).integers(0, 255, (480, 640, 3), dtype=np.uint8)

        tensor, _scale, _pad_offset = preprocessor.preprocess(image)

        assert tensor.shape == (1, 3, 640, 640)
        assert tensor.dtype == np.float32

    def test_scale_factor_for_landscape_image(self) -> None:
        preprocessor = YoloPreprocessor(target_size=640)
        image = np.zeros((480, 640, 3), dtype=np.uint8)

        _, scale, _ = preprocessor.preprocess(image)

        expected_scale = 640 / 640
        assert scale == pytest.approx(expected_scale)

    def test_scale_factor_for_portrait_image(self) -> None:
        preprocessor = YoloPreprocessor(target_size=640)
        image = np.zeros((800, 400, 3), dtype=np.uint8)

        _, scale, _ = preprocessor.preprocess(image)

        expected_scale = 640 / 800
        assert scale == pytest.approx(expected_scale)

    def test_padding_is_centered(self) -> None:
        preprocessor = YoloPreprocessor(target_size=640)
        image = np.zeros((480, 640, 3), dtype=np.uint8)

        _, _, pad_offset = preprocessor.preprocess(image)

        pad_left, pad_top = pad_offset
        new_h = int(480 * (640 / 640))
        expected_pad_top = (640 - new_h) // 2
        assert pad_left == 0
        assert pad_top == expected_pad_top

    def test_padding_values_are_114_normalized(self) -> None:
        preprocessor = YoloPreprocessor(target_size=640)
        image = np.zeros((480, 640, 3), dtype=np.uint8)

        tensor, _, pad_offset = preprocessor.preprocess(image)
        _, pad_top = pad_offset

        if pad_top > 0:
            pad_region = tensor[0, :, :pad_top, :]
            expected_pad = LETTERBOX_PAD_VALUE / YOLO_NORMALIZE_DIVISOR
            np.testing.assert_allclose(pad_region, expected_pad, atol=1e-6)

    def test_letterbox_cross_validation(self) -> None:
        """Cross-validate preprocessor against manual letterbox computation."""
        preprocessor = YoloPreprocessor(target_size=640)
        rng = np.random.default_rng(123)
        image = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)

        tensor, scale, pad_offset = preprocessor.preprocess(image)
        pad_left, pad_top = pad_offset

        assert tensor.shape == (1, 3, 640, 640)
        assert tensor.dtype == np.float32

        expected_scale = min(640 / 480, 640 / 640)
        assert scale == pytest.approx(expected_scale)

        new_w = int(640 * expected_scale)
        new_h = int(480 * expected_scale)
        expected_pad_left = (640 - new_w) // 2
        expected_pad_top = (640 - new_h) // 2

        assert pad_left == expected_pad_left
        assert pad_top == expected_pad_top

        assert tensor.min() >= 0.0
        assert tensor.max() <= 1.0

        image_region = tensor[0, :, pad_top : pad_top + new_h, pad_left : pad_left + new_w]
        assert image_region.shape == (3, new_h, new_w)
        assert image_region.max() > 0.0

    def test_square_image_no_padding(self) -> None:
        preprocessor = YoloPreprocessor(target_size=640)
        image = np.ones((640, 640, 3), dtype=np.uint8) * 128

        tensor, scale, pad_offset = preprocessor.preprocess(image)

        assert scale == pytest.approx(1.0)
        assert pad_offset == (0, 0)
        assert tensor.shape == (1, 3, 640, 640)

    def test_bgr_to_rgb_conversion(self) -> None:
        preprocessor = YoloPreprocessor(target_size=4)
        image = np.zeros((4, 4, 3), dtype=np.uint8)
        image[:, :, 0] = 255
        image[:, :, 1] = 0
        image[:, :, 2] = 128

        tensor, _, _ = preprocessor.preprocess(image)

        np.testing.assert_allclose(tensor[0, 0, :, :], 128.0 / 255.0, atol=1e-5)
        np.testing.assert_allclose(tensor[0, 1, :, :], 0.0, atol=1e-5)
        np.testing.assert_allclose(tensor[0, 2, :, :], 1.0, atol=1e-5)

    def test_custom_target_size(self) -> None:
        preprocessor = YoloPreprocessor(target_size=320)
        image = np.zeros((240, 320, 3), dtype=np.uint8)

        tensor, _, _ = preprocessor.preprocess(image)

        assert tensor.shape == (1, 3, 320, 320)

    def test_nonsquare_target_size_tuple(self) -> None:
        preprocessor = YoloPreprocessor(target_size=(960, 640))
        image = np.zeros((480, 640, 3), dtype=np.uint8)

        tensor, scale, _pad_offset = preprocessor.preprocess(image)

        assert tensor.shape == (1, 3, 960, 640)
        assert tensor.dtype == np.float32
        expected_scale = min(960 / 480, 640 / 640)
        assert scale == pytest.approx(expected_scale)

    def test_nonsquare_target_size_padding(self) -> None:
        preprocessor = YoloPreprocessor(target_size=(384, 640))
        image = np.zeros((300, 400, 3), dtype=np.uint8)

        tensor, scale, (pad_left, pad_top) = preprocessor.preprocess(image)

        assert tensor.shape == (1, 3, 384, 640)
        new_w = int(400 * scale)
        new_h = int(300 * scale)
        expected_pad_left = (640 - new_w) // 2
        expected_pad_top = (384 - new_h) // 2
        assert pad_left == expected_pad_left
        assert pad_top == expected_pad_top

    @pytest.mark.parametrize(
        ("target_size", "expected_shape"),
        [
            (640, (1, 3, 640, 640)),
            ((960, 640), (1, 3, 960, 640)),
            ((384, 352), (1, 3, 384, 352)),
            (320, (1, 3, 320, 320)),
        ],
    )
    def test_parametrized_output_shapes(
        self,
        target_size: int | tuple[int, int],
        expected_shape: tuple[int, ...],
    ) -> None:
        preprocessor = YoloPreprocessor(target_size=target_size)
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        tensor, _, _ = preprocessor.preprocess(image)
        assert tensor.shape == expected_shape


class TestYoloPostprocessorEnd2End:
    def test_end2end_filters_by_confidence(self) -> None:
        postprocessor = YoloPostprocessor(end2end=True)
        raw = np.array(
            [
                [10, 20, 50, 60, 0.9, 0],
                [15, 25, 55, 65, 0.1, 1],
                [20, 30, 60, 70, 0.8, 2],
            ],
            dtype=np.float32,
        )
        outputs = {"output0": raw.reshape(1, 3, 6)}

        detections = postprocessor.postprocess(outputs, scale=1.0, pad_offset=(0, 0), conf_threshold=0.5)

        assert len(detections) == 2
        assert detections[0].confidence == pytest.approx(0.9)
        assert detections[1].confidence == pytest.approx(0.8)

    def test_end2end_rescales_coordinates(self) -> None:
        postprocessor = YoloPostprocessor(end2end=True)
        raw = np.array([[100, 100, 200, 200, 0.9, 0]], dtype=np.float32)
        outputs = {"output0": raw.reshape(1, 1, 6)}

        detections = postprocessor.postprocess(
            outputs,
            scale=0.5,
            pad_offset=(10, 20),
            conf_threshold=0.1,
        )

        assert len(detections) == 1
        det = detections[0]
        assert det.x1 == pytest.approx((100 - 10) / 0.5)
        assert det.y1 == pytest.approx((100 - 20) / 0.5)
        assert det.x2 == pytest.approx((200 - 10) / 0.5)
        assert det.y2 == pytest.approx((200 - 20) / 0.5)

    def test_end2end_empty_output(self) -> None:
        postprocessor = YoloPostprocessor(end2end=True)
        raw = np.array([[10, 20, 50, 60, 0.01, 0]], dtype=np.float32)
        outputs = {"output0": raw.reshape(1, 1, 6)}

        detections = postprocessor.postprocess(outputs, scale=1.0, pad_offset=(0, 0), conf_threshold=0.5)

        assert len(detections) == 0


class TestYoloPostprocessorStandard:
    def test_standard_detects_objects(self) -> None:
        postprocessor = YoloPostprocessor(end2end=False)

        num_classes = 80
        num_preds = 5
        features = 4 + num_classes

        raw = np.zeros((1, features, num_preds), dtype=np.float32)
        raw[0, 0, 0] = 320.0
        raw[0, 1, 0] = 320.0
        raw[0, 2, 0] = 100.0
        raw[0, 3, 0] = 100.0
        raw[0, 4, 0] = 0.9

        outputs = {"output0": raw}

        detections = postprocessor.postprocess(
            outputs,
            scale=1.0,
            pad_offset=(0, 0),
            conf_threshold=0.5,
        )

        assert len(detections) == 1
        assert detections[0].class_id == 0
        assert detections[0].confidence == pytest.approx(0.9)

    def test_standard_applies_nms(self) -> None:
        postprocessor = YoloPostprocessor(end2end=False)

        num_classes = 80
        num_preds = 3
        features = 4 + num_classes

        raw = np.zeros((1, features, num_preds), dtype=np.float32)
        for i in range(num_preds):
            raw[0, 0, i] = 320.0
            raw[0, 1, i] = 320.0
            raw[0, 2, i] = 100.0
            raw[0, 3, i] = 100.0
            raw[0, 4, i] = 0.9 - i * 0.1

        outputs = {"output0": raw}
        detections = postprocessor.postprocess(
            outputs,
            scale=1.0,
            pad_offset=(0, 0),
            conf_threshold=0.5,
            iou_threshold=0.5,
        )

        assert len(detections) == 1

    def test_standard_rescales_coordinates(self) -> None:
        postprocessor = YoloPostprocessor(end2end=False)
        num_classes = 80
        features = 4 + num_classes

        raw = np.zeros((1, features, 1), dtype=np.float32)
        raw[0, 0, 0] = 320.0
        raw[0, 1, 0] = 320.0
        raw[0, 2, 0] = 100.0
        raw[0, 3, 0] = 80.0
        raw[0, 4, 0] = 0.9

        outputs = {"output0": raw}
        detections = postprocessor.postprocess(
            outputs,
            scale=0.5,
            pad_offset=(10, 20),
            conf_threshold=0.1,
        )

        assert len(detections) == 1
        det = detections[0]

        box_x1 = 320.0 - 50.0
        box_y1 = 320.0 - 40.0
        box_x2 = 320.0 + 50.0
        box_y2 = 320.0 + 40.0

        assert det.x1 == pytest.approx((box_x1 - 10) / 0.5)
        assert det.y1 == pytest.approx((box_y1 - 20) / 0.5)
        assert det.x2 == pytest.approx((box_x2 - 10) / 0.5)
        assert det.y2 == pytest.approx((box_y2 - 20) / 0.5)

    def test_standard_no_detections_above_threshold(self) -> None:
        postprocessor = YoloPostprocessor(end2end=False)
        num_classes = 80
        features = 4 + num_classes
        raw = np.zeros((1, features, 1), dtype=np.float32)
        raw[0, 0, 0] = 320.0
        raw[0, 1, 0] = 320.0
        raw[0, 2, 0] = 100.0
        raw[0, 3, 0] = 100.0
        raw[0, 4, 0] = 0.1

        outputs = {"output0": raw}
        detections = postprocessor.postprocess(outputs, scale=1.0, pad_offset=(0, 0), conf_threshold=0.5)

        assert len(detections) == 0


class TestNmsNumpy:
    def test_suppresses_overlapping_boxes(self) -> None:
        boxes = np.array(
            [
                [0, 0, 100, 100],
                [5, 5, 105, 105],
                [200, 200, 300, 300],
            ],
            dtype=np.float32,
        )
        scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)

        keep = _nms_numpy(boxes, scores, iou_threshold=0.5)

        assert 0 in keep
        assert 1 not in keep
        assert 2 in keep

    def test_keeps_non_overlapping_boxes(self) -> None:
        boxes = np.array(
            [
                [0, 0, 50, 50],
                [100, 100, 150, 150],
            ],
            dtype=np.float32,
        )
        scores = np.array([0.9, 0.8], dtype=np.float32)

        keep = _nms_numpy(boxes, scores, iou_threshold=0.5)

        assert keep == [0, 1]

    def test_single_box(self) -> None:
        boxes = np.array([[10, 10, 50, 50]], dtype=np.float32)
        scores = np.array([0.9], dtype=np.float32)

        keep = _nms_numpy(boxes, scores, iou_threshold=0.5)

        assert keep == [0]

    def test_empty_input(self) -> None:
        boxes = np.zeros((0, 4), dtype=np.float32)
        scores = np.zeros(0, dtype=np.float32)

        keep = _nms_numpy(boxes, scores, iou_threshold=0.5)

        assert keep == []


class TestTritonInferenceClient:
    def test_infer_raw_calls_triton(self) -> None:
        mock_grpc = MagicMock()
        mock_client_instance = MagicMock()
        mock_grpc.InferenceServerClient.return_value = mock_client_instance

        mock_result = MagicMock()
        mock_result.as_numpy.return_value = np.ones((1, 84, 8400), dtype=np.float32)
        mock_client_instance.infer.return_value = mock_result

        with patch("triton_model_tools.inference.client.grpcclient", mock_grpc):
            client = TritonInferenceClient(url="localhost:8001")
            tensor = np.random.default_rng(42).random((1, 3, 640, 640)).astype(np.float32)
            results = client.infer_raw("yolo11s", tensor, ["output0"])

        assert "output0" in results

    def test_infer_dali_sends_bytes(self) -> None:
        mock_grpc = MagicMock()
        mock_client_instance = MagicMock()
        mock_grpc.InferenceServerClient.return_value = mock_client_instance

        mock_result = MagicMock()
        mock_result.as_numpy.return_value = np.ones((1, 84, 8400), dtype=np.float32)
        mock_client_instance.infer.return_value = mock_result

        with patch("triton_model_tools.inference.client.grpcclient", mock_grpc):
            client = TritonInferenceClient(url="localhost:8001")
            result = client.infer_dali("yolo11s_dali_ensemble", b"\xff\xd8\xff\xe0fake_jpeg_bytes")

        assert "OUTPUT" in result

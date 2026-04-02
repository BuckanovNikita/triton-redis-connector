"""Tests for detailed postprocessors with all-class scores."""

from __future__ import annotations

import numpy as np
import pytest
from triton_model_tools.inference.classification_postprocessing import (
    DetailedClassificationPostprocessor,
)
from triton_model_tools.inference.postprocessing import (
    YoloDetailedPostprocessor,
)
from triton_shared_models.classification import DetailedClassificationPrediction


class TestYoloDetailedPostprocessor:
    def test_attaches_all_class_scores_to_detections(self) -> None:
        postprocessor = YoloDetailedPostprocessor()

        num_classes = 80
        num_preds = 100
        features = 4 + num_classes

        raw = np.zeros((1, features, num_preds), dtype=np.float32)
        raw[0, 0, 0] = 320.0
        raw[0, 1, 0] = 320.0
        raw[0, 2, 0] = 100.0
        raw[0, 3, 0] = 100.0
        raw[0, 4, 0] = 0.9
        raw[0, 5, 0] = 0.1

        outputs = {"output0": raw}

        detections = postprocessor.postprocess(outputs, scale=1.0, pad_offset=(0, 0), conf_threshold=0.5)

        assert len(detections) == 1
        det = detections[0]
        assert det.all_class_scores is not None
        assert len(det.all_class_scores) == num_classes
        assert det.all_class_scores[0] == pytest.approx(0.9)
        assert det.all_class_scores[1] == pytest.approx(0.1)

    def test_class_scores_length_matches_num_classes(self) -> None:
        postprocessor = YoloDetailedPostprocessor()

        num_classes = 10
        features = 4 + num_classes
        raw = np.zeros((1, features, 50), dtype=np.float32)
        raw[0, 0, 0] = 100.0
        raw[0, 1, 0] = 100.0
        raw[0, 2, 0] = 50.0
        raw[0, 3, 0] = 50.0
        raw[0, 4, 0] = 0.8

        detections = postprocessor.postprocess({"output0": raw}, scale=1.0, pad_offset=(0, 0), conf_threshold=0.5)

        assert len(detections) == 1
        assert detections[0].all_class_scores is not None
        assert len(detections[0].all_class_scores) == num_classes

    def test_preserves_nms_behavior(self) -> None:
        postprocessor = YoloDetailedPostprocessor()

        num_classes = 80
        features = 4 + num_classes
        raw = np.zeros((1, features, 100), dtype=np.float32)
        for i in range(3):
            raw[0, 0, i] = 320.0
            raw[0, 1, i] = 320.0
            raw[0, 2, i] = 100.0
            raw[0, 3, i] = 100.0
            raw[0, 4, i] = 0.9 - i * 0.1

        detections = postprocessor.postprocess(
            {"output0": raw},
            scale=1.0,
            pad_offset=(0, 0),
            conf_threshold=0.5,
            iou_threshold=0.5,
        )

        assert len(detections) == 1
        assert detections[0].all_class_scores is not None

    def test_empty_output_returns_empty_list(self) -> None:
        postprocessor = YoloDetailedPostprocessor()

        num_classes = 80
        features = 4 + num_classes
        raw = np.zeros((1, features, 100), dtype=np.float32)
        raw[0, 0, 0] = 320.0
        raw[0, 1, 0] = 320.0
        raw[0, 2, 0] = 100.0
        raw[0, 3, 0] = 100.0
        raw[0, 4, 0] = 0.1

        detections = postprocessor.postprocess({"output0": raw}, scale=1.0, pad_offset=(0, 0), conf_threshold=0.5)

        assert len(detections) == 0

    def test_is_always_non_end2end(self) -> None:
        postprocessor = YoloDetailedPostprocessor()
        assert postprocessor.end2end is False

    def test_rescales_coordinates_with_all_scores(self) -> None:
        postprocessor = YoloDetailedPostprocessor()

        num_classes = 80
        features = 4 + num_classes
        raw = np.zeros((1, features, 100), dtype=np.float32)
        raw[0, 0, 0] = 320.0
        raw[0, 1, 0] = 320.0
        raw[0, 2, 0] = 100.0
        raw[0, 3, 0] = 80.0
        raw[0, 4, 0] = 0.9

        detections = postprocessor.postprocess(
            {"output0": raw},
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
        assert det.all_class_scores is not None
        assert len(det.all_class_scores) == num_classes


class TestDetailedClassificationPostprocessor:
    def test_returns_detailed_predictions_with_all_probabilities(self) -> None:
        postprocessor = DetailedClassificationPostprocessor(top_k=3, apply_softmax=False)

        probs = np.array([0.1, 0.5, 0.2, 0.15, 0.05], dtype=np.float32)
        outputs = {"output0": probs}

        predictions = postprocessor.postprocess(outputs)

        assert len(predictions) == 3
        for pred in predictions:
            assert isinstance(pred, DetailedClassificationPrediction)
            assert len(pred.all_probabilities) == 5

        assert predictions[0].class_id == 1
        assert predictions[0].probability == pytest.approx(0.5)

    def test_all_probabilities_are_identical_across_top_k(self) -> None:
        postprocessor = DetailedClassificationPostprocessor(top_k=2, apply_softmax=False)

        probs = np.array([0.3, 0.7], dtype=np.float32)
        predictions = postprocessor.postprocess({"output0": probs})

        assert predictions[0].all_probabilities == predictions[1].all_probabilities

    def test_softmax_applied_when_configured(self) -> None:
        postprocessor = DetailedClassificationPostprocessor(top_k=3, apply_softmax=True)

        logits = np.array([1.0, 2.0, 3.0, 0.5], dtype=np.float32)
        predictions = postprocessor.postprocess({"output0": logits})

        total = sum(predictions[0].all_probabilities)
        assert total == pytest.approx(1.0, abs=1e-5)

    def test_class_names_applied(self) -> None:
        postprocessor = DetailedClassificationPostprocessor(top_k=2, apply_softmax=False)

        probs = np.array([0.1, 0.9, 0.0], dtype=np.float32)
        class_names = {0: "cat", 1: "dog", 2: "bird"}

        predictions = postprocessor.postprocess({"output0": probs}, class_names=class_names)

        assert predictions[0].class_name == "dog"
        assert predictions[1].class_name == "cat"

    def test_batched_input_squeezed(self) -> None:
        postprocessor = DetailedClassificationPostprocessor(top_k=2, apply_softmax=False)

        probs = np.array([[0.3, 0.7]], dtype=np.float32)
        predictions = postprocessor.postprocess({"output0": probs})

        assert len(predictions) == 2
        assert predictions[0].class_id == 1

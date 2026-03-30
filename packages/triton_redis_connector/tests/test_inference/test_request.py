"""Tests for inference input/output event models."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError
from triton_redis_connector.inference import InferenceInput, InferenceOutput


class TestInferenceInput:
    def test_creation_with_defaults(self) -> None:
        tensor = np.zeros((1, 3, 640, 640), dtype=np.float32)
        event = InferenceInput(
            model_name="yolo11s",
            raw_image=tensor,
        )
        assert event.model_name == "yolo11s"
        assert event.model_version == ""
        assert event.input_name == "images"
        assert event.output_names == ("output0",)
        assert event.raw_image.shape == (1, 3, 640, 640)

    def test_creation_with_explicit_fields(self) -> None:
        tensor = np.ones((1, 3, 416, 416), dtype=np.float32)
        event = InferenceInput(
            model_name="yolo11m",
            model_version="2",
            raw_image=tensor,
            input_name="input_tensor",
            output_names=("output0", "output1"),
        )
        assert event.model_version == "2"
        assert event.input_name == "input_tensor"
        assert event.output_names == ("output0", "output1")

    def test_frozen_prevents_mutation(self) -> None:
        tensor = np.zeros((1, 3, 640, 640), dtype=np.float32)
        event = InferenceInput(
            model_name="yolo11s",
            raw_image=tensor,
        )
        with pytest.raises(ValidationError):
            event.model_name = "modified"  # type: ignore[misc]

    def test_numpy_array_preserved(self) -> None:
        tensor = np.random.default_rng(42).random((1, 3, 640, 640), dtype=np.float32)
        event = InferenceInput(
            model_name="yolo11s",
            raw_image=tensor,
        )
        np.testing.assert_array_equal(event.raw_image, tensor)
        assert event.raw_image.dtype == np.float32

    def test_missing_required_field_raises(self) -> None:
        with pytest.raises(ValidationError):
            InferenceInput(  # type: ignore[call-arg]
                model_version="1",
            )


class TestInferenceOutput:
    def test_creation(self) -> None:
        output_data = {
            "output0": np.zeros((1, 84, 8400), dtype=np.float32),
        }
        event = InferenceOutput(
            model_name="yolo11s",
            model_version="1",
            outputs=output_data,
            inference_time_ms=12.5,
        )
        assert event.model_name == "yolo11s"
        assert event.model_version == "1"
        assert event.inference_time_ms == 12.5
        assert "output0" in event.outputs
        assert event.outputs["output0"].shape == (1, 84, 8400)

    def test_multiple_outputs(self) -> None:
        output_data = {
            "output0": np.zeros((1, 84, 8400), dtype=np.float32),
            "output1": np.zeros((1, 100, 4), dtype=np.float32),
        }
        event = InferenceOutput(
            model_name="yolo11s",
            model_version="1",
            outputs=output_data,
            inference_time_ms=15.0,
        )
        assert len(event.outputs) == 2

    def test_default_model_version(self) -> None:
        event = InferenceOutput(
            model_name="yolo11s",
            model_version="",
            outputs={},
            inference_time_ms=0.0,
        )
        assert event.model_version == ""

    def test_frozen_prevents_mutation(self) -> None:
        event = InferenceOutput(
            model_name="yolo11s",
            model_version="1",
            outputs={},
            inference_time_ms=10.0,
        )
        with pytest.raises(ValidationError):
            event.model_name = "modified"  # type: ignore[misc]

    def test_missing_required_field_raises(self) -> None:
        with pytest.raises(ValidationError):
            InferenceOutput(  # type: ignore[call-arg]
                model_name="yolo11s",
                inference_time_ms=10.0,
            )

"""Tests for TritonRequest/InferenceOutput event models."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError
from triton_redis_connector.inference import InferenceOutput, TritonRequest


class TestTritonRequest:
    def test_numpy_array_preserved(self) -> None:
        tensor = np.random.default_rng(42).random((1, 3, 640, 640), dtype=np.float32)
        event = TritonRequest(
            model_name="yolo11s",
            raw_image=tensor,
        )
        assert event.raw_image is not None
        np.testing.assert_array_equal(event.raw_image, tensor)
        assert event.raw_image.dtype == np.float32

    def test_image_bytes_mode(self) -> None:
        event = TritonRequest(
            model_name="yolo11s",
            image_bytes=b"\xff\xd8\xff",
        )
        assert event.image_bytes == b"\xff\xd8\xff"
        assert event.raw_image is None

    def test_both_modes_raises(self) -> None:
        tensor = np.zeros((1, 3, 640, 640), dtype=np.float32)
        with pytest.raises(ValidationError, match="Exactly one"):
            TritonRequest(
                model_name="yolo11s",
                raw_image=tensor,
                image_bytes=b"\xff\xd8\xff",
            )

    def test_neither_mode_raises(self) -> None:
        with pytest.raises(ValidationError, match="Exactly one"):
            TritonRequest(model_name="yolo11s")

    def test_missing_required_field_raises(self) -> None:
        with pytest.raises(ValidationError):
            TritonRequest(  # type: ignore[call-arg]
                model_version="1",
            )


class TestInferenceOutput:
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

    def test_missing_required_field_raises(self) -> None:
        with pytest.raises(ValidationError):
            InferenceOutput(  # type: ignore[call-arg]
                model_name="yolo11s",
                inference_time_ms=10.0,
            )

"""Tests for shared Triton type definitions."""

from __future__ import annotations

from triton_shared_models.triton_config import (
    ModelWarmupConfig,
    WarmupInputSpec,
)
from triton_shared_models.triton_types import (
    TritonDataType,
)


class TestModelWarmupConfig:
    def test_creates_warmup_config(self) -> None:
        warmup = ModelWarmupConfig(
            name="warmup_request",
            batch_size=1,
            inputs={
                "images": WarmupInputSpec(
                    data_type=TritonDataType.TYPE_FP32,
                    dims=[3, 640, 640],
                ),
            },
        )
        assert warmup.name == "warmup_request"
        assert warmup.batch_size == 1
        assert "images" in warmup.inputs
        assert warmup.inputs["images"].zero_data is True

    def test_warmup_input_spec_random_data(self) -> None:
        spec = WarmupInputSpec(
            data_type=TritonDataType.TYPE_FP32,
            dims=[3, 640, 640],
            zero_data=False,
        )
        assert spec.zero_data is False

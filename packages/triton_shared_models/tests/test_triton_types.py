"""Tests for shared Triton type definitions."""

from __future__ import annotations

from triton_shared_models.triton_config import TritonModelConfig
from triton_shared_models.triton_types import (
    TritonDataType,
    TritonInstanceKind,
    TritonTensorSpec,
)


class TestTritonDataType:
    def test_fp32_value(self) -> None:
        assert TritonDataType.TYPE_FP32 == "TYPE_FP32"

    def test_fp16_value(self) -> None:
        assert TritonDataType.TYPE_FP16 == "TYPE_FP16"

    def test_int8_value(self) -> None:
        assert TritonDataType.TYPE_INT8 == "TYPE_INT8"

    def test_int32_value(self) -> None:
        assert TritonDataType.TYPE_INT32 == "TYPE_INT32"

    def test_int64_value(self) -> None:
        assert TritonDataType.TYPE_INT64 == "TYPE_INT64"

    def test_uint8_value(self) -> None:
        assert TritonDataType.TYPE_UINT8 == "TYPE_UINT8"

    def test_bool_value(self) -> None:
        assert TritonDataType.TYPE_BOOL == "TYPE_BOOL"

    def test_all_members_count(self) -> None:
        assert len(TritonDataType) == 8


class TestTritonInstanceKind:
    def test_gpu_value(self) -> None:
        assert TritonInstanceKind.KIND_GPU == "KIND_GPU"

    def test_cpu_value(self) -> None:
        assert TritonInstanceKind.KIND_CPU == "KIND_CPU"

    def test_all_members_count(self) -> None:
        assert len(TritonInstanceKind) == 2


class TestTritonTensorSpec:
    def test_creates_tensor_spec(self) -> None:
        spec = TritonTensorSpec(
            name="images",
            data_type=TritonDataType.TYPE_FP32,
            dims=[1, 3, 640, 640],
        )
        assert spec.name == "images"
        assert spec.data_type == TritonDataType.TYPE_FP32
        assert spec.dims == [1, 3, 640, 640]

    def test_creates_spec_with_dynamic_dims(self) -> None:
        spec = TritonTensorSpec(
            name="output",
            data_type=TritonDataType.TYPE_INT64,
            dims=[-1, 6],
        )
        assert spec.dims == [-1, 6]

    def test_creates_spec_with_uint8(self) -> None:
        spec = TritonTensorSpec(
            name="raw_input",
            data_type=TritonDataType.TYPE_UINT8,
            dims=[1, 640, 640, 3],
        )
        assert spec.data_type == TritonDataType.TYPE_UINT8


class TestTritonModelConfig:
    def _make_simple_config(
        self,
        name: str = "test_model",
        **kwargs: object,
    ) -> TritonModelConfig:
        defaults: dict[str, object] = {
            "name": name,
            "inputs": [
                TritonTensorSpec(
                    name="images",
                    data_type=TritonDataType.TYPE_FP32,
                    dims=[1, 3, 640, 640],
                ),
            ],
            "outputs": [
                TritonTensorSpec(
                    name="output0",
                    data_type=TritonDataType.TYPE_FP32,
                    dims=[1, 300, 6],
                ),
            ],
        }
        defaults.update(kwargs)
        return TritonModelConfig(**defaults)  # type: ignore[arg-type]

    def test_defaults(self) -> None:
        config = self._make_simple_config()
        assert config.platform == "onnxruntime_onnx"
        assert config.max_batch_size == 0
        assert config.instance_count == 1
        assert config.instance_kind == TritonInstanceKind.KIND_GPU

    def test_to_pbtxt_contains_name(self) -> None:
        pbtxt = self._make_simple_config().to_pbtxt()
        assert 'name: "test_model"' in pbtxt

    def test_to_pbtxt_contains_platform(self) -> None:
        pbtxt = self._make_simple_config().to_pbtxt()
        assert 'platform: "onnxruntime_onnx"' in pbtxt

    def test_to_pbtxt_contains_max_batch_size(self) -> None:
        pbtxt = self._make_simple_config().to_pbtxt()
        assert "max_batch_size: 0" in pbtxt

    def test_to_pbtxt_contains_input_tensor(self) -> None:
        pbtxt = self._make_simple_config().to_pbtxt()
        assert 'name: "images"' in pbtxt
        assert "data_type: TYPE_FP32" in pbtxt
        assert "dims: [ 1, 3, 640, 640 ]" in pbtxt

    def test_to_pbtxt_contains_output_tensor(self) -> None:
        pbtxt = self._make_simple_config().to_pbtxt()
        assert 'name: "output0"' in pbtxt
        assert "dims: [ 1, 300, 6 ]" in pbtxt

    def test_to_pbtxt_contains_instance_group(self) -> None:
        pbtxt = self._make_simple_config().to_pbtxt()
        assert "kind: KIND_GPU" in pbtxt
        assert "count: 1" in pbtxt

    def test_to_pbtxt_with_cpu_instance(self) -> None:
        config = self._make_simple_config(
            instance_kind=TritonInstanceKind.KIND_CPU,
            instance_count=2,
        )
        pbtxt = config.to_pbtxt()
        assert "kind: KIND_CPU" in pbtxt
        assert "count: 2" in pbtxt

    def test_to_pbtxt_multiple_outputs(self) -> None:
        config = self._make_simple_config(
            outputs=[
                TritonTensorSpec(
                    name="output0",
                    data_type=TritonDataType.TYPE_FP32,
                    dims=[1, 84, 8400],
                ),
                TritonTensorSpec(
                    name="output1",
                    data_type=TritonDataType.TYPE_FP32,
                    dims=[1, 32, 160, 160],
                ),
            ],
        )
        pbtxt = config.to_pbtxt()
        assert pbtxt.count("output [") == 2
        assert 'name: "output0"' in pbtxt
        assert 'name: "output1"' in pbtxt

"""Tests for shared Triton type definitions."""

from __future__ import annotations

from triton_shared_models.triton_config import (
    DynamicBatchingConfig,
    EnsembleStep,
    EnsembleStepMapping,
    ModelWarmupConfig,
    OptimizationConfig,
    TritonModelConfig,
    WarmupInputSpec,
)
from triton_shared_models.triton_types import (
    EXPORT_FORMAT_TO_FILENAME,
    EXPORT_FORMAT_TO_PLATFORM,
    ExportFormat,
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


class TestExportFormat:
    def test_onnx_value(self) -> None:
        assert ExportFormat.ONNX == "onnx"

    def test_tensorrt_value(self) -> None:
        assert ExportFormat.TENSORRT == "tensorrt"

    def test_all_members_count(self) -> None:
        assert len(ExportFormat) == 2

    def test_is_str_enum(self) -> None:
        assert isinstance(ExportFormat.ONNX, str)


class TestExportFormatMappings:
    def test_onnx_platform(self) -> None:
        assert EXPORT_FORMAT_TO_PLATFORM[ExportFormat.ONNX] == "onnxruntime_onnx"

    def test_tensorrt_platform(self) -> None:
        assert EXPORT_FORMAT_TO_PLATFORM[ExportFormat.TENSORRT] == "tensorrt_plan"

    def test_onnx_filename(self) -> None:
        assert EXPORT_FORMAT_TO_FILENAME[ExportFormat.ONNX] == "model.onnx"

    def test_tensorrt_filename(self) -> None:
        assert EXPORT_FORMAT_TO_FILENAME[ExportFormat.TENSORRT] == "model.plan"

    def test_platform_mapping_covers_all_formats(self) -> None:
        assert set(EXPORT_FORMAT_TO_PLATFORM.keys()) == set(ExportFormat)

    def test_filename_mapping_covers_all_formats(self) -> None:
        assert set(EXPORT_FORMAT_TO_FILENAME.keys()) == set(ExportFormat)


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

    def test_is_shape_tensor_default_false(self) -> None:
        spec = TritonTensorSpec(
            name="data",
            data_type=TritonDataType.TYPE_FP32,
            dims=[1],
        )
        assert spec.is_shape_tensor is False

    def test_is_shape_tensor_set_true(self) -> None:
        spec = TritonTensorSpec(
            name="shape_input",
            data_type=TritonDataType.TYPE_INT32,
            dims=[2],
            is_shape_tensor=True,
        )
        assert spec.is_shape_tensor is True


def _make_simple_config(
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


class TestTritonModelConfig:
    def test_defaults(self) -> None:
        config = _make_simple_config()
        assert config.platform == "onnxruntime_onnx"
        assert config.max_batch_size == 0
        assert config.instance_count == 1
        assert config.instance_kind == TritonInstanceKind.KIND_GPU
        assert config.default_model_filename == ""
        assert config.dynamic_batching is None
        assert config.optimization is None
        assert config.model_warmup is None

    def test_to_pbtxt_contains_name(self) -> None:
        pbtxt = _make_simple_config().to_pbtxt()
        assert 'name: "test_model"' in pbtxt

    def test_to_pbtxt_contains_platform(self) -> None:
        pbtxt = _make_simple_config().to_pbtxt()
        assert 'platform: "onnxruntime_onnx"' in pbtxt

    def test_to_pbtxt_contains_max_batch_size(self) -> None:
        pbtxt = _make_simple_config().to_pbtxt()
        assert "max_batch_size: 0" in pbtxt

    def test_to_pbtxt_contains_input_tensor(self) -> None:
        pbtxt = _make_simple_config().to_pbtxt()
        assert 'name: "images"' in pbtxt
        assert "data_type: TYPE_FP32" in pbtxt
        assert "dims: [ 1, 3, 640, 640 ]" in pbtxt

    def test_to_pbtxt_contains_output_tensor(self) -> None:
        pbtxt = _make_simple_config().to_pbtxt()
        assert 'name: "output0"' in pbtxt
        assert "dims: [ 1, 300, 6 ]" in pbtxt

    def test_to_pbtxt_contains_instance_group(self) -> None:
        pbtxt = _make_simple_config().to_pbtxt()
        assert "kind: KIND_GPU" in pbtxt
        assert "count: 1" in pbtxt

    def test_to_pbtxt_with_cpu_instance(self) -> None:
        config = _make_simple_config(
            instance_kind=TritonInstanceKind.KIND_CPU,
            instance_count=2,
        )
        pbtxt = config.to_pbtxt()
        assert "kind: KIND_CPU" in pbtxt
        assert "count: 2" in pbtxt

    def test_to_pbtxt_multiple_outputs(self) -> None:
        config = _make_simple_config(
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

    def test_to_pbtxt_no_default_filename_when_empty(self) -> None:
        pbtxt = _make_simple_config().to_pbtxt()
        assert "default_model_filename" not in pbtxt

    def test_to_pbtxt_with_default_model_filename(self) -> None:
        config = _make_simple_config(default_model_filename="model.plan")
        pbtxt = config.to_pbtxt()
        assert 'default_model_filename: "model.plan"' in pbtxt

    def test_to_pbtxt_with_is_shape_tensor(self) -> None:
        config = _make_simple_config(
            inputs=[
                TritonTensorSpec(
                    name="shape_input",
                    data_type=TritonDataType.TYPE_INT32,
                    dims=[2],
                    is_shape_tensor=True,
                ),
            ],
        )
        pbtxt = config.to_pbtxt()
        assert "is_shape_tensor: true" in pbtxt

    def test_to_pbtxt_no_is_shape_tensor_when_false(self) -> None:
        pbtxt = _make_simple_config().to_pbtxt()
        assert "is_shape_tensor" not in pbtxt


class TestDynamicBatchingConfig:
    def test_defaults(self) -> None:
        config = DynamicBatchingConfig()
        assert config.max_queue_delay_microseconds == 0
        assert config.preferred_batch_size == []

    def test_custom_values(self) -> None:
        config = DynamicBatchingConfig(
            max_queue_delay_microseconds=100,
            preferred_batch_size=[1, 4, 8],
        )
        assert config.max_queue_delay_microseconds == 100
        assert config.preferred_batch_size == [1, 4, 8]

    def test_to_pbtxt_renders_dynamic_batching(self) -> None:
        config = _make_simple_config(
            dynamic_batching=DynamicBatchingConfig(
                max_queue_delay_microseconds=200,
                preferred_batch_size=[1, 4],
            ),
        )
        pbtxt = config.to_pbtxt()
        assert "dynamic_batching {" in pbtxt
        assert "max_queue_delay_microseconds: 200" in pbtxt
        assert "preferred_batch_size: [ 1, 4 ]" in pbtxt

    def test_to_pbtxt_empty_dynamic_batching(self) -> None:
        config = _make_simple_config(
            dynamic_batching=DynamicBatchingConfig(),
        )
        pbtxt = config.to_pbtxt()
        assert "dynamic_batching {" in pbtxt
        assert "max_queue_delay_microseconds" not in pbtxt
        assert "preferred_batch_size" not in pbtxt

    def test_to_pbtxt_no_dynamic_batching_when_none(self) -> None:
        pbtxt = _make_simple_config().to_pbtxt()
        assert "dynamic_batching" not in pbtxt


class TestOptimizationConfig:
    def test_defaults(self) -> None:
        config = OptimizationConfig()
        assert config.graph_level == -1
        assert config.cuda_graphs is False

    def test_custom_values(self) -> None:
        config = OptimizationConfig(graph_level=3, cuda_graphs=True)
        assert config.graph_level == 3
        assert config.cuda_graphs is True

    def test_to_pbtxt_renders_optimization(self) -> None:
        config = _make_simple_config(
            optimization=OptimizationConfig(graph_level=1, cuda_graphs=True),
        )
        pbtxt = config.to_pbtxt()
        assert "optimization {" in pbtxt
        assert "graph_level: 1" in pbtxt
        assert "cuda { graphs: true }" in pbtxt

    def test_to_pbtxt_default_optimization(self) -> None:
        config = _make_simple_config(
            optimization=OptimizationConfig(),
        )
        pbtxt = config.to_pbtxt()
        assert "optimization {" in pbtxt
        assert "graph_level" not in pbtxt.split("optimization {")[1].split("}")[0]
        assert "cuda" not in pbtxt.split("optimization {")[1].split("}")[0]

    def test_to_pbtxt_no_optimization_when_none(self) -> None:
        pbtxt = _make_simple_config().to_pbtxt()
        assert "optimization" not in pbtxt


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

    def test_to_pbtxt_renders_model_warmup(self) -> None:
        config = _make_simple_config(
            model_warmup=[
                ModelWarmupConfig(
                    name="warmup_images",
                    batch_size=1,
                    inputs={
                        "images": WarmupInputSpec(
                            data_type=TritonDataType.TYPE_FP32,
                            dims=[3, 640, 640],
                        ),
                    },
                ),
            ],
        )
        pbtxt = config.to_pbtxt()
        assert "model_warmup [" in pbtxt
        assert 'name: "warmup_images"' in pbtxt
        assert "batch_size: 1" in pbtxt
        assert 'key: "images"' in pbtxt
        assert "data_type: TYPE_FP32" in pbtxt
        assert "dims: [ 3, 640, 640 ]" in pbtxt
        assert "zero_data {}" in pbtxt

    def test_to_pbtxt_random_data_warmup(self) -> None:
        config = _make_simple_config(
            model_warmup=[
                ModelWarmupConfig(
                    name="warmup_random",
                    batch_size=2,
                    inputs={
                        "images": WarmupInputSpec(
                            data_type=TritonDataType.TYPE_FP32,
                            dims=[3, 640, 640],
                            zero_data=False,
                        ),
                    },
                ),
            ],
        )
        pbtxt = config.to_pbtxt()
        assert "random_data {}" in pbtxt
        assert "zero_data" not in pbtxt

    def test_to_pbtxt_no_warmup_when_none(self) -> None:
        pbtxt = _make_simple_config().to_pbtxt()
        assert "model_warmup" not in pbtxt

    def test_to_pbtxt_no_warmup_when_empty_list(self) -> None:
        config = _make_simple_config(model_warmup=[])
        pbtxt = config.to_pbtxt()
        assert "model_warmup" not in pbtxt


class TestPbtxtCombinedFields:
    def test_all_new_fields_together(self) -> None:
        config = _make_simple_config(
            default_model_filename="model.plan",
            dynamic_batching=DynamicBatchingConfig(
                max_queue_delay_microseconds=500,
                preferred_batch_size=[1, 8],
            ),
            optimization=OptimizationConfig(graph_level=2, cuda_graphs=True),
            model_warmup=[
                ModelWarmupConfig(
                    name="warmup_req",
                    batch_size=1,
                    inputs={
                        "images": WarmupInputSpec(
                            data_type=TritonDataType.TYPE_FP32,
                            dims=[3, 640, 640],
                        ),
                    },
                ),
            ],
        )
        pbtxt = config.to_pbtxt()
        assert 'default_model_filename: "model.plan"' in pbtxt
        assert "dynamic_batching {" in pbtxt
        assert "max_queue_delay_microseconds: 500" in pbtxt
        assert "optimization {" in pbtxt
        assert "graph_level: 2" in pbtxt
        assert "model_warmup [" in pbtxt

    def test_to_pbtxt_renders_multiple_warmup_inputs(self) -> None:
        config = _make_simple_config(
            model_warmup=[
                ModelWarmupConfig(
                    name="warmup_multi",
                    batch_size=1,
                    inputs={
                        "images": WarmupInputSpec(
                            data_type=TritonDataType.TYPE_FP32,
                            dims=[3, 640, 640],
                        ),
                        "shapes": WarmupInputSpec(
                            data_type=TritonDataType.TYPE_INT32,
                            dims=[2],
                            zero_data=False,
                        ),
                    },
                ),
            ],
        )
        pbtxt = config.to_pbtxt()
        assert pbtxt.count("inputs {") == 2
        assert 'key: "images"' in pbtxt
        assert 'key: "shapes"' in pbtxt
        assert "random_data {}" in pbtxt
        assert "zero_data {}" in pbtxt

    def test_ensemble_scheduling_single_step_block(self) -> None:
        config = _make_simple_config(
            ensemble_scheduling=[
                EnsembleStep(
                    model_name="preprocess",
                    model_version=-1,
                    input_map=[EnsembleStepMapping(key="INPUT_0", value="RAW")],
                    output_map=[EnsembleStepMapping(key="OUTPUT_0", value="PROCESSED")],
                ),
                EnsembleStep(
                    model_name="inference",
                    model_version=-1,
                    input_map=[EnsembleStepMapping(key="images", value="PROCESSED")],
                    output_map=[EnsembleStepMapping(key="output0", value="RESULT")],
                ),
            ],
        )
        pbtxt = config.to_pbtxt()
        assert pbtxt.count("step [") == 1
        assert "ensemble_scheduling {" in pbtxt
        assert 'model_name: "preprocess"' in pbtxt
        assert 'model_name: "inference"' in pbtxt
        assert "instance_group" not in pbtxt

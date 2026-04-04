"""Integration tests for generated config.pbtxt validation against Triton."""

from __future__ import annotations

import contextlib
import shutil
import time
from typing import TYPE_CHECKING

import pytest
from triton_model_tools.triton_config import (
    DynamicBatchingConfig,
    TritonModelConfig,
)
from triton_shared_models.triton_types import (
    TritonDataType,
    TritonInstanceKind,
    TritonTensorSpec,
)

if TYPE_CHECKING:
    from pathlib import Path

    import tritonclient.grpc as grpcclient

pytestmark = [pytest.mark.integration, pytest.mark.xdist_group(name="integration")]


def _build_identity_config(name: str = "identity") -> TritonModelConfig:
    return TritonModelConfig(
        name=name,
        platform="onnxruntime_onnx",
        max_batch_size=0,
        inputs=[
            TritonTensorSpec(
                name="INPUT",
                data_type=TritonDataType.TYPE_FP32,
                dims=[-1],
            ),
        ],
        outputs=[
            TritonTensorSpec(
                name="OUTPUT",
                data_type=TritonDataType.TYPE_FP32,
                dims=[-1],
            ),
        ],
        instance_count=1,
        instance_kind=TritonInstanceKind.KIND_CPU,
        default_model_filename="model.onnx",
        dynamic_batching=DynamicBatchingConfig(),
    )


class TestGeneratedConfigValidation:
    def test_triton_loads_generated_config(
        self,
        model_repository_path: Path,
        triton_grpc_client: grpcclient.InferenceServerClient,
    ) -> None:
        test_model_name = "test_identity_generated"
        test_model_dir = model_repository_path / test_model_name
        try:
            version_dir = test_model_dir / "1"
            version_dir.mkdir(parents=True, exist_ok=True)

            source_onnx = model_repository_path / "identity" / "1" / "model.onnx"
            shutil.copy2(source_onnx, version_dir / "model.onnx")

            config = _build_identity_config(name=test_model_name)
            (test_model_dir / "config.pbtxt").write_text(config.to_pbtxt())

            ready = False
            for _ in range(15):
                time.sleep(2)
                with contextlib.suppress(Exception):
                    ready = triton_grpc_client.is_model_ready(test_model_name)
                if ready:
                    break

            assert ready, f"Triton did not load {test_model_name} within 30 seconds"
        finally:
            if test_model_dir.exists():
                shutil.rmtree(test_model_dir)

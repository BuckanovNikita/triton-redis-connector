"""Integration tests for identity model inference round-trip via gRPC."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    import tritonclient.grpc as grpcclient

pytestmark = [pytest.mark.integration, pytest.mark.xdist_group(name="integration")]


def _infer_identity(
    client: grpcclient.InferenceServerClient,
    data: np.ndarray,
) -> np.ndarray:
    import tritonclient.grpc as grpcclient

    infer_input = grpcclient.InferInput("INPUT", list(data.shape), "FP32")
    infer_input.set_data_from_numpy(data)
    infer_output = grpcclient.InferRequestedOutput("OUTPUT")

    result = client.infer(model_name="identity", inputs=[infer_input], outputs=[infer_output])
    return result.as_numpy("OUTPUT")


class TestIdentityRoundtrip:
    def test_identity_scalar_input(
        self,
        triton_grpc_client: grpcclient.InferenceServerClient,
    ) -> None:
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        output = _infer_identity(triton_grpc_client, data)
        np.testing.assert_array_equal(output, data)

    def test_identity_large_tensor(
        self,
        triton_grpc_client: grpcclient.InferenceServerClient,
    ) -> None:
        rng = np.random.default_rng(42)
        data = rng.random(1000, dtype=np.float32)
        output = _infer_identity(triton_grpc_client, data)
        np.testing.assert_array_equal(output, data)

    def test_identity_zeros(
        self,
        triton_grpc_client: grpcclient.InferenceServerClient,
    ) -> None:
        data = np.zeros(10, dtype=np.float32)
        output = _infer_identity(triton_grpc_client, data)
        np.testing.assert_array_equal(output, data)

    def test_identity_single_element(
        self,
        triton_grpc_client: grpcclient.InferenceServerClient,
    ) -> None:
        data = np.array([42.0], dtype=np.float32)
        output = _infer_identity(triton_grpc_client, data)
        np.testing.assert_array_equal(output, data)

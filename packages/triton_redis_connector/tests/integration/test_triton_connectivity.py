"""Integration tests for Triton Inference Server connectivity."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    import tritonclient.grpc as grpcclient

pytestmark = [pytest.mark.integration, pytest.mark.xdist_group(name="integration")]


class TestTritonConnectivity:
    def test_server_is_ready(self, triton_grpc_client: grpcclient.InferenceServerClient) -> None:
        assert triton_grpc_client.is_server_ready()

    def test_server_is_live(self, triton_grpc_client: grpcclient.InferenceServerClient) -> None:
        assert triton_grpc_client.is_server_live()

    def test_model_repository_contains_identity(
        self,
        triton_grpc_client: grpcclient.InferenceServerClient,
    ) -> None:
        repo_index = triton_grpc_client.get_model_repository_index()
        model_names = [m.name for m in repo_index.models]
        assert "identity" in model_names

    def test_identity_model_is_ready(
        self,
        triton_grpc_client: grpcclient.InferenceServerClient,
    ) -> None:
        assert triton_grpc_client.is_model_ready("identity")

    def test_identity_model_metadata(
        self,
        triton_grpc_client: grpcclient.InferenceServerClient,
    ) -> None:
        metadata = triton_grpc_client.get_model_metadata("identity")
        assert metadata.name == "identity"

        input_names = [inp.name for inp in metadata.inputs]
        assert "INPUT" in input_names

        output_names = [out.name for out in metadata.outputs]
        assert "OUTPUT" in output_names

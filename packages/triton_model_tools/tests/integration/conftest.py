"""Fixtures for Triton config validation integration tests."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

    import tritonclient.grpc as grpcclient


@pytest.fixture(scope="session")
def triton_grpc_url() -> str:
    return os.environ.get("TRITON_GRPC_URL", "localhost:8001")


@pytest.fixture(scope="session")
def model_repository_path() -> Path:
    from pathlib import Path

    repo = Path(__file__).resolve().parents[4] / "model_repository"
    if not repo.is_dir():
        pytest.skip("model_repository not found")
    return repo


@pytest.fixture(scope="session")
def triton_grpc_client(triton_grpc_url: str) -> grpcclient.InferenceServerClient:
    import tritonclient.grpc as grpcclient

    client = grpcclient.InferenceServerClient(url=triton_grpc_url)
    try:
        if not client.is_server_ready():
            pytest.skip("Triton server not available")
    except Exception:  # noqa: BLE001
        pytest.skip("Triton server not reachable")
    return client

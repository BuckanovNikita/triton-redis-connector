"""Shared fixtures for triton_model_tools tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest


@pytest.fixture
def coco_image_dir() -> Path:
    """Return path to COCO dataset images; skip test if unavailable."""
    candidates = [
        Path("datasets/coco8/images/val"),
        Path("datasets/coco8/images/train"),
        Path.home() / "datasets" / "coco8" / "images" / "val",
    ]
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.iterdir()):
            return candidate
    pytest.skip("COCO image directory not found")


@pytest.fixture
def triton_client() -> Any:
    """Return a connected Triton gRPC client; skip test if Triton is down."""
    try:
        import tritonclient.grpc as grpcclient  # pyright: ignore[reportMissingTypeStubs]

        client: Any = grpcclient.InferenceServerClient(url="localhost:8001")
        if not client.is_server_live():
            pytest.skip("Triton server is not live")
    except Exception:  # noqa: BLE001
        pytest.skip("Cannot connect to Triton server")
    else:
        return client


@pytest.fixture
def redis_client() -> Any:
    """Return a connected Redis client; skip test if Redis is down."""
    try:
        import redis

        client: Any = redis.Redis(host="localhost", port=6379, decode_responses=False)
        client.ping()
    except Exception:  # noqa: BLE001
        pytest.skip("Cannot connect to Redis server")
    else:
        return client

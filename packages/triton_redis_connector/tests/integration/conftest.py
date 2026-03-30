"""Fixtures for integration tests requiring Docker services (Redis, Triton)."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import pytest
import redis.asyncio

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    import tritonclient.grpc as grpcclient


@pytest.fixture(scope="session")
def redis_host() -> str:
    return os.environ.get("REDIS_HOST", "localhost")


@pytest.fixture(scope="session")
def redis_port() -> int:
    return int(os.environ.get("REDIS_PORT", "6379"))


@pytest.fixture(scope="session")
def triton_grpc_url() -> str:
    return os.environ.get("TRITON_GRPC_URL", "localhost:8001")


@pytest.fixture(scope="session")
async def redis_client(
    redis_host: str,
    redis_port: int,
) -> AsyncIterator[redis.asyncio.Redis[str]]:
    client: redis.asyncio.Redis[str] = redis.asyncio.Redis(
        host=redis_host,
        port=redis_port,
        decode_responses=True,
    )
    try:
        await client.ping()
    except (redis.exceptions.ConnectionError, OSError):
        pytest.skip("Redis not available")
    yield client
    await client.aclose()


@pytest.fixture(autouse=True)
async def _clean_test_streams(
    redis_client: redis.asyncio.Redis[str],
) -> AsyncIterator[None]:
    yield
    keys: Any = await redis_client.keys("test:*")
    if keys:
        await redis_client.delete(*keys)


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

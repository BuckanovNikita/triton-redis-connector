"""Pure inference processor for Triton gRPC requests.

Business logic only — no threading, no retry, no semaphore.
Concurrency and retry are handled by framework wrappers and middleware.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import numpy as np
import tritonclient.grpc as grpcclient

from triton_redis_connector.errors.models import InferenceError  # noqa: TC001 — needed at runtime for get_type_hints
from triton_redis_connector.inference.request import InferenceInput, InferenceOutput

if TYPE_CHECKING:
    from triton_redis_connector.config.schema import InferenceConfig


class InferenceWorker:
    """Processor that sends a single inference request to Triton.

    Creates its own tritonclient.grpc.InferenceServerClient on construction.
    Raises on transient failures — retry is handled by RetryMiddleware.
    """

    def __init__(self, grpc_url: str) -> None:  # noqa: D107
        self._client: grpcclient.InferenceServerClient = grpcclient.InferenceServerClient(url=grpc_url)

    def __call__(self, event: InferenceInput) -> InferenceOutput | InferenceError | None:
        """Run inference on a single InferenceInput event."""
        start = time.monotonic()

        if event.raw_image is not None:
            input_tensor = grpcclient.InferInput(event.input_name, event.raw_image.shape, "FP32")
            input_tensor.set_data_from_numpy(event.raw_image)
        elif event.image_bytes is not None:
            data = np.frombuffer(event.image_bytes, dtype=np.uint8)
            input_tensor = grpcclient.InferInput(event.input_name, [len(data)], "UINT8")
            input_tensor.set_data_from_numpy(data)
        else:
            msg = "Neither raw_image nor image_bytes set on InferenceInput"
            raise ValueError(msg)
        outputs = [grpcclient.InferRequestedOutput(name) for name in event.output_names]

        result = self._client.infer(
            model_name=event.model_name,
            inputs=[input_tensor],
            outputs=outputs,
            model_version=event.model_version or "",
        )
        if result is None:
            msg = f"Triton returned None for model {event.model_name}"
            raise RuntimeError(msg)

        elapsed_ms = (time.monotonic() - start) * 1000.0

        output_arrays: dict[str, np.ndarray[Any, np.dtype[np.float32]]] = {
            name: np.asarray(result.as_numpy(name), dtype=np.float32) for name in event.output_names
        }

        return InferenceOutput(
            model_name=event.model_name,
            model_version=event.model_version,
            outputs=output_arrays,
            inference_time_ms=elapsed_ms,
            metadata=event.metadata,
        )


class InferenceWorkerFactory:
    """Factory for creating InferenceWorker instances in a MultiProcessProcessor.

    create_shared() returns the picklable InferenceConfig.
    create_processor(shared) creates a worker wrapped with RetryMiddleware.
    """

    def __init__(self, config: InferenceConfig) -> None:  # noqa: D107
        self._config = config

    def create_shared(self) -> InferenceConfig:
        """Return inference config as shared context (picklable pydantic model)."""
        return self._config

    def create_processor(self, shared: InferenceConfig) -> InferenceWorker:
        """Create an InferenceWorker with its own gRPC client."""
        return InferenceWorker(grpc_url=shared.grpc_url)

"""Pure inference processor for Triton gRPC requests.

Business logic only — no threading, no retry, no semaphore.
Concurrency and retry are handled by framework wrappers and middleware.
"""

import time

import numpy as np
import numpy.typing as npt
import tritonclient.grpc as grpcclient
from typed_flow.engine.middleware import RetryMiddleware
from typed_flow.protocols.metrics import TimeMetric

from triton_redis_connector.config.schema import InferenceConfig
from triton_redis_connector.errors.models import InferenceError
from triton_redis_connector.inference.model_metadata import ModelMetadataCache
from triton_redis_connector.inference.request import InferenceOutput, TritonRequest


class InferenceWorker:
    """Processor that sends a single inference request to Triton.

    Creates its own tritonclient.grpc.InferenceServerClient on construction.
    Resolves tensor names from ``ModelMetadataCache`` instead of event fields.
    Raises on transient failures — retry is handled by RetryMiddleware.
    """

    def __init__(self, grpc_url: str, metadata_cache: ModelMetadataCache) -> None:
        """Initialize with a Triton gRPC endpoint URL and metadata cache."""
        self._client: grpcclient.InferenceServerClient = grpcclient.InferenceServerClient(url=grpc_url)
        self._metadata_cache = metadata_cache

    def __call__(self, event: TritonRequest) -> InferenceOutput | InferenceError | None:
        """Run inference on a single TritonRequest event."""
        start = time.monotonic()
        meta = self._metadata_cache.get(event.model_name, event.model_version)

        if event.raw_image is not None:
            dtype_str = "FP16" if event.raw_image.dtype == np.float16 else "FP32"
            input_tensor = grpcclient.InferInput(meta.input_name, event.raw_image.shape, dtype_str)
            input_tensor.set_data_from_numpy(event.raw_image)
        elif event.image_bytes is not None:
            data = np.frombuffer(event.image_bytes, dtype=np.uint8)
            input_tensor = grpcclient.InferInput(meta.input_name, [len(data)], "UINT8")
            input_tensor.set_data_from_numpy(data)
        else:
            msg = "Neither raw_image nor image_bytes set on TritonRequest"
            raise ValueError(msg)

        outputs = [grpcclient.InferRequestedOutput(name) for name in meta.output_names]

        result = self._client.infer(
            model_name=event.model_name,
            inputs=[input_tensor],
            outputs=outputs,
            model_version=event.model_version or "",
        )
        if result is None:
            msg = f"Triton returned None for model {event.model_name}"
            raise RuntimeError(msg)

        elapsed = time.monotonic() - start

        output_arrays: dict[str, npt.NDArray[np.float32]] = {
            name: np.asarray(result.as_numpy(name), dtype=np.float32) for name in meta.output_names
        }

        return InferenceOutput(
            model_name=event.model_name,
            model_version=event.model_version,
            outputs=output_arrays,
            inference_time_ms=TimeMetric.from_seconds(elapsed),
        )


class InferenceWorkerFactory:
    """Factory for creating InferenceWorker instances in a MultiProcessProcessor.

    create_shared() returns the picklable InferenceConfig.
    create_processor(shared) creates a worker wrapped with RetryMiddleware.
    Each worker creates its own gRPC client and metadata cache (fork-safe).
    """

    def __init__(self, config: InferenceConfig) -> None:
        """Initialize factory with inference configuration."""
        self._config = config

    def create_shared(self) -> InferenceConfig:
        """Return inference config as shared context (picklable pydantic model)."""
        return self._config

    def create_processor(
        self, shared: InferenceConfig
    ) -> RetryMiddleware[TritonRequest, InferenceOutput | InferenceError]:
        """Create an InferenceWorker with its own gRPC client, wrapped with RetryMiddleware."""
        client = grpcclient.InferenceServerClient(url=shared.grpc_url)
        cache = ModelMetadataCache(client)
        worker = InferenceWorker(grpc_url=shared.grpc_url, metadata_cache=cache)
        return RetryMiddleware(
            worker,
            max_attempts=shared.max_retry_attempts,
            base_delay_s=shared.retry_base_delay_s,
            backoff_multiplier=shared.retry_backoff_multiplier,
        )

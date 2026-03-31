"""Inference worker as a ProcessPoolProcessor.

Each worker creates its own gRPC client (connections are not fork-safe).
A threading.Semaphore limits concurrent in-flight requests per worker.
Retry logic is delegated to RetryHandler.
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING, Any

import numpy as np
import tritonclient.grpc as grpcclient  # pyright: ignore[reportMissingTypeStubs]
from loguru import logger

from triton_redis_connector.errors.models import InferenceError
from triton_redis_connector.inference.request import InferenceInput, InferenceOutput
from triton_redis_connector.inference.retry import RetryHandler

if TYPE_CHECKING:
    from triton_redis_connector.config.schema import InferenceConfig


class InferenceWorker:
    """ProcessPoolProcessor that sends inference requests to Triton.

    Creates its own tritonclient.grpc.InferenceServerClient on construction.
    Uses a semaphore to limit concurrent in-flight requests.
    """

    def __init__(  # noqa: D107
        self,
        grpc_url: str,
        max_concurrent: int = 8,
        max_retry_attempts: int = 3,
        retry_base_delay_s: float = 0.1,
        retry_backoff_multiplier: float = 4.0,
    ) -> None:
        self._grpc_module: Any = grpcclient
        self._client: Any = grpcclient.InferenceServerClient(url=grpc_url)
        self._semaphore = threading.Semaphore(max_concurrent)
        self._retry_handler = RetryHandler(
            max_attempts=max_retry_attempts,
            base_delay_s=retry_base_delay_s,
            backoff_multiplier=retry_backoff_multiplier,
        )
        self.stop = threading.Event()

    def __call__(self, event: InferenceInput) -> InferenceOutput | InferenceError | None:
        """Run inference on a single InferenceInput event.

        Acquires a semaphore slot, calls Triton with retry logic,
        and returns the result or an error event.
        """
        acquired = self._semaphore.acquire(timeout=0.1)
        if not acquired:
            logger.error("Semaphore full, dropping inference request for model {}", event.model_name)
            return None

        result_holder: list[InferenceOutput | InferenceError | None] = [None]
        done_event = threading.Event()

        def execute_inference() -> None:
            if event.raw_image is None:
                msg = "InferenceWorker requires raw_image to be set"
                raise ValueError(msg)
            start = time.monotonic()
            raw_image = event.raw_image
            grpc = self._grpc_module
            input_tensor = grpc.InferInput(event.input_name, raw_image.shape, "FP32")
            input_tensor.set_data_from_numpy(raw_image)

            outputs = [grpc.InferRequestedOutput(name) for name in event.output_names]

            result = self._client.infer(
                model_name=event.model_name,
                inputs=[input_tensor],
                outputs=outputs,
                model_version=event.model_version or "",
            )

            elapsed_ms = (time.monotonic() - start) * 1000.0

            output_arrays: dict[str, np.ndarray[Any, np.dtype[np.float32]]] = {
                name: np.asarray(result.as_numpy(name), dtype=np.float32) for name in event.output_names
            }

            result_holder[0] = InferenceOutput(
                model_name=event.model_name,
                model_version=event.model_version,
                outputs=output_arrays,
                inference_time_ms=elapsed_ms,
            )

        def on_success(_: object) -> None:
            done_event.set()

        def on_failure(error_message: str) -> None:
            result_holder[0] = InferenceError(
                model_name=event.model_name,
                model_version=event.model_version,
                error_type="InferenceFailure",
                error_message=error_message,
                retry_count=self._retry_handler.max_attempts,
                timestamp_ms=int(time.time() * 1000),
            )
            done_event.set()

        self._retry_handler.execute_with_retry(
            execute_fn=execute_inference,
            on_success=on_success,
            on_failure=on_failure,
            semaphore=self._semaphore,
        )

        done_event.wait()
        return result_holder[0]


class InferenceWorkerFactory:
    """Factory for creating InferenceWorker instances in a ProcessPoolProcessor.

    create_shared() returns the picklable InferenceConfig.
    create_processor(shared) creates a worker with its own gRPC client.
    """

    def __init__(self, config: InferenceConfig) -> None:  # noqa: D107
        self._config = config

    def create_shared(self) -> InferenceConfig:
        """Return inference config as shared context (picklable pydantic model)."""
        return self._config

    def create_processor(self, shared: InferenceConfig) -> InferenceWorker:
        """Create an InferenceWorker with its own gRPC client."""
        return InferenceWorker(
            grpc_url=shared.grpc_url,
            max_concurrent=shared.max_concurrent_per_worker,
            max_retry_attempts=shared.max_retry_attempts,
            retry_base_delay_s=shared.retry_base_delay_s,
            retry_backoff_multiplier=shared.retry_backoff_multiplier,
        )

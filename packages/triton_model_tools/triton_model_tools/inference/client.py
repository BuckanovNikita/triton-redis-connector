"""Synchronous gRPC client for Triton Inference Server."""

from typing import Any

import numpy as np
import numpy.typing as npt
import tritonclient.grpc as grpcclient


class TritonInferenceClient:
    """Synchronous gRPC client for Triton Inference Server."""

    def __init__(self, url: str = "localhost:8001") -> None:
        """Create a Triton gRPC inference client.

        Args:
            url: Triton server gRPC endpoint.

        """
        self._client: Any = grpcclient.InferenceServerClient(url=url)
        self._grpc: Any = grpcclient

    def infer_raw(
        self,
        model_name: str,
        tensor: npt.NDArray[np.floating[Any]],
        output_names: list[str],
    ) -> dict[str, npt.NDArray[np.float32]]:
        """Run inference with a preprocessed tensor (FP32 or FP16).

        Args:
            model_name: Triton model name.
            tensor: Input tensor with batch dimension (e.g. [1, 3, 640, 640]).
            output_names: Names of requested output tensors.

        Returns:
            Dict mapping output name to numpy array.

        """
        grpc = self._grpc
        triton_dtype = "FP16" if tensor.dtype == np.float16 else "FP32"
        input_tensor = grpc.InferInput("images", tensor.shape, triton_dtype)
        input_tensor.set_data_from_numpy(tensor)

        outputs = [grpc.InferRequestedOutput(name) for name in output_names]

        result = self._client.infer(
            model_name=model_name,
            inputs=[input_tensor],
            outputs=outputs,
        )

        return {name: np.asarray(result.as_numpy(name), dtype=np.float32) for name in output_names}

    def infer_dali(
        self,
        model_name: str,
        image_bytes: bytes,
    ) -> dict[str, npt.NDArray[np.float32]]:
        """Run DALI ensemble inference with raw image bytes.

        Args:
            model_name: Triton DALI ensemble model name.
            image_bytes: Raw image file bytes (JPEG/PNG).

        Returns:
            Dict mapping output name to numpy array.

        """
        grpc = self._grpc
        raw_input = np.frombuffer(image_bytes, dtype=np.uint8)
        input_array = raw_input.reshape(1, -1)

        input_tensor = grpc.InferInput("INPUT", input_array.shape, "UINT8")
        input_tensor.set_data_from_numpy(input_array)

        output = grpc.InferRequestedOutput("OUTPUT")
        result = self._client.infer(
            model_name=model_name,
            inputs=[input_tensor],
            outputs=[output],
        )
        output_data: npt.NDArray[np.float32] = np.array(
            result.as_numpy("OUTPUT"),
            dtype=np.float32,
        )
        return {"OUTPUT": output_data}

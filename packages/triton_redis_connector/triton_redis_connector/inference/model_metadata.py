"""Triton model metadata cache for auto-detecting tensor specs at startup.

Queries ``get_model_metadata()`` from Triton gRPC and caches frozen
``ModelMetadata`` dataclasses keyed by ``(model_name, model_version)``.
Thread-safe for concurrent access from pipeline nodes.
"""

import threading
from collections.abc import Iterable
from dataclasses import dataclass

import tritonclient.grpc as grpcclient


@dataclass(frozen=True)
class TensorSpec:
    """Single tensor specification from Triton model metadata."""

    name: str
    datatype: str
    shape: tuple[int, ...]


@dataclass(frozen=True)
class ModelMetadata:
    """Cached model metadata with convenience accessors for common patterns."""

    name: str
    inputs: tuple[TensorSpec, ...]
    outputs: tuple[TensorSpec, ...]

    @property
    def input_name(self) -> str:
        """Primary input tensor name (first input)."""
        return self.inputs[0].name

    @property
    def output_names(self) -> tuple[str, ...]:
        """All output tensor names."""
        return tuple(o.name for o in self.outputs)

    @property
    def input_shape(self) -> tuple[int, ...]:
        """Primary input tensor shape (may contain -1 for dynamic dims)."""
        return self.inputs[0].shape

    @property
    def target_size(self) -> tuple[int, int]:
        """Spatial dimensions (H, W) from primary input shape.

        Extracts the last two positive dimensions from an NCHW shape,
        skipping batch (-1) and channel dims.
        """
        positive = [d for d in self.input_shape if d > 0]
        return (positive[-2], positive[-1])

    @property
    def input_dtype_str(self) -> str:
        """Primary input tensor datatype string (e.g. ``FP32``, ``FP16``, ``UINT8``)."""
        return self.inputs[0].datatype


def _parse_metadata_response(response: object) -> ModelMetadata:
    """Parse a Triton ``ModelMetadataResponse`` protobuf into a frozen dataclass."""
    inputs = tuple(
        TensorSpec(name=t.name, datatype=t.datatype, shape=tuple(t.shape))
        for t in response.inputs  # type: ignore[attr-defined]
    )
    outputs = tuple(
        TensorSpec(name=t.name, datatype=t.datatype, shape=tuple(t.shape))
        for t in response.outputs  # type: ignore[attr-defined]
    )
    return ModelMetadata(
        name=response.name,  # type: ignore[attr-defined]
        inputs=inputs,
        outputs=outputs,
    )


class ModelMetadataCache:
    """Thread-safe cache for Triton model metadata.

    Fetches metadata via ``client.get_model_metadata()`` on first access
    and caches the result. Supports startup prefetch for known models.
    """

    def __init__(self, client: grpcclient.InferenceServerClient) -> None:
        """Initialize with a Triton gRPC client for metadata queries."""
        self._client = client
        self._cache: dict[tuple[str, str], ModelMetadata] = {}
        self._lock = threading.Lock()

    def get(self, model_name: str, model_version: str = "") -> ModelMetadata:
        """Return cached metadata, fetching from Triton on first access."""
        key = (model_name, model_version)
        with self._lock:
            cached = self._cache.get(key)
        if cached is not None:
            return cached

        response = self._client.get_model_metadata(model_name, model_version)
        metadata = _parse_metadata_response(response)

        with self._lock:
            self._cache[key] = metadata
        return metadata

    def prefetch(self, model_names: Iterable[str]) -> None:
        """Eagerly fetch and cache metadata for known models."""
        for name in model_names:
            self.get(name)

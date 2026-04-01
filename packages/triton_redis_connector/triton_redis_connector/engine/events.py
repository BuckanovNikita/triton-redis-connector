"""Base event types for the processing graph.

These are domain event models that flow through the pipeline. They are
distinct from the framework-internal Envelope wrapper.

InferenceInput lives here (not in inference/request.py) so that both
preprocessing and inference layers can import it without violating
the import-linter sibling constraint.
"""

from __future__ import annotations

from typing import Annotated, Self

import numpy as np  # noqa: TC002 — pydantic needs runtime access for model_rebuild()
import numpy.typing as npt  # noqa: TC002
from pydantic import BaseModel, ConfigDict, model_validator

from triton_redis_connector.protocols import Counter, Gauge


class Event(BaseModel):
    """Base marker for all domain events in the processing graph.

    All user-defined events should inherit from this model.
    Frozen for immutability, strict to prevent silent coercion.
    """

    model_config = ConfigDict(frozen=True, strict=True)


class TensorSpec(BaseModel):
    """Specification of a tensor's shape and data type.

    Used to describe input/output tensor requirements for inference.
    """

    model_config = ConfigDict(frozen=True, strict=True)

    name: str
    shape: tuple[int, ...]
    dtype: str


class InferenceOutput(BaseModel):
    """Result of a Triton inference request.

    Carries the model name, version, and output tensor specifications.
    Raw output data is transported separately (as numpy arrays via the
    processing graph, not serialized in this model).
    """

    model_config = ConfigDict(frozen=True, strict=True)

    model_name: str
    model_version: str
    outputs: tuple[TensorSpec, ...]
    request_id: str = ""


class DecodedImage(BaseModel):
    """BGR image decoded from raw bytes with original dimensions.

    Produced by ImageDecodeProcessor, consumed by preprocessing nodes.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    _metrics_ = [Counter("decode_success")]

    image: npt.NDArray[np.uint8]
    original_height: int
    original_width: int
    source_stream: str = ""
    message_id: str = ""


class DetectionResult(BaseModel):
    """Serializable detection output ready for Redis sink."""

    model_config = ConfigDict(frozen=True)

    _metrics_ = [Counter("detection_done")]

    detections_json: str
    model_name: str
    inference_time_ms: Annotated[float, Gauge("inference_time_ms")]
    num_detections: int
    source_stream: str = ""
    message_id: str = ""


class InferenceInput(BaseModel):
    """Input event for Triton inference processor.

    Carries the preprocessed tensor and metadata needed to dispatch
    a gRPC inference request to Triton. Lives in engine layer so both
    preprocessing and inference can import it.

    Supports two modes: either ``raw_image`` (preprocessed tensor) or
    ``image_bytes`` (raw encoded bytes for server-side decoding) must
    be set, but not both.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    model_name: str
    model_version: str = ""
    raw_image: npt.NDArray[np.float32] | None = None
    image_bytes: bytes | None = None
    input_name: str = "images"
    output_names: tuple[str, ...] = ("output0",)

    @model_validator(mode="after")
    def _exactly_one_input(self) -> Self:
        has_raw = self.raw_image is not None
        has_bytes = self.image_bytes is not None
        if has_raw == has_bytes:
            msg = "Exactly one of raw_image or image_bytes must be set"
            raise ValueError(msg)
        return self


DecodedImage.model_rebuild()
InferenceInput.model_rebuild()

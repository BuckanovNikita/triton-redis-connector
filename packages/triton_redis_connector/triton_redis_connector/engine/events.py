"""Base event types for the processing graph.

These are domain event models that flow through the pipeline. They are
distinct from the framework-internal Envelope wrapper.

InferenceInput lives here (not in inference/request.py) so that both
preprocessing and inference layers can import it without violating
the import-linter sibling constraint.
"""

from __future__ import annotations

import numpy as np  # noqa: TC002 — pydantic needs runtime access for model_rebuild()
import numpy.typing as npt  # noqa: TC002
from pydantic import BaseModel, ConfigDict


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

    image: npt.NDArray[np.uint8]
    original_height: int
    original_width: int
    source_stream: str = ""
    message_id: str = ""


class DetectionResult(BaseModel):
    """Serializable detection output ready for Redis sink."""

    model_config = ConfigDict(frozen=True)

    detections_json: str
    model_name: str
    inference_time_ms: float
    num_detections: int
    source_stream: str = ""
    message_id: str = ""


class InferenceInput(BaseModel):
    """Input event for Triton inference processor.

    Carries the preprocessed tensor and metadata needed to dispatch
    a gRPC inference request to Triton. Lives in engine layer so both
    preprocessing and inference can import it.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    model_name: str
    model_version: str = ""
    raw_image: npt.NDArray[np.float32]
    input_name: str = "images"
    output_names: tuple[str, ...] = ("output0",)


DecodedImage.model_rebuild()
InferenceInput.model_rebuild()

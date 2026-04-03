"""Base event types for the processing graph.

These are domain event models that flow through the pipeline. They are
distinct from the framework-internal Envelope wrapper.

TritonRequest lives here (not in inference/request.py) so that both
preprocessing and inference layers can import it without violating
the import-linter sibling constraint.
"""

from __future__ import annotations

from typing import Annotated, ClassVar, Self

import numpy as np  # noqa: TC002 — pydantic needs runtime access for model_rebuild()
import numpy.typing as npt  # noqa: TC002
from pydantic import BaseModel, ConfigDict, model_validator
from triton_shared_models.classification import ClassificationPrediction  # noqa: TC002
from triton_shared_models.detection import Detection  # noqa: TC002

from triton_redis_connector.protocols import Counter, Gauge


class InferenceTarget(BaseModel):
    """Mixin for user events that carry per-event model routing info.

    When a user event inherits from both ``Event`` and ``InferenceTarget``,
    the ``ClientPreprocessorNode`` will use ``target_model`` instead of the
    pipeline-default model name (if non-empty). This enables dynamic model
    routing on a per-event basis.
    """

    model_config = ConfigDict(frozen=True)

    target_model: str = ""
    target_model_version: str = ""


class FrameworkMeta(BaseModel):
    """Framework-owned metadata carried through the pipeline on Event.fw.

    Set by LiveRedisStreamSource from the Redis envelope. Stripped by
    TypedRedisStreamSink before writing to the output stream.
    """

    model_config = ConfigDict(frozen=True)

    source_stream: str = ""
    message_id: str = ""
    priority: int = 0


class Event(BaseModel):
    """Base marker for all domain events in the processing graph.

    All user-defined events should inherit from this model.
    Frozen for immutability, strict to prevent silent coercion.
    The ``fw`` field carries framework metadata (source_stream, message_id)
    and is NOT serialized to Redis output.
    """

    model_config = ConfigDict(frozen=True, strict=True)

    fw: FrameworkMeta = FrameworkMeta()


class PipelineEvent[TCtx: Event, TPayload: BaseModel](BaseModel):
    """Typed composition of user context and processing payload.

    Carries the original user event (context) alongside the current
    processing stage payload. Framework routes by exact generic match
    on ``PipelineEvent[ConcreteCtx, ConcretePayload]``.
    """

    model_config = ConfigDict(frozen=True)

    context: TCtx
    payload: TPayload


class ImagePayload(BaseModel):
    """Payload carrying image data for preprocessing nodes.

    Either ``image_path`` (filesystem) or ``image_bytes`` (in-memory)
    should be set for the preprocessor to consume.
    """

    model_config = ConfigDict(frozen=True)

    image_path: str = ""
    image_bytes: bytes | None = None


class DetectionResult(BaseModel):
    """Typed detection output with structured detections ready for Redis sink."""

    model_config = ConfigDict(frozen=True)

    _metrics_: ClassVar[list[Counter]] = [Counter("detection_done")]

    detections: tuple[Detection, ...]
    model_name: str
    inference_time_ms: Annotated[float, Gauge("inference_time_ms")]


class ClassificationResult(BaseModel):
    """Typed classification output with structured predictions ready for Redis sink."""

    model_config = ConfigDict(frozen=True)

    _metrics_: ClassVar[list[Counter]] = [Counter("classification_done")]

    predictions: tuple[ClassificationPrediction, ...]
    model_name: str
    inference_time_ms: Annotated[float, Gauge("classification_inference_time_ms")]


class TritonRequest(BaseModel):
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


TritonRequest.model_rebuild()

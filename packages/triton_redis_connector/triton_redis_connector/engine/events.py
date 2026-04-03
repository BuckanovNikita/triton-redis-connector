"""Domain event types for Triton inference pipelines.

Triton-specific event models that extend the generic typed_flow
framework with image processing and inference capabilities.
"""

from __future__ import annotations

from typing import Annotated, ClassVar, Self

import numpy as np  # noqa: TC002 — pydantic needs runtime access for model_rebuild()
import numpy.typing as npt  # noqa: TC002
from pydantic import BaseModel, ConfigDict, model_validator
from triton_shared_models.classification import ClassificationPrediction  # noqa: TC002
from triton_shared_models.detection import Detection  # noqa: TC002
from typed_flow.protocols import Counter, Gauge
from typed_flow.protocols.messaging import BaseEvent, FrameworkMeta
from typed_flow.protocols.metrics import TimeMetric  # noqa: TC002


class InferenceTarget(BaseModel):
    """Mixin for user events that carry per-event model routing info.

    When a user event inherits from ``InferenceTarget``,
    the ``ClientPreprocessorNode`` will use ``target_model`` instead of the
    pipeline-default model name (if non-empty). This enables dynamic model
    routing on a per-event basis.
    """

    model_config = ConfigDict(frozen=True)

    flow_meta: FrameworkMeta = FrameworkMeta()
    target_model: str = ""
    target_model_version: str = ""


class ImagePayload(BaseModel):
    """Mixin carrying image data for preprocessing nodes.

    Either ``image_path`` (filesystem) or ``image_bytes`` (in-memory)
    should be set for the preprocessor to consume. Carries flow_meta
    for error reporting in the preprocessor.
    """

    model_config = ConfigDict(frozen=True)

    flow_meta: FrameworkMeta = FrameworkMeta()
    image_path: str = ""
    image_bytes: bytes | None = None


class DetectionResult(BaseEvent):
    """Typed detection output with structured detections ready for Redis sink."""

    _metrics_: ClassVar[list[Counter]] = [Counter("detection_done")]

    detections: tuple[Detection, ...]
    model_name: str
    inference_time_ms: Annotated[TimeMetric, Gauge("inference_time_ms")]


class ClassificationResult(BaseEvent):
    """Typed classification output with structured predictions ready for Redis sink."""

    _metrics_: ClassVar[list[Counter]] = [Counter("classification_done")]

    predictions: tuple[ClassificationPrediction, ...]
    model_name: str
    inference_time_ms: Annotated[TimeMetric, Gauge("classification_inference_time_ms")]


class TritonRequest(BaseEvent):
    """Input event for Triton inference processor.

    Carries the preprocessed tensor and metadata needed to dispatch
    a gRPC inference request to Triton.

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

"""Domain event models for the dual-model steel inspection pipeline.

Events flow through the pipeline graph:
  SteelInspectionInput -> EvenCameraFrame | OddCameraFrame
    -> MetadataInferenceInput -> MetadataInferenceOutput -> InspectionResult
"""

from __future__ import annotations

from typing import Annotated, ClassVar

from pydantic import BaseModel, ConfigDict
from triton_redis_connector.engine.events import Event, InferenceInput
from triton_redis_connector.inference.request import InferenceOutput  # noqa: TC002
from triton_redis_connector.protocols import Counter, Gauge, GroupBy


class SteelInspectionInput(Event):
    """Raw inspection event parsed from a Redis stream envelope."""

    _metrics_: ClassVar[list[Counter]] = [Counter("inspection_received")]

    receipt_ts: float
    camera_no: int
    image_path: str
    frame_counter: int
    metal_type: str | None = None


class EvenCameraFrame(Event):
    """Even camera_no frames routed to DALI + yolo11s_960x608 ensemble."""

    model_config = ConfigDict(frozen=True, strict=True, arbitrary_types_allowed=True)

    _metrics_: ClassVar[list[Counter]] = [Counter("even_camera_frame")]

    image_bytes: bytes
    camera_no: Annotated[int, GroupBy("camera_no_per_minute", window_s=60)]
    frame_counter: int
    metal_type: str
    receipt_ts: float
    image_path: str


class OddCameraFrame(Event):
    """Odd camera_no frames routed to TurboJPEG + yolo26l."""

    model_config = ConfigDict(frozen=True, strict=True, arbitrary_types_allowed=True)

    _metrics_: ClassVar[list[Counter]] = [Counter("odd_camera_frame")]

    image_bytes: bytes
    camera_no: Annotated[int, GroupBy("camera_no_per_minute", window_s=60)]
    frame_counter: int
    metal_type: str
    receipt_ts: float
    image_path: str


class MetadataInferenceInput(BaseModel):
    """Carries InferenceInput through the inference boundary with pipeline metadata."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    inference_input: InferenceInput
    camera_no: int
    frame_counter: int
    metal_type: Annotated[str, GroupBy("metal_type_per_minute", window_s=60)]
    receipt_ts: float
    image_path: str


class MetadataInferenceOutput(BaseModel):
    """Carries InferenceOutput after inference with pipeline metadata."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    inference_output: InferenceOutput
    camera_no: int
    frame_counter: int
    metal_type: str
    receipt_ts: float
    image_path: str


class InspectionResult(Event):
    """Final detection result ready for Redis sink."""

    _metrics_: ClassVar[list[Counter]] = [Counter("inspection_complete")]

    camera_no: int
    metal_type: str
    detections_json: str
    model_name: str
    inference_time_ms: Annotated[float, Gauge("inspection_inference_ms")]
    num_detections: int
    frame_counter: int
    receipt_ts: float
    image_path: str


MetadataInferenceInput.model_rebuild()
MetadataInferenceOutput.model_rebuild()

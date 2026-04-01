"""Domain event models for the dual-model steel inspection pipeline.

Events flow through the pipeline graph:
  SteelInspectionInput -> EvenCameraFrame | OddCameraFrame
    -> (shipped preprocessors) -> InferenceWorker -> YoloPostProcessorBridge
    -> DetectionResult (framework event with metadata)
"""

from __future__ import annotations

from typing import Annotated, Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field
from triton_redis_connector.engine.events import Event
from triton_redis_connector.protocols import Counter, GroupBy


class SteelInspectionInput(Event):
    """Raw inspection event parsed from a Redis stream envelope."""

    _metrics_: ClassVar[list[Counter]] = [Counter("inspection_received")]

    receipt_ts: float
    camera_no: int
    image_path: str
    frame_counter: int
    metal_type: str | None = None


class EvenCameraFrame(BaseModel):
    """Even camera_no frames routed to DALI + yolo11s_960x608 ensemble."""

    model_config = ConfigDict(frozen=True, strict=True)

    _metrics_: ClassVar[list[Counter]] = [Counter("even_camera_frame")]

    image_path: str
    camera_no: Annotated[int, GroupBy("camera_no_per_minute", window_s=60)]
    metadata: dict[str, Any] = Field(default_factory=dict)


class OddCameraFrame(BaseModel):
    """Odd camera_no frames routed to TurboJPEG + yolo26l."""

    model_config = ConfigDict(frozen=True, strict=True)

    _metrics_: ClassVar[list[Counter]] = [Counter("odd_camera_frame")]

    image_path: str
    camera_no: Annotated[int, GroupBy("camera_no_per_minute", window_s=60)]
    metadata: dict[str, Any] = Field(default_factory=dict)

"""Domain event models for the dual-model steel inspection pipeline.

Events flow through the pipeline graph using PipelineEvent composition:
  SteelInspectionInput
    -> CameraBranchRouter
    -> PipelineEvent[SteelMeta, EvenImagePayload | OddImagePayload]
    -> ClientPreprocessorNode (x2, one per model)
    -> PipelineEvent[SteelMeta, TritonRequest]  (merge point)
    -> InferenceNode
    -> PipelineEvent[SteelMeta, InferenceOutput]
    -> PostProcessorNode
    -> PipelineEvent[SteelMeta, DetectionResult]
    -> ModelResultRouter
    -> PipelineEvent[SteelMeta, EvenDetectionResult | OddDetectionResult]
    -> EvenResultSink / OddResultSink (per-model output streams)

Per-model ImagePayload subclasses split the preprocessing path.
Per-model DetectionResult subclasses split the sink path.
The middle stages (inference + postprocessing) are shared merge points.
"""

from __future__ import annotations

from typing import Annotated, ClassVar

from triton_redis_connector.engine.events import (
    DetectionResult,
    Event,
    ImagePayload,
)
from triton_redis_connector.protocols import Counter, GroupBy

EVEN_MODEL = "yolo11s_960x608"
ODD_MODEL = "yolo26l"


class SteelMeta(Event):
    """Typed context carrying domain data through the pipeline.

    Replaces the old untyped ``metadata: dict[str, Any]`` pattern.
    Travels inside ``PipelineEvent.context`` through all processing stages.
    """

    _metrics_: ClassVar[list[Counter]] = [Counter("inspection_received")]

    receipt_ts: float
    camera_no: Annotated[int, GroupBy("camera_no_per_minute", window_s=60)]
    image_path: str
    frame_counter: int
    metal_type: str


class SteelInspectionInput(Event):
    """Raw inspection event parsed from a Redis stream envelope."""

    receipt_ts: float
    camera_no: int
    image_path: str
    frame_counter: int
    metal_type: str | None = None


class EvenImagePayload(ImagePayload):
    """Image payload for even camera_no frames (yolo11s_960x608)."""

    _metrics_: ClassVar[list[Counter]] = [Counter("even_camera_frame")]


class OddImagePayload(ImagePayload):
    """Image payload for odd camera_no frames (yolo26l)."""

    _metrics_: ClassVar[list[Counter]] = [Counter("odd_camera_frame")]


class EvenDetectionResult(DetectionResult):
    """Detection result from the even-camera model (yolo11s_960x608)."""

    _metrics_: ClassVar[list[Counter]] = [Counter("even_model_result")]


class OddDetectionResult(DetectionResult):
    """Detection result from the odd-camera model (yolo26l)."""

    _metrics_: ClassVar[list[Counter]] = [Counter("odd_model_result")]

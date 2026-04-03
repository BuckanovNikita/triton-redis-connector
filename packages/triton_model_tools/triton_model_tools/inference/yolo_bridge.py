"""Bridge YoloPostprocessor into the pipeline as a callable processor."""

from __future__ import annotations

from triton_redis_connector.engine.events import DetectionResult
from triton_redis_connector.inference.request import (
    InferenceOutput,  # noqa: TC002 — required at runtime for annotation resolution
)

from triton_model_tools.inference.postprocessing import (
    YoloDetailedPostprocessor,
    YoloPostprocessor,
)


class YoloPostProcessorBridge:
    """Bridges YoloPostprocessor output into a pipeline DetectionResult event.

    Takes raw InferenceOutput from Triton, runs YOLO postprocessing
    (NMS + confidence filtering), and produces a typed DetectionResult
    with structured Detection objects.

    Config-friendly: accepts simple params and constructs YoloPostprocessor
    internally. For programmatic use, pass a pre-built ``postprocessor``.
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        postprocessor: YoloPostprocessor | None = None,
        end2end: bool = False,
        scale: float = 1.0,
        pad_offset: tuple[int, int] = (0, 0),
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ) -> None:
        """Initialize bridge with postprocessor and detection thresholds."""
        self._postprocessor = postprocessor or YoloPostprocessor(end2end=end2end)
        self._scale = scale
        self._pad_offset = pad_offset
        self._conf_threshold = conf_threshold
        self._iou_threshold = iou_threshold

    def __call__(self, output: InferenceOutput) -> DetectionResult:
        """Run YOLO postprocessing and return a typed DetectionResult."""
        detections = self._postprocessor.postprocess(
            outputs=output.outputs,
            scale=self._scale,
            pad_offset=self._pad_offset,
            conf_threshold=self._conf_threshold,
            iou_threshold=self._iou_threshold,
        )
        return DetectionResult(
            detections=tuple(detections),
            model_name=output.model_name,
            inference_time_ms=output.inference_time_ms,
        )


class YoloDetailedPostProcessorBridge:
    """Bridges YoloDetailedPostprocessor into the pipeline with all-class scores.

    Returns DetectionResult where each Detection has ``all_class_scores``
    populated with the full class score vector from the raw model output.

    Only works with standard (non-end2end) models.
    """

    def __init__(
        self,
        *,
        postprocessor: YoloDetailedPostprocessor | None = None,
        scale: float = 1.0,
        pad_offset: tuple[int, int] = (0, 0),
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ) -> None:
        """Initialize bridge with detailed postprocessor and detection thresholds."""
        self._postprocessor = postprocessor or YoloDetailedPostprocessor()
        self._scale = scale
        self._pad_offset = pad_offset
        self._conf_threshold = conf_threshold
        self._iou_threshold = iou_threshold

    def __call__(self, output: InferenceOutput) -> DetectionResult:
        """Run detailed YOLO postprocessing with all-class scores."""
        detections = self._postprocessor.postprocess(
            outputs=output.outputs,
            scale=self._scale,
            pad_offset=self._pad_offset,
            conf_threshold=self._conf_threshold,
            iou_threshold=self._iou_threshold,
        )
        return DetectionResult(
            detections=tuple(detections),
            model_name=output.model_name,
            inference_time_ms=output.inference_time_ms,
        )

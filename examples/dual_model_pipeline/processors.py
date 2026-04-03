"""Custom pipeline processors for the dual-model steel inspection pipeline.

Only domain-specific routing and sink logic lives here. All preprocessing,
inference, and postprocessing use shipped Node classes.
"""

from __future__ import annotations

from loguru import logger
from redis import Redis  # noqa: TC002 — needed at runtime for sink __init__
from triton_redis_connector.engine.events import (
    DetectionResult,
    PipelineEvent,
)
from triton_redis_connector.protocols import BaseThreadProcessor, BaseThreadSink

from examples.dual_model_pipeline.events import (
    EVEN_MODEL,
    EvenDetectionResult,
    EvenImagePayload,
    OddDetectionResult,
    OddImagePayload,
    SteelInspectionInput,
    SteelMeta,
)


class CameraBranchRouter(
    BaseThreadProcessor[
        SteelInspectionInput,
        PipelineEvent[SteelMeta, EvenImagePayload] | PipelineEvent[SteelMeta, OddImagePayload],
    ],
):
    """Routes inspection input to even/odd camera preprocessing paths.

    Even camera_no -> PipelineEvent[SteelMeta, EvenImagePayload]
    Odd camera_no  -> PipelineEvent[SteelMeta, OddImagePayload]

    Domain metadata is packed into the typed SteelMeta context,
    replacing the old untyped metadata dict bypass pattern.
    """

    def __init__(self) -> None:
        """Initialize the camera branch router."""
        super().__init__()

    def __call__(
        self, event: SteelInspectionInput
    ) -> PipelineEvent[SteelMeta, EvenImagePayload] | PipelineEvent[SteelMeta, OddImagePayload] | None:
        """Route event to even or odd camera branch based on camera number."""
        metal_type = event.metal_type or ("HK" if event.camera_no % 2 == 0 else "GK")
        context = SteelMeta(
            fw=event.fw,
            receipt_ts=event.receipt_ts,
            camera_no=event.camera_no,
            image_path=event.image_path,
            frame_counter=event.frame_counter,
            metal_type=metal_type,
        )

        if event.camera_no % 2 == 0:
            return PipelineEvent[SteelMeta, EvenImagePayload](
                context=context,
                payload=EvenImagePayload(image_path=event.image_path),
            )

        return PipelineEvent(
            context=context,
            payload=OddImagePayload(image_path=event.image_path),
        )


class ModelResultRouter(
    BaseThreadProcessor[
        PipelineEvent[SteelMeta, DetectionResult],
        PipelineEvent[SteelMeta, EvenDetectionResult] | PipelineEvent[SteelMeta, OddDetectionResult],
    ],
):
    """Routes detection results to per-model typed sinks based on model_name.

    Converts generic DetectionResult payload into model-specific subclasses
    so the type-based routing table sends results to the correct sink.
    """

    def __init__(self) -> None:
        """Initialize the model result router."""
        super().__init__()

    def __call__(
        self, event: PipelineEvent[SteelMeta, DetectionResult]
    ) -> PipelineEvent[SteelMeta, EvenDetectionResult] | PipelineEvent[SteelMeta, OddDetectionResult] | None:
        """Route detection result to even or odd typed sink by model name."""
        result = event.payload
        if result.model_name == EVEN_MODEL:
            typed_result = EvenDetectionResult(
                detections=result.detections,
                model_name=result.model_name,
                inference_time_ms=result.inference_time_ms,
            )
            return PipelineEvent[SteelMeta, EvenDetectionResult](context=event.context, payload=typed_result)

        typed_result_odd = OddDetectionResult(
            detections=result.detections,
            model_name=result.model_name,
            inference_time_ms=result.inference_time_ms,
        )
        return PipelineEvent[SteelMeta, OddDetectionResult](context=event.context, payload=typed_result_odd)


class EvenResultSink(
    BaseThreadSink[PipelineEvent[SteelMeta, EvenDetectionResult]],
):
    """Writes even-camera detection results to a dedicated Redis stream.

    Serialization is built into the sink: context fields become top-level
    Redis hash entries, payload is serialized as a ``result`` JSON field.
    """

    def __init__(self, client: Redis[bytes], stream: str, maxlen: int = 10_000) -> None:
        """Initialize sink with Redis client, stream name, and max length."""
        super().__init__()
        self._client = client
        self._stream = stream
        self._maxlen = maxlen

    def __call__(self, event: PipelineEvent[SteelMeta, EvenDetectionResult]) -> None:
        """Serialize and write even-camera result to Redis stream."""
        fields = _serialize_pipeline_event(event)
        self._client.xadd(self._stream, fields, maxlen=self._maxlen, approximate=True)
        logger.debug("Wrote even result to {}", self._stream)


class OddResultSink(
    BaseThreadSink[PipelineEvent[SteelMeta, OddDetectionResult]],
):
    """Writes odd-camera detection results to a dedicated Redis stream.

    Same serialization as EvenResultSink, targeting a different stream.
    """

    def __init__(self, client: Redis[bytes], stream: str, maxlen: int = 10_000) -> None:
        """Initialize sink with Redis client, stream name, and max length."""
        super().__init__()
        self._client = client
        self._stream = stream
        self._maxlen = maxlen

    def __call__(self, event: PipelineEvent[SteelMeta, OddDetectionResult]) -> None:
        """Serialize and write odd-camera result to Redis stream."""
        fields = _serialize_pipeline_event(event)
        self._client.xadd(self._stream, fields, maxlen=self._maxlen, approximate=True)
        logger.debug("Wrote odd result to {}", self._stream)


def _serialize_pipeline_event(
    event: PipelineEvent[SteelMeta, EvenDetectionResult] | PipelineEvent[SteelMeta, OddDetectionResult],
) -> dict[str, str]:
    """Serialize a PipelineEvent into Redis hash fields.

    Context fields (excluding ``fw``) become top-level entries.
    Payload (DetectionResult) becomes a single ``result`` JSON field.
    """
    ctx = event.context
    result = event.payload

    return {
        "camera_no": str(ctx.camera_no),
        "frame_counter": str(ctx.frame_counter),
        "receipt_ts": str(ctx.receipt_ts),
        "metal_type": ctx.metal_type,
        "image_path": ctx.image_path,
        "result": result.model_dump_json(),
        "model_name": result.model_name,
        "num_detections": str(len(result.detections)),
    }

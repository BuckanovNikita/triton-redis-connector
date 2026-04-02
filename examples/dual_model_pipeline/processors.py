"""Custom pipeline processors for the dual-model steel inspection pipeline.

Only domain-specific routing logic lives here. All preprocessing, inference,
and postprocessing use shipped processors configured in config.yaml.
"""

from __future__ import annotations

import json

from redis import Redis  # noqa: TC002 — needed at runtime for sink __init__
from triton_redis_connector.engine.events import DetectionResult
from triton_redis_connector.protocols import BaseThreadProcessor, BaseThreadSink

from examples.dual_model_pipeline.events import (
    EVEN_MODEL,
    EvenCameraFrame,
    EvenModelResult,
    OddCameraFrame,
    OddModelResult,
    SteelInspectionInput,
)


class CameraBranchRouter(
    BaseThreadProcessor[SteelInspectionInput, EvenCameraFrame | OddCameraFrame],
):
    """Routes inspection input to even/odd camera preprocessing paths.

    Even camera_no -> EvenCameraFrame (DALI ensemble path)
    Odd camera_no  -> OddCameraFrame  (client-side TurboJPEG path)

    Domain metadata is packed into the metadata dict so shipped processors
    can bypass it through inference and back into the final output.
    """

    def __init__(self) -> None:  # noqa: D107
        super().__init__()

    def __call__(self, event: SteelInspectionInput) -> EvenCameraFrame | OddCameraFrame | None:
        """Route based on camera_no parity, packing metadata for bypass."""
        meta = {
            "camera_no": event.camera_no,
            "frame_counter": event.frame_counter,
            "receipt_ts": event.receipt_ts,
            "metal_type": event.metal_type or ("HK" if event.camera_no % 2 == 0 else "GK"),
            "image_path": event.image_path,
        }

        if event.camera_no % 2 == 0:
            return EvenCameraFrame(
                image_path=event.image_path,
                camera_no=event.camera_no,
                metadata=meta,
            )
        return OddCameraFrame(
            image_path=event.image_path,
            camera_no=event.camera_no,
            metadata=meta,
        )


class ModelResultRouter(
    BaseThreadProcessor[DetectionResult, EvenModelResult | OddModelResult],
):
    """Routes detection results to per-model output streams based on model_name."""

    def __init__(self) -> None:  # noqa: D107
        super().__init__()

    def __call__(self, event: DetectionResult) -> EvenModelResult | OddModelResult | None:
        """Route result to the output type matching the model that produced it."""
        cls = EvenModelResult if event.model_name == EVEN_MODEL else OddModelResult
        return cls(
            detections=event.detections,
            model_name=event.model_name,
            inference_time_ms=event.inference_time_ms,
        )


def _result_to_redis_fields(event: EvenModelResult | OddModelResult) -> dict[str, str]:
    """Convert a model result to Redis hash fields for XADD."""
    detections_json = json.dumps([d.model_dump() for d in event.detections])
    fields: dict[str, str] = {
        "detections": detections_json,
        "model_name": event.model_name,
        "inference_time_ms": str(event.inference_time_ms),
        "num_detections": str(len(event.detections)),
    }
    for key, value in event.metadata.items():
        fields[str(key)] = str(value)
    return fields


class EvenModelSink(BaseThreadSink[EvenModelResult]):
    """Writes even-camera model results to a dedicated Redis stream."""

    def __init__(self, client: Redis[bytes], stream: str, maxlen: int = 10_000) -> None:  # noqa: D107
        super().__init__()
        self._client = client
        self._stream = stream
        self._maxlen = maxlen

    def __call__(self, event: EvenModelResult) -> None:
        """Write result fields to the even-model output stream."""
        self._client.xadd(self._stream, _result_to_redis_fields(event), maxlen=self._maxlen, approximate=True)


class OddModelSink(BaseThreadSink[OddModelResult]):
    """Writes odd-camera model results to a dedicated Redis stream."""

    def __init__(self, client: Redis[bytes], stream: str, maxlen: int = 10_000) -> None:  # noqa: D107
        super().__init__()
        self._client = client
        self._stream = stream
        self._maxlen = maxlen

    def __call__(self, event: OddModelResult) -> None:
        """Write result fields to the odd-model output stream."""
        self._client.xadd(self._stream, _result_to_redis_fields(event), maxlen=self._maxlen, approximate=True)

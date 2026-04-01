"""Custom pipeline processor for the dual-model steel inspection pipeline.

Only domain-specific routing logic lives here. All preprocessing, inference,
and postprocessing use shipped processors configured in config.yaml.
"""

from __future__ import annotations

from triton_redis_connector.protocols import BaseThreadProcessor

from examples.dual_model_pipeline.events import (
    EvenCameraFrame,
    OddCameraFrame,
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

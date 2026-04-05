"""Domain-specific stream entry models and re-exports from typed_flow."""

from pydantic import BaseModel, ConfigDict
from typed_flow.connectors.stream_writer import (
    PopulateResult,
    StreamFields,
    populate_stream,
    serialize_model,
)

__all__ = [
    "ImageStreamEntry",
    "InspectionStreamEntry",
    "PopulateResult",
    "StreamFields",
    "populate_stream",
    "serialize_model",
]


class ImageStreamEntry(BaseModel):
    """Single image entry for Redis stream population (binary image data)."""

    model_config = ConfigDict(frozen=True)

    data: bytes
    request_id: str
    model_name: str
    image_name: str


class InspectionStreamEntry(BaseModel):
    """Steel inspection event entry for Redis stream population."""

    model_config = ConfigDict(frozen=True)

    camera_no: int
    image_path: str
    frame_counter: int
    receipt_ts: float
    metal_type: str
    event_type: str = "image"
    model_name: str = ""

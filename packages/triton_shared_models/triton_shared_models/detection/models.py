"""Detection model for object detection results."""

from pydantic import BaseModel, ConfigDict, Field


class Detection(BaseModel):
    """Single object detection with bounding box, confidence, and class info."""

    model_config = ConfigDict(frozen=True)

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float = Field(ge=0.0)
    class_id: int = Field(ge=0)
    class_name: str = ""
    all_class_scores: tuple[float, ...] | None = None

"""Classification prediction model."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ClassificationPrediction(BaseModel):
    """Single classification prediction with class and probability."""

    model_config = ConfigDict(frozen=True)

    class_id: int = Field(ge=0)
    class_name: str = ""
    probability: float = Field(ge=0.0, le=1.0)

"""Classification prediction model."""

from pydantic import BaseModel, ConfigDict, Field


class ClassificationPrediction(BaseModel):
    """Single classification prediction with class and probability."""

    model_config = ConfigDict(frozen=True)

    class_id: int = Field(ge=0)
    class_name: str = ""
    probability: float = Field(ge=0.0, le=1.0)


class DetailedClassificationPrediction(ClassificationPrediction):
    """Classification prediction carrying the full probability vector.

    Extends the base prediction with all class probabilities, allowing
    downstream consumers to inspect scores beyond the top-K.
    """

    all_probabilities: tuple[float, ...] = ()

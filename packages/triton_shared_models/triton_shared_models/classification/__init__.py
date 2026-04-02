"""Classification types and preprocessing."""

from __future__ import annotations

from triton_shared_models.classification.models import ClassificationPrediction
from triton_shared_models.classification.preprocessing import CenterCropPreprocessor

__all__ = ["CenterCropPreprocessor", "ClassificationPrediction"]

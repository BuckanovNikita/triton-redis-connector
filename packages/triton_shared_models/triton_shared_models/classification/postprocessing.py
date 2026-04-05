"""Postprocessing for classification model outputs."""

import numpy as np
import numpy.typing as npt

from triton_shared_models.classification import (
    ClassificationPrediction,
    DetailedClassificationPrediction,
)


class ClassificationPostprocessor:
    """Postprocesses classification model outputs into ClassificationPrediction objects.

    Takes raw softmax/logits output and produces top-K predictions.
    """

    def __init__(self, *, top_k: int = 5, apply_softmax: bool = True) -> None:
        """Initialize with top-K count and softmax application flag.

        Args:
            top_k: Number of top predictions to return.
            apply_softmax: Whether to apply softmax to raw logits before ranking.

        """
        self._top_k = top_k
        self._apply_softmax = apply_softmax

    def _compute_probabilities(
        self,
        outputs: dict[str, npt.NDArray[np.float32]],
    ) -> npt.NDArray[np.float32]:
        output_key = next(iter(outputs))
        logits = outputs[output_key]

        if logits.ndim > 1:
            logits = logits.squeeze(0)

        if self._apply_softmax:
            exp_logits = np.exp(logits - np.max(logits))
            probabilities: npt.NDArray[np.float32] = exp_logits / np.sum(exp_logits)
        else:
            probabilities = logits

        return probabilities

    def postprocess(
        self,
        outputs: dict[str, npt.NDArray[np.float32]],
        class_names: dict[int, str] | None = None,
    ) -> list[ClassificationPrediction]:
        """Convert raw model output to ranked classification predictions.

        Args:
            outputs: Dict mapping output tensor name to numpy array.
                     Uses the first key's array as logits/probabilities.
            class_names: Optional mapping from class ID to human-readable name.

        Returns:
            Top-K predictions sorted by descending probability.

        """
        probabilities = self._compute_probabilities(outputs)
        top_indices = np.argsort(probabilities)[::-1][: self._top_k]

        predictions: list[ClassificationPrediction] = []
        for idx in top_indices:
            class_id = int(idx)
            class_name = class_names.get(class_id, "") if class_names else ""
            predictions.append(
                ClassificationPrediction(
                    class_id=class_id,
                    class_name=class_name,
                    probability=float(probabilities[idx]),
                ),
            )
        return predictions


class DetailedClassificationPostprocessor(ClassificationPostprocessor):
    """Extended classification postprocessor carrying the full probability vector.

    Returns DetailedClassificationPrediction objects where each prediction
    includes ``all_probabilities`` — the complete probability distribution
    over all classes, allowing downstream consumers to inspect beyond top-K.
    """

    def postprocess(  # type: ignore[override]
        self,
        outputs: dict[str, npt.NDArray[np.float32]],
        class_names: dict[int, str] | None = None,
    ) -> list[DetailedClassificationPrediction]:
        """Convert raw model output to detailed predictions with full probability vector."""
        probabilities = self._compute_probabilities(outputs)
        all_probs = tuple(float(p) for p in probabilities)
        top_indices = np.argsort(probabilities)[::-1][: self._top_k]

        predictions: list[DetailedClassificationPrediction] = []
        for idx in top_indices:
            class_id = int(idx)
            class_name = class_names.get(class_id, "") if class_names else ""
            predictions.append(
                DetailedClassificationPrediction(
                    class_id=class_id,
                    class_name=class_name,
                    probability=float(probabilities[idx]),
                    all_probabilities=all_probs,
                ),
            )
        return predictions

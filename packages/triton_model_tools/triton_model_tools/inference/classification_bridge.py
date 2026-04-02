"""Bridge ClassificationPostprocessor into the pipeline as a callable processor."""

from __future__ import annotations

from triton_redis_connector.engine.events import ClassificationResult
from triton_redis_connector.inference.request import (
    InferenceOutput,  # noqa: TC002 — required at runtime for annotation resolution
)

from triton_model_tools.inference.classification_postprocessing import (
    ClassificationPostprocessor,
    DetailedClassificationPostprocessor,
)


class ClassificationPostProcessorBridge:
    """Bridges ClassificationPostprocessor output into a pipeline ClassificationResult.

    Takes raw InferenceOutput from Triton, runs classification postprocessing
    (softmax + top-K), and produces a ClassificationResult with typed predictions.

    Config-friendly: accepts simple params and constructs ClassificationPostprocessor
    internally. For programmatic use, pass a pre-built ``postprocessor``.
    """

    def __init__(  # noqa: D107
        self,
        *,
        postprocessor: ClassificationPostprocessor | None = None,
        top_k: int = 5,
        apply_softmax: bool = True,
        class_names: dict[int, str] | None = None,
    ) -> None:
        self._postprocessor = postprocessor or ClassificationPostprocessor(top_k=top_k, apply_softmax=apply_softmax)
        self._class_names = class_names

    def __call__(self, output: InferenceOutput) -> ClassificationResult:
        """Run classification postprocessing and return a ClassificationResult."""
        predictions = self._postprocessor.postprocess(
            outputs=output.outputs,
            class_names=self._class_names,
        )
        return ClassificationResult(
            predictions=tuple(predictions),
            model_name=output.model_name,
            inference_time_ms=output.inference_time_ms,
        )


class DetailedClassificationPostProcessorBridge:
    """Bridges DetailedClassificationPostprocessor with full probability vectors.

    Returns ClassificationResult where each prediction carries
    ``all_probabilities`` -- the complete probability distribution over all classes.
    """

    def __init__(  # noqa: D107
        self,
        *,
        postprocessor: DetailedClassificationPostprocessor | None = None,
        top_k: int = 5,
        apply_softmax: bool = True,
        class_names: dict[int, str] | None = None,
    ) -> None:
        self._postprocessor = postprocessor or DetailedClassificationPostprocessor(
            top_k=top_k,
            apply_softmax=apply_softmax,
        )
        self._class_names = class_names

    def __call__(self, output: InferenceOutput) -> ClassificationResult:
        """Run detailed classification postprocessing with all probabilities."""
        predictions = self._postprocessor.postprocess(
            outputs=output.outputs,
            class_names=self._class_names,
        )
        return ClassificationResult(
            predictions=tuple(predictions),
            model_name=output.model_name,
            inference_time_ms=output.inference_time_ms,
        )

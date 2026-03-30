"""Inference input/output event models flowing through the pipeline graph.

These are regular event types routed by the graph's type-matching system.
InferenceInput is produced by preprocessing processors and consumed by
TritonInferenceProcessor. InferenceOutput is produced by inference and
consumed by downstream result-handling processors.

Numpy arrays are used directly (with arbitrary_types_allowed) because these
events flow through queues, not Redis serialization. Numpy arrays pickle
efficiently for cross-process transport.
"""

from __future__ import annotations

import numpy as np  # noqa: TC002 — pydantic needs runtime access for model_rebuild()
import numpy.typing as npt  # noqa: TC002
from pydantic import BaseModel, ConfigDict


class InferenceInput(BaseModel):
    """Input event for Triton inference processor.

    Carries the preprocessed tensor and metadata needed to dispatch
    a gRPC inference request to Triton.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    model_name: str
    model_version: str = ""
    raw_image: npt.NDArray[np.float32]
    input_name: str = "images"
    output_names: tuple[str, ...] = ("output0",)


class InferenceOutput(BaseModel):
    """Output event from Triton inference processor.

    Contains the raw inference results as numpy arrays keyed by output
    tensor name, plus timing information for metrics.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    model_name: str
    model_version: str
    outputs: dict[str, npt.NDArray[np.float32]]
    inference_time_ms: float


InferenceInput.model_rebuild()
InferenceOutput.model_rebuild()

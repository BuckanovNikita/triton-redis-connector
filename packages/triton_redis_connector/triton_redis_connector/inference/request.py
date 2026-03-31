"""Inference input/output event models flowing through the pipeline graph.

InferenceInput is defined in engine.events (so both preprocessing and
inference layers can import it). Re-exported here for backward compatibility.

InferenceOutput carries raw numpy arrays for cross-process transport.
"""

from __future__ import annotations

import numpy as np  # noqa: TC002 — pydantic needs runtime access for model_rebuild()
import numpy.typing as npt  # noqa: TC002
from pydantic import BaseModel, ConfigDict

from triton_redis_connector.engine.events import InferenceInput

__all__ = ["InferenceInput", "InferenceOutput"]


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


InferenceOutput.model_rebuild()

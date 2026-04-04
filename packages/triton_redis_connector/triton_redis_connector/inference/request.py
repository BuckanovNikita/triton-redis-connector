"""Inference input/output event models flowing through the pipeline graph.

TritonRequest is defined in engine.events (so both preprocessing and
inference layers can import it). Re-exported here for convenience.

InferenceOutput carries raw numpy arrays for cross-process transport.
"""

import numpy as np
import numpy.typing as npt
from pydantic import ConfigDict
from typed_flow.protocols.messaging import BaseEvent
from typed_flow.protocols.metrics import TimeMetric

from triton_redis_connector.engine.events import TritonRequest

__all__ = ["InferenceOutput", "TritonRequest"]


class InferenceOutput(BaseEvent):
    """Output event from Triton inference processor.

    Contains the raw inference results as numpy arrays keyed by output
    tensor name, plus timing information for metrics.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    model_name: str
    model_version: str
    outputs: dict[str, npt.NDArray[np.float32]]
    inference_time_ms: TimeMetric
    preprocess_scale: float = 1.0
    preprocess_pad: tuple[int, int] = (0, 0)


InferenceOutput.model_rebuild()

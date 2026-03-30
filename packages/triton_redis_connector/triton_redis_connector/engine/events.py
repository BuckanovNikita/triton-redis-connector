"""Base event types for the processing graph.

These are domain event models that flow through the pipeline. They are
distinct from the framework-internal Envelope wrapper.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class Event(BaseModel):
    """Base marker for all domain events in the processing graph.

    All user-defined events should inherit from this model.
    Frozen for immutability, strict to prevent silent coercion.
    """

    model_config = ConfigDict(frozen=True, strict=True)


class TensorSpec(BaseModel):
    """Specification of a tensor's shape and data type.

    Used to describe input/output tensor requirements for inference.
    """

    model_config = ConfigDict(frozen=True, strict=True)

    name: str
    shape: tuple[int, ...]
    dtype: str


class InferenceOutput(BaseModel):
    """Result of a Triton inference request.

    Carries the model name, version, and output tensor specifications.
    Raw output data is transported separately (as numpy arrays via the
    processing graph, not serialized in this model).
    """

    model_config = ConfigDict(frozen=True, strict=True)

    model_name: str
    model_version: str
    outputs: tuple[TensorSpec, ...]
    request_id: str = ""

"""Pydantic event models for the restreamer benchmark pipeline."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class SampleEvent(BaseModel):
    """User event written to the input stream by the load generator."""

    model_config = ConfigDict(frozen=True)

    sequence_id: int
    timestamp: float
    payload: str


class PassthroughResult(BaseModel):
    """Result model carrying the same fields through to the output stream."""

    model_config = ConfigDict(frozen=True)

    sequence_id: int
    timestamp: float
    payload: str

"""Base event types for the processing graph.

These are domain event models that flow through the pipeline. They are
distinct from the framework-internal Envelope wrapper.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class FrameworkMeta(BaseModel):
    """Framework-owned metadata carried through the pipeline on PipelineEvent.fw.

    Set by source connectors from the Redis envelope. Stripped by
    sinks before writing to the output stream. Subclass to extend
    with processing-stage-specific metadata.
    """

    model_config = ConfigDict(frozen=True)

    source_stream: str = ""
    message_id: str = ""
    priority: int = 0


class Unprocessed(BaseModel):
    """Sentinel payload for events that have not entered the processing pipeline.

    Sources produce ``PipelineEvent[T, Unprocessed]`` to carry framework
    metadata alongside raw user events before routing.
    """

    model_config = ConfigDict(frozen=True)


class PipelineEvent[TCtx: BaseModel, TPayload: BaseModel](BaseModel):
    """Typed composition of framework metadata, user context, and processing payload.

    Carries framework metadata (fw), the original user event (context),
    and the current processing stage payload. Framework routes by exact
    generic match on ``PipelineEvent[ConcreteCtx, ConcretePayload]``.

    The ``fw`` field is NOT serialized to Redis output -- sinks write
    only context and payload fields.
    """

    model_config = ConfigDict(frozen=True)

    fw: FrameworkMeta = FrameworkMeta()
    context: TCtx
    payload: TPayload

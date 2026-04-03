"""Custom pipeline processors for the multi-task pipeline.

Contains the TaskRouter (splits by task_type) and typed sinks for
detection and classification results.
"""

from __future__ import annotations

from loguru import logger
from redis import Redis  # noqa: TC002 — needed at runtime for sink __init__
from triton_redis_connector.engine.events import (
    ClassificationResult,
    DetectionResult,
    PipelineEvent,
)
from triton_redis_connector.protocols import BaseThreadProcessor, BaseThreadSink

from examples.multi_task_pipeline.events import (
    ClassificationImagePayload,
    DetectionImagePayload,
    ImageTaskInput,
    TaskMeta,
    TaskType,
)


class TaskRouter(
    BaseThreadProcessor[
        ImageTaskInput,
        PipelineEvent[TaskMeta, DetectionImagePayload] | PipelineEvent[TaskMeta, ClassificationImagePayload],
    ],
):
    """Routes incoming tasks to detection or classification branches.

    Parses the ``task_type`` field and produces a PipelineEvent with
    the appropriate typed payload for downstream preprocessors.
    """

    def __init__(self) -> None:
        """Initialize the task router."""
        super().__init__()

    def __call__(
        self, event: ImageTaskInput
    ) -> PipelineEvent[TaskMeta, DetectionImagePayload] | PipelineEvent[TaskMeta, ClassificationImagePayload] | None:
        """Route event to detection or classification branch based on task type."""
        try:
            task_type = TaskType(event.task_type)
        except ValueError:
            logger.error("Unknown task_type={}, dropping event", event.task_type)
            return None

        context = TaskMeta(
            fw=event.fw,
            task_type=task_type,
            image_path=event.image_path,
            source_id=event.source_id,
            priority=event.priority,
        )

        if task_type == TaskType.DETECT:
            return PipelineEvent[TaskMeta, DetectionImagePayload](
                context=context,
                payload=DetectionImagePayload(image_path=event.image_path),
            )

        return PipelineEvent[TaskMeta, ClassificationImagePayload](
            context=context,
            payload=ClassificationImagePayload(image_path=event.image_path),
        )


class DetectionResultSink(
    BaseThreadSink[PipelineEvent[TaskMeta, DetectionResult]],
):
    """Writes detection results to a dedicated Redis stream.

    Context fields (excluding ``fw``) become top-level hash entries.
    Payload (DetectionResult) is serialized as a ``result`` JSON field.
    """

    def __init__(self, client: Redis[bytes], stream: str, maxlen: int = 10_000) -> None:
        """Initialize sink with Redis client, stream name, and max length."""
        super().__init__()
        self._client = client
        self._stream = stream
        self._maxlen = maxlen

    def __call__(self, event: PipelineEvent[TaskMeta, DetectionResult]) -> None:
        """Serialize and write detection result to Redis stream."""
        ctx = event.context
        fields: dict[str, str] = {
            "task_type": ctx.task_type.value,
            "image_path": ctx.image_path,
            "source_id": ctx.source_id,
            "result": event.payload.model_dump_json(),
            "model_name": event.payload.model_name,
            "num_detections": str(len(event.payload.detections)),
        }
        self._client.xadd(self._stream, fields, maxlen=self._maxlen, approximate=True)
        logger.debug("Wrote detection result to {}", self._stream)


class ClassificationResultSink(
    BaseThreadSink[PipelineEvent[TaskMeta, ClassificationResult]],
):
    """Writes classification results to a dedicated Redis stream.

    Context fields (excluding ``fw``) become top-level hash entries.
    Payload (ClassificationResult) is serialized as a ``result`` JSON field.
    """

    def __init__(self, client: Redis[bytes], stream: str, maxlen: int = 10_000) -> None:
        """Initialize sink with Redis client, stream name, and max length."""
        super().__init__()
        self._client = client
        self._stream = stream
        self._maxlen = maxlen

    def __call__(self, event: PipelineEvent[TaskMeta, ClassificationResult]) -> None:
        """Serialize and write classification result to Redis stream."""
        ctx = event.context
        fields: dict[str, str] = {
            "task_type": ctx.task_type.value,
            "image_path": ctx.image_path,
            "source_id": ctx.source_id,
            "result": event.payload.model_dump_json(),
            "model_name": event.payload.model_name,
            "num_predictions": str(len(event.payload.predictions)),
        }
        self._client.xadd(self._stream, fields, maxlen=self._maxlen, approximate=True)
        logger.debug("Wrote classification result to {}", self._stream)

"""Write items to a Redis stream via XADD with automatic pydantic serialization."""

import time
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    import redis
from pydantic import BaseModel, ConfigDict, Field

StreamFields = dict[str | bytes, str | bytes]


class PopulateResult(BaseModel):
    """Result of populating a Redis stream."""

    model_config = ConfigDict(frozen=True)

    message_ids: list[str] = Field(default_factory=list)
    push_times: dict[str, float] = Field(default_factory=dict)
    count: int = Field(default=0, ge=0)


def serialize_model(model: BaseModel) -> StreamFields:
    """Serialize a pydantic model to Redis stream fields.

    Preserves ``bytes`` values as-is, stringifies everything else.
    """
    fields: StreamFields = {}
    for key, value in model.model_dump().items():
        if isinstance(value, bytes):
            fields[key] = value
        else:
            fields[key] = str(value).encode()
    return fields


def populate_stream(
    redis_client: "redis.Redis[bytes]",
    stream: str,
    items: Iterable[BaseModel | StreamFields],
    *,
    maxlen: int | None = None,
    approximate: bool = True,
) -> PopulateResult:
    """Push items to *stream* via XADD.

    Each item is either a pydantic ``BaseModel`` (serialized automatically)
    or a raw ``StreamFields`` dict passed through as-is.

    Progress bars are the caller's responsibility — wrap *items*
    in ``tqdm(gen, ...)`` before passing.
    """
    message_ids: list[str] = []
    push_times: dict[str, float] = {}

    for item in items:
        fields = serialize_model(item) if isinstance(item, BaseModel) else item
        push_time = time.monotonic()
        raw_id: Any = redis_client.xadd(
            name=stream,
            fields=fields,
            maxlen=maxlen,
            approximate=approximate,
        )
        decoded_id: str = raw_id.decode() if isinstance(raw_id, bytes) else str(raw_id)
        message_ids.append(decoded_id)
        push_times[decoded_id] = push_time

    count = len(message_ids)
    logger.info("Pushed {} messages to stream {}", count, stream)

    return PopulateResult(
        message_ids=message_ids,
        push_times=push_times,
        count=count,
    )

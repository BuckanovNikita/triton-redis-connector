"""Populate a Redis stream from an iterable of field dicts."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from collections.abc import Iterable

    import redis

StreamFields = dict[str | bytes, str | bytes]


class PopulateResult(BaseModel):
    """Result of populating a Redis stream."""

    model_config = ConfigDict(frozen=True)

    message_ids: list[str] = Field(default_factory=list)
    push_times: dict[str, float] = Field(default_factory=dict)
    count: int = Field(default=0, ge=0)


def populate_stream(
    redis_client: redis.Redis[bytes],
    stream: str,
    fields_iter: Iterable[StreamFields],
    *,
    maxlen: int | None = None,
    approximate: bool = True,
) -> PopulateResult:
    """Push each field dict from *fields_iter* to *stream* via XADD.

    Callers provide their own generators that yield field dicts.
    Progress bars are the caller's responsibility — wrap *fields_iter*
    in ``tqdm(gen, ...)`` before passing.

    """
    message_ids: list[str] = []
    push_times: dict[str, float] = {}

    for fields in fields_iter:
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

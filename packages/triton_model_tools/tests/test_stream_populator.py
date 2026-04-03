"""Tests for stream_populator module."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

from triton_model_tools.testing.stream_populator import (
    PopulateResult,
    StreamFields,
    populate_stream,
)

if TYPE_CHECKING:
    from collections.abc import Iterator


def _single_field() -> StreamFields:
    return {b"k": b"v"}


class TestPopulateStream:
    def test_basic_flow(self) -> None:
        mock_client = MagicMock()
        mock_client.xadd.side_effect = [b"1-0", b"2-0", b"3-0"]

        fields_list: list[StreamFields] = [
            {b"key": b"val1"},
            {b"key": b"val2"},
            {b"key": b"val3"},
        ]

        result = populate_stream(mock_client, "test-stream", fields_list)

        assert isinstance(result, PopulateResult)
        assert result.count == 3
        assert result.message_ids == ["1-0", "2-0", "3-0"]
        assert len(result.push_times) == 3
        assert all(mid in result.push_times for mid in result.message_ids)
        assert mock_client.xadd.call_count == 3

    def test_empty_iterable(self) -> None:
        mock_client = MagicMock()

        result = populate_stream(mock_client, "test-stream", [])

        assert result.count == 0
        assert result.message_ids == []
        assert result.push_times == {}
        mock_client.xadd.assert_not_called()

    def test_maxlen_passed_through(self) -> None:
        mock_client = MagicMock()
        mock_client.xadd.return_value = b"1-0"

        populate_stream(
            mock_client,
            "test-stream",
            [_single_field()],
            maxlen=50_000,
        )

        call_kwargs = mock_client.xadd.call_args.kwargs
        assert call_kwargs["maxlen"] == 50_000
        assert call_kwargs["approximate"] is True

    def test_maxlen_none_by_default(self) -> None:
        mock_client = MagicMock()
        mock_client.xadd.return_value = b"1-0"

        populate_stream(mock_client, "test-stream", [_single_field()])

        call_kwargs = mock_client.xadd.call_args.kwargs
        assert call_kwargs["maxlen"] is None

    def test_bytes_message_id_decoded(self) -> None:
        mock_client = MagicMock()
        mock_client.xadd.return_value = b"1234567890-0"

        result = populate_stream(mock_client, "s", [_single_field()])

        assert result.message_ids == ["1234567890-0"]

    def test_str_message_id_handled(self) -> None:
        mock_client = MagicMock()
        mock_client.xadd.return_value = "1234567890-0"

        result = populate_stream(mock_client, "s", [_single_field()])

        assert result.message_ids == ["1234567890-0"]

    def test_push_times_are_monotonically_increasing(self) -> None:
        mock_client = MagicMock()
        mock_client.xadd.side_effect = [b"1-0", b"2-0"]
        fields: list[StreamFields] = [{b"k": b"v1"}, {b"k": b"v2"}]

        result = populate_stream(mock_client, "s", fields)

        times = [result.push_times[mid] for mid in result.message_ids]
        assert times[0] <= times[1]

    def test_generator_consumed_lazily(self) -> None:
        mock_client = MagicMock()
        mock_client.xadd.side_effect = [b"1-0", b"2-0"]
        call_order: list[str] = []

        def gen() -> Iterator[StreamFields]:
            call_order.append("yield-1")
            yield {b"k": b"v1"}
            call_order.append("yield-2")
            yield {b"k": b"v2"}

        populate_stream(mock_client, "s", gen())

        assert call_order == ["yield-1", "yield-2"]

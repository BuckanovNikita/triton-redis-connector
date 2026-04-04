"""Tests for stream_populator module."""

from collections.abc import Iterator
from unittest.mock import MagicMock

from pydantic import BaseModel, ConfigDict
from triton_model_tools.testing.stream_populator import (
    ImageStreamEntry,
    InspectionStreamEntry,
    PopulateResult,
    StreamFields,
    populate_stream,
)


def _single_field() -> StreamFields:
    return {b"k": b"v"}


class TestPopulateStream:
    def test_basic_flow_with_dicts(self) -> None:
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


class TestBaseModelSerialization:
    def test_pydantic_model_serialized(self) -> None:
        mock_client = MagicMock()
        mock_client.xadd.return_value = b"1-0"

        entry = ImageStreamEntry(
            data=b"\xff\xd8image-data",
            request_id="abc123",
            model_name="yolo11s",
            image_name="test.jpg",
        )

        populate_stream(mock_client, "s", [entry])

        fields = mock_client.xadd.call_args.kwargs["fields"]
        assert fields["data"] == b"\xff\xd8image-data"
        assert fields["request_id"] == b"abc123"
        assert fields["model_name"] == b"yolo11s"
        assert fields["image_name"] == b"test.jpg"

    def test_bytes_fields_preserved(self) -> None:
        mock_client = MagicMock()
        mock_client.xadd.return_value = b"1-0"
        binary_data = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100

        entry = ImageStreamEntry(
            data=binary_data,
            request_id="id1",
            model_name="m",
            image_name="img.png",
        )

        populate_stream(mock_client, "s", [entry])

        fields = mock_client.xadd.call_args.kwargs["fields"]
        assert fields["data"] == binary_data
        assert isinstance(fields["data"], bytes)

    def test_non_bytes_fields_stringified(self) -> None:
        mock_client = MagicMock()
        mock_client.xadd.return_value = b"1-0"

        entry = InspectionStreamEntry(
            camera_no=3,
            image_path="/data/img.jpg",
            frame_counter=42,
            receipt_ts=1234567890.123,
            metal_type="steel",
        )

        populate_stream(mock_client, "s", [entry])

        fields = mock_client.xadd.call_args.kwargs["fields"]
        assert fields["camera_no"] == b"3"
        assert fields["image_path"] == b"/data/img.jpg"
        assert fields["frame_counter"] == b"42"
        assert fields["metal_type"] == b"steel"
        assert fields["event_type"] == b"image"

    def test_all_fields_serialized(self) -> None:
        """All fields on a user model are serialized to Redis."""
        mock_client = MagicMock()
        mock_client.xadd.return_value = b"1-0"

        class FakeEvent(BaseModel):
            model_config = ConfigDict(frozen=True)
            tag: str = "metadata"
            data: str = "payload"

        populate_stream(mock_client, "s", [FakeEvent()])

        fields = mock_client.xadd.call_args.kwargs["fields"]
        assert fields["tag"] == b"metadata"
        assert fields["data"] == b"payload"

    def test_model_without_fw_includes_all_fields(self) -> None:
        mock_client = MagicMock()
        mock_client.xadd.return_value = b"1-0"

        entry = ImageStreamEntry(
            data=b"img",
            request_id="r",
            model_name="m",
            image_name="n",
        )

        populate_stream(mock_client, "s", [entry])

        fields = mock_client.xadd.call_args.kwargs["fields"]
        assert len(fields) == 4

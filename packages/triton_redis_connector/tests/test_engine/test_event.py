"""Tests for Envelope event wrapper."""

from __future__ import annotations

import time

from triton_redis_connector.protocols import Envelope


class TestEnvelopeCreation:
    def test_default_created_at_uses_monotonic(self) -> None:
        before = time.monotonic()
        envelope: Envelope[str] = Envelope(payload="test")
        after = time.monotonic()
        assert before <= envelope.created_at <= after

    def test_custom_created_at(self) -> None:
        envelope: Envelope[str] = Envelope(payload="test", created_at=42.0)
        assert envelope.created_at == 42.0


class TestTrackState:
    def test_appends_state_with_timestamp(self) -> None:
        envelope: Envelope[str] = Envelope(payload="test")
        before = time.monotonic()
        envelope.track_state("decoded")
        after = time.monotonic()

        assert len(envelope.states) == 1
        state_name, state_time = envelope.states[0]
        assert state_name == "decoded"
        assert before <= state_time <= after

    def test_multiple_states_accumulate(self) -> None:
        envelope: Envelope[str] = Envelope(payload="test")
        envelope.track_state("received")
        envelope.track_state("decoded")
        envelope.track_state("inferred")

        assert len(envelope.states) == 3
        assert [name for name, _ in envelope.states] == ["received", "decoded", "inferred"]

    def test_state_timestamps_are_monotonically_increasing(self) -> None:
        envelope: Envelope[str] = Envelope(payload="test")
        envelope.track_state("first")
        envelope.track_state("second")

        _, first_time = envelope.states[0]
        _, second_time = envelope.states[1]
        assert first_time <= second_time


class TestTransitTimeMs:
    def test_returns_positive_elapsed_time(self) -> None:
        envelope: Envelope[str] = Envelope(payload="test")
        transit = envelope.transit_time_ms()
        assert transit >= 0.0

    def test_increases_over_time(self) -> None:
        envelope: Envelope[str] = Envelope(payload="test")
        first = envelope.transit_time_ms()
        time.sleep(0.005)
        second = envelope.transit_time_ms()
        assert second > first

    def test_unit_is_milliseconds(self) -> None:
        fixed_start = time.monotonic() - 1.0
        envelope: Envelope[str] = Envelope(payload="test", created_at=fixed_start)
        transit = envelope.transit_time_ms()
        assert transit >= 900.0


class TestDerive:
    def test_derive_changes_payload(self) -> None:
        original: Envelope[str] = Envelope(payload="raw")
        derived: Envelope[int] = original.derive(42)
        assert derived.payload == 42

    def test_derive_copies_states(self) -> None:
        original: Envelope[str] = Envelope(payload="raw")
        original.track_state("received")
        original.track_state("decoded")

        derived: Envelope[int] = original.derive(42)
        assert len(derived.states) == 2
        assert [name for name, _ in derived.states] == ["received", "decoded"]

    def test_derive_state_list_is_independent(self) -> None:
        original: Envelope[str] = Envelope(payload="raw")
        original.track_state("received")

        derived: Envelope[int] = original.derive(42)
        derived.track_state("processed")

        assert len(original.states) == 1
        assert len(derived.states) == 2


class TestOrdering:
    def test_older_envelope_is_less_than_newer(self) -> None:
        older: Envelope[str] = Envelope(payload="old", created_at=1.0)
        newer: Envelope[str] = Envelope(payload="new", created_at=2.0)
        assert older < newer
        assert not newer < older

    def test_sorting_produces_oldest_first(self) -> None:
        envelopes = [
            Envelope(payload="c", created_at=3.0),
            Envelope(payload="a", created_at=1.0),
            Envelope(payload="b", created_at=2.0),
        ]
        sorted_envelopes = sorted(envelopes)
        assert [e.payload for e in sorted_envelopes] == ["a", "b", "c"]

    def test_different_payload_types_comparable(self) -> None:
        str_env: Envelope[str] = Envelope(payload="text", created_at=1.0)
        int_env: Envelope[int] = Envelope(payload=42, created_at=2.0)
        assert str_env < int_env

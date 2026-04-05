"""Tests for QueueBus — queue creation, put/get, and backpressure."""

import queue

import pytest
from typed_flow.config.schema import QueueConfig
from typed_flow.engine.queue_bus import QueueBus, QueueOverflow


@pytest.fixture
def bus() -> QueueBus:
    return QueueBus(config=QueueConfig(maxsize=3))


class TestQueueCreation:
    def test_get_queue_creates_new(self, bus: QueueBus) -> None:
        q = bus.get_queue("test_q")
        assert isinstance(q, queue.Queue)
        assert q.maxsize == 3

    def test_get_queue_returns_same_instance(self, bus: QueueBus) -> None:
        q1 = bus.get_queue("test_q")
        q2 = bus.get_queue("test_q")
        assert q1 is q2

    def test_get_queue_custom_maxsize(self, bus: QueueBus) -> None:
        q = bus.get_queue("custom", maxsize=10)
        assert q.maxsize == 10

    def test_get_queue_ignores_maxsize_on_existing(self, bus: QueueBus) -> None:
        q1 = bus.get_queue("test_q", maxsize=5)
        q2 = bus.get_queue("test_q", maxsize=99)
        assert q1 is q2
        assert q2.maxsize == 5

    def test_queue_names_lists_created_queues(self, bus: QueueBus) -> None:
        bus.get_queue("alpha")
        bus.get_queue("beta")
        names = bus.queue_names
        assert "alpha" in names
        assert "beta" in names

    def test_queue_names_empty_initially(self, bus: QueueBus) -> None:
        assert bus.queue_names == []


class TestPutGet:
    def test_put_and_get_roundtrip(self, bus: QueueBus) -> None:
        overflow = bus.put("test_q", "item1")
        assert overflow is None
        result = bus.get("test_q", timeout=0.1)
        assert result == "item1"

    def test_put_multiple_items(self, bus: QueueBus) -> None:
        bus.put("test_q", "a")
        bus.put("test_q", "b")
        bus.put("test_q", "c")
        assert bus.get("test_q", timeout=0.1) == "a"
        assert bus.get("test_q", timeout=0.1) == "b"
        assert bus.get("test_q", timeout=0.1) == "c"

    def test_get_raises_empty_on_timeout(self, bus: QueueBus) -> None:
        bus.get_queue("empty_q")
        with pytest.raises(queue.Empty):
            bus.get("empty_q", timeout=0.01)


class TestBackpressure:
    def test_drop_on_full_queue_returns_overflow(self, bus: QueueBus) -> None:
        bus.put("test_q", "1")
        bus.put("test_q", "2")
        bus.put("test_q", "3")
        overflow = bus.put("test_q", "4_dropped")
        assert overflow is not None
        assert isinstance(overflow, QueueOverflow)
        assert overflow.queue_name == "test_q"

    def test_dropped_item_not_in_queue(self, bus: QueueBus) -> None:
        bus.put("test_q", "1")
        bus.put("test_q", "2")
        bus.put("test_q", "3")
        bus.put("test_q", "dropped")
        items = []
        while True:
            try:
                items.append(bus.get("test_q", timeout=0.01))
            except queue.Empty:
                break
        assert "dropped" not in items
        assert len(items) == 3

    def test_successful_put_returns_none(self, bus: QueueBus) -> None:
        assert bus.put("test_q", "item") is None

    def test_multiple_overflows_each_return_indicator(self, bus: QueueBus) -> None:
        for i in range(3):
            bus.put("test_q", str(i))
        m1 = bus.put("test_q", "drop1")
        m2 = bus.put("test_q", "drop2")
        assert m1 is not None
        assert m2 is not None
        assert m1.queue_name == "test_q"
        assert m2.queue_name == "test_q"


class TestQueueMetrics:
    def test_queue_depths_returns_sizes(self, bus: QueueBus) -> None:
        bus.put("q1", "a")
        bus.put("q1", "b")
        bus.put("q2", "x")
        depths = bus.queue_depths()
        assert depths["q1"] == 2
        assert depths["q2"] == 1

    def test_queue_depths_empty_queues(self, bus: QueueBus) -> None:
        bus.get_queue("empty_q")
        depths = bus.queue_depths()
        assert depths["empty_q"] == 0

    def test_queue_depths_empty_when_no_queues(self, bus: QueueBus) -> None:
        assert bus.queue_depths() == {}

    def test_drop_counts_zero_initially(self, bus: QueueBus) -> None:
        bus.get_queue("q")
        assert bus.drop_counts() == {}

    def test_drop_counts_tracks_overflows(self, bus: QueueBus) -> None:
        for i in range(3):
            bus.put("q", str(i))
        bus.put("q", "drop1")
        bus.put("q", "drop2")
        counts = bus.drop_counts()
        assert counts["q"] == 2

    def test_drop_counts_per_queue(self, bus: QueueBus) -> None:
        for i in range(3):
            bus.put("q1", str(i))
            bus.put("q2", str(i))
        bus.put("q1", "drop")
        bus.put("q2", "drop1")
        bus.put("q2", "drop2")
        counts = bus.drop_counts()
        assert counts["q1"] == 1
        assert counts["q2"] == 2

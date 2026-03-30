"""Tests for RetryHandler: exponential backoff, callbacks, semaphore release."""

from __future__ import annotations

import threading
import time

from triton_redis_connector.inference.retry import RetryHandler


class TestRetryHandler:
    def test_success_on_first_attempt(self) -> None:
        handler = RetryHandler(max_attempts=3, base_delay_s=0.01, backoff_multiplier=2.0)
        semaphore = threading.Semaphore(1)
        semaphore.acquire()

        success_called = threading.Event()
        failure_called = threading.Event()

        def execute_fn() -> None:
            pass

        handler.execute_with_retry(
            execute_fn=execute_fn,
            on_success=lambda _: success_called.set(),
            on_failure=lambda _: failure_called.set(),
            semaphore=semaphore,
        )

        assert success_called.is_set()
        assert not failure_called.is_set()
        assert semaphore._value == 1

    def test_failure_after_all_retries_exhausted(self) -> None:
        handler = RetryHandler(max_attempts=3, base_delay_s=0.01, backoff_multiplier=2.0)
        semaphore = threading.Semaphore(1)
        semaphore.acquire()

        failure_event = threading.Event()
        failure_messages: list[str] = []

        def always_fail() -> None:
            msg = "always fails"
            raise RuntimeError(msg)

        def on_failure(msg: str) -> None:
            failure_messages.append(msg)
            failure_event.set()

        handler.execute_with_retry(
            execute_fn=always_fail,
            on_success=lambda _: None,
            on_failure=on_failure,
            semaphore=semaphore,
        )

        failure_event.wait(timeout=5.0)

        assert failure_event.is_set()
        assert len(failure_messages) == 1
        assert "always fails" in failure_messages[0]
        assert semaphore._value == 1

    def test_success_after_transient_failures(self) -> None:
        handler = RetryHandler(max_attempts=3, base_delay_s=0.01, backoff_multiplier=2.0)
        semaphore = threading.Semaphore(1)
        semaphore.acquire()

        call_count = 0
        success_event = threading.Event()

        def fail_twice() -> None:
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                msg = "transient error"
                raise RuntimeError(msg)

        handler.execute_with_retry(
            execute_fn=fail_twice,
            on_success=lambda _: success_event.set(),
            on_failure=lambda _: None,
            semaphore=semaphore,
        )

        success_event.wait(timeout=5.0)

        assert success_event.is_set()
        assert call_count == 3
        assert semaphore._value == 1

    def test_exponential_backoff_delays(self) -> None:
        handler = RetryHandler(max_attempts=4, base_delay_s=0.1, backoff_multiplier=4.0)

        assert handler.delay_for_attempt(0) == pytest.approx(0.1)
        assert handler.delay_for_attempt(1) == pytest.approx(0.4)
        assert handler.delay_for_attempt(2) == pytest.approx(1.6)

    def test_three_attempts_with_timing(self) -> None:
        handler = RetryHandler(max_attempts=3, base_delay_s=0.05, backoff_multiplier=2.0)
        semaphore = threading.Semaphore(1)
        semaphore.acquire()

        attempt_times: list[float] = []
        failure_event = threading.Event()

        def fail_and_record() -> None:
            attempt_times.append(time.monotonic())
            msg = "fail"
            raise RuntimeError(msg)

        handler.execute_with_retry(
            execute_fn=fail_and_record,
            on_success=lambda _: None,
            on_failure=lambda _: failure_event.set(),
            semaphore=semaphore,
        )

        failure_event.wait(timeout=5.0)

        assert len(attempt_times) == 3
        delay_1 = attempt_times[1] - attempt_times[0]
        delay_2 = attempt_times[2] - attempt_times[1]
        assert delay_1 >= 0.04
        assert delay_2 >= 0.08

    def test_semaphore_released_on_success(self) -> None:
        handler = RetryHandler(max_attempts=3)
        semaphore = threading.Semaphore(2)
        semaphore.acquire()

        done = threading.Event()

        handler.execute_with_retry(
            execute_fn=lambda: None,
            on_success=lambda _: done.set(),
            on_failure=lambda _: None,
            semaphore=semaphore,
        )

        done.wait(timeout=1.0)
        assert semaphore._value == 2

    def test_semaphore_released_on_failure(self) -> None:
        handler = RetryHandler(max_attempts=1, base_delay_s=0.01)
        semaphore = threading.Semaphore(2)
        semaphore.acquire()

        done = threading.Event()

        def always_fail() -> None:
            msg = "boom"
            raise RuntimeError(msg)

        handler.execute_with_retry(
            execute_fn=always_fail,
            on_success=lambda _: None,
            on_failure=lambda _: done.set(),
            semaphore=semaphore,
        )

        done.wait(timeout=1.0)
        assert semaphore._value == 2

    def test_cancel_pending_timers(self) -> None:
        handler = RetryHandler(max_attempts=3, base_delay_s=10.0, backoff_multiplier=1.0)
        semaphore = threading.Semaphore(1)
        semaphore.acquire()

        call_count = 0

        def fail_once() -> None:
            nonlocal call_count
            call_count += 1
            msg = "fail"
            raise RuntimeError(msg)

        handler.execute_with_retry(
            execute_fn=fail_once,
            on_success=lambda _: None,
            on_failure=lambda _: None,
            semaphore=semaphore,
        )

        handler.cancel_pending()
        assert call_count == 1


import pytest  # noqa: E402

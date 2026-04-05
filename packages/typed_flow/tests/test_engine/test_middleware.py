"""Tests for RetryMiddleware exponential backoff and failure handling."""

from unittest.mock import MagicMock, patch

import pytest
from typed_flow.engine.middleware import RetryMiddleware


class TestRetryMiddleware:
    def test_success_on_first_attempt(self) -> None:
        inner = MagicMock(return_value=42)
        mw = RetryMiddleware(inner)
        result = mw("input")
        assert result == 42
        inner.assert_called_once_with("input")

    def test_retries_on_exception_then_succeeds(self) -> None:
        inner = MagicMock(side_effect=[ValueError("fail"), ValueError("fail"), 99])
        mw = RetryMiddleware(inner, max_attempts=3, base_delay_s=0.001)
        result = mw("x")
        assert result == 99
        assert inner.call_count == 3

    def test_exhausts_attempts_and_reraises(self) -> None:
        inner = MagicMock(side_effect=ValueError("boom"))
        mw = RetryMiddleware(inner, max_attempts=3, base_delay_s=0.001)
        with pytest.raises(ValueError, match="boom"):
            mw("x")
        assert inner.call_count == 3

    def test_on_failure_callback_called_after_exhaustion(self) -> None:
        inner = MagicMock(side_effect=ValueError("err"))
        on_fail = MagicMock(return_value="fallback")
        mw = RetryMiddleware(inner, max_attempts=2, base_delay_s=0.001, on_failure=on_fail)
        result = mw("event_data")
        assert result == "fallback"
        on_fail.assert_called_once_with("event_data", "err")

    def test_exponential_backoff_delays(self) -> None:
        inner = MagicMock(side_effect=[ValueError(), ValueError(), 1])
        with patch("typed_flow.engine.middleware.time.sleep") as mock_sleep:
            mw = RetryMiddleware(inner, max_attempts=3, base_delay_s=0.1, backoff_multiplier=4.0)
            mw("x")
        assert mock_sleep.call_count == 2
        delays = [call.args[0] for call in mock_sleep.call_args_list]
        assert abs(delays[0] - 0.1) < 1e-9
        assert abs(delays[1] - 0.4) < 1e-9

    def test_max_attempts_one_no_retry(self) -> None:
        inner = MagicMock(side_effect=ValueError("immediate"))
        with patch("typed_flow.engine.middleware.time.sleep") as mock_sleep:
            mw = RetryMiddleware(inner, max_attempts=1)
            with pytest.raises(ValueError, match="immediate"):
                mw("x")
        mock_sleep.assert_not_called()
        inner.assert_called_once()

    def test_returns_none_when_inner_returns_none(self) -> None:
        inner = MagicMock(return_value=None)
        mw = RetryMiddleware(inner)
        result = mw("x")
        assert result is None
        inner.assert_called_once()

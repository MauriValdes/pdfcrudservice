"""RED contract tests for pure worker retry-time planning."""

from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from math import inf, nan
from unittest.mock import Mock

import pytest


UTC = timezone.utc
NOW = datetime(2026, 11, 28, 12, 0, 0, tzinfo=UTC)
START = NOW


def _retry_api():
    from src.worker.retry_policy import (
        RetryExhausted,
        RetryExhaustionReason,
        RetryScheduled,
        plan_retry,
    )

    return RetryExhausted, RetryExhaustionReason, RetryScheduled, plan_retry


def _plan(*, attempt_count=1, cycle_started_at=START, now=NOW, **kwargs):
    _, _, _, plan_retry = _retry_api()
    return plan_retry(
        attempt_count=attempt_count,
        cycle_started_at=cycle_started_at,
        now=now,
        **kwargs,
    )


@pytest.mark.parametrize(
    ("attempt_count", "low", "high", "sample"),
    [
        (1, 1.0, 2.0, 1.0),
        (1, 1.0, 2.0, 1.5),
        (1, 1.0, 2.0, 2.0),
        (2, 2.0, 4.0, 2.0),
        (2, 2.0, 4.0, 3.0),
        (2, 2.0, 4.0, 4.0),
        (3, 4.0, 8.0, 4.0),
        (3, 4.0, 8.0, 6.0),
        (3, 4.0, 8.0, 8.0),
        (4, 8.0, 16.0, 8.0),
        (4, 8.0, 16.0, 12.0),
        (4, 8.0, 16.0, 16.0),
    ],
)
def test_local_backoff_uses_attempt_specific_uniform_bounds(
    attempt_count, low, high, sample
):
    _, _, RetryScheduled, _ = _retry_api()
    uniform = Mock(return_value=sample)

    result = _plan(attempt_count=attempt_count, uniform=uniform)

    assert isinstance(result, RetryScheduled)
    assert uniform.call_args.args == (low, min(high, 30.0))
    assert result.next_retry_at == NOW + timedelta(seconds=sample)
    assert result.retry_after_used is False


@pytest.mark.parametrize("attempt_count", [True, False, -1, 0, 1.5, 6, "1"])
def test_attempt_count_requires_an_integer_from_one_through_five(attempt_count):
    with pytest.raises((TypeError, ValueError)):
        _plan(attempt_count=attempt_count)


@pytest.mark.parametrize(
    ("attempt_count", "now", "reason_name"),
    [
        (5, NOW, "ATTEMPTS_EXHAUSTED"),
        (1, NOW + timedelta(minutes=10), "WINDOW_EXPIRED"),
        (1, NOW + timedelta(minutes=11), "WINDOW_EXPIRED"),
        (5, NOW + timedelta(minutes=10), "ATTEMPTS_EXHAUSTED"),
    ],
)
def test_exhaustion_is_typed_prioritized_and_does_not_sample_or_need_retry_after(
    attempt_count, now, reason_name
):
    RetryExhausted, RetryExhaustionReason, _, _ = _retry_api()
    uniform = Mock(side_effect=AssertionError("must not sample exhausted retry"))

    result = _plan(
        attempt_count=attempt_count,
        now=now,
        retry_after="malformed",
        uniform=uniform,
    )

    assert isinstance(result, RetryExhausted)
    assert result.reason is getattr(RetryExhaustionReason, reason_name)
    uniform.assert_not_called()


@pytest.mark.parametrize(
    ("cycle_started_at", "now"),
    [
        (NOW + timedelta(microseconds=1), NOW),
        ("2026-11-28T12:00:00Z", NOW),
        (NOW, None),
    ],
)
def test_invalid_dates_and_future_cycle_start_are_rejected(cycle_started_at, now):
    with pytest.raises((TypeError, ValueError)):
        _plan(cycle_started_at=cycle_started_at, now=now)


def test_naive_mongo_utc_dates_and_aware_offsets_normalize_to_utc():
    _, _, RetryScheduled, _ = _retry_api()
    naive_start = datetime(2026, 11, 28, 12, 0, 0)
    offset_now = datetime(2026, 11, 28, 7, 0, 0, tzinfo=timezone(timedelta(hours=-5)))

    result = _plan(
        cycle_started_at=naive_start,
        now=offset_now,
        uniform=lambda low, high: low,
    )

    assert isinstance(result, RetryScheduled)
    assert result.next_retry_at == NOW + timedelta(seconds=1)
    assert result.next_retry_at.tzinfo is UTC


def test_cycle_window_overflow_is_rejected_before_arithmetic_escapes():
    maximum = datetime.max.replace(tzinfo=UTC)
    cycle_started_at = maximum - timedelta(minutes=5)
    now = maximum - timedelta(minutes=2)

    with pytest.raises((OverflowError, ValueError)):
        _plan(cycle_started_at=cycle_started_at, now=now)


def test_uniform_source_must_be_callable():
    with pytest.raises((TypeError, ValueError)):
        _plan(uniform=None)


@pytest.mark.parametrize("sample", [None, True, "1.5", inf, -inf, nan, 0.99, 2.01])
def test_invalid_uniform_results_are_rejected(sample):
    with pytest.raises((TypeError, ValueError)):
        _plan(uniform=lambda low, high: sample)


@pytest.mark.parametrize(
    ("retry_after", "sample", "expected_seconds", "used"),
    [
        ("0", 2.0, 2.0, False),
        ("5", 2.0, 5.0, True),
        ("45", 2.0, 45.0, True),
        ("-1", 2.0, 2.0, False),
        ("1.5", 2.0, 2.0, False),
        ("nonsense", 2.0, 2.0, False),
        ("", 2.0, 2.0, False),
        (None, 2.0, 2.0, False),
    ],
)
def test_retry_after_seconds_and_invalid_headers(
    retry_after, sample, expected_seconds, used
):
    _, _, RetryScheduled, _ = _retry_api()

    result = _plan(
        retry_after=retry_after,
        uniform=lambda low, high: sample,
    )

    assert isinstance(result, RetryScheduled)
    assert result.next_retry_at == NOW + timedelta(seconds=expected_seconds)
    assert result.retry_after_used is used


@pytest.mark.parametrize(
    ("header_delta", "sample", "expected_seconds", "used"),
    [
        (5, 2.0, 5.0, True),
        (-5, 2.0, 2.0, False),
        (0, 2.0, 2.0, False),
    ],
)
def test_http_date_retry_after_uses_max_and_past_dates_do_not_reduce_backoff(
    header_delta, sample, expected_seconds, used
):
    target = NOW + timedelta(seconds=header_delta)
    retry_after = format_datetime(target, usegmt=True)
    _, _, RetryScheduled, _ = _retry_api()

    result = _plan(
        retry_after=retry_after,
        uniform=lambda low, high: sample,
    )

    assert isinstance(result, RetryScheduled)
    assert result.next_retry_at == NOW + timedelta(seconds=expected_seconds)
    assert result.retry_after_used is used


def test_unrepresentably_large_seconds_saturate_at_cycle_end_without_overflow():
    _, _, RetryScheduled, _ = _retry_api()
    huge_retry_after = "9" * 5000

    result = _plan(
        retry_after=huge_retry_after,
        uniform=lambda low, high: low,
    )

    assert isinstance(result, RetryScheduled)
    assert result.next_retry_at == START + timedelta(minutes=10)
    assert result.retry_after_used is True


def test_retry_after_date_outside_window_is_capped_without_overflow():
    _, _, RetryScheduled, _ = _retry_api()
    retry_after = "Fri, 31 Dec 9999 23:59:59 GMT"

    result = _plan(
        retry_after=retry_after,
        uniform=lambda low, high: low,
    )

    assert isinstance(result, RetryScheduled)
    assert result.next_retry_at == START + timedelta(minutes=10)
    assert result.retry_after_used is True


def test_both_delays_capped_to_same_cycle_end_do_not_mark_retry_after_used():
    _, _, RetryScheduled, _ = _retry_api()
    cycle_started_at = NOW - timedelta(minutes=9, seconds=59)

    result = _plan(
        cycle_started_at=cycle_started_at,
        retry_after="45",
        uniform=lambda low, high: high,
    )

    assert isinstance(result, RetryScheduled)
    assert result.next_retry_at == cycle_started_at + timedelta(minutes=10)
    assert result.retry_after_used is False


def test_retry_after_that_increases_wait_to_cycle_end_is_marked_used():
    _, _, RetryScheduled, _ = _retry_api()
    cycle_started_at = NOW - timedelta(minutes=9, seconds=58)

    result = _plan(
        cycle_started_at=cycle_started_at,
        retry_after="45",
        uniform=lambda low, high: low,
    )

    assert isinstance(result, RetryScheduled)
    assert result.next_retry_at == cycle_started_at + timedelta(minutes=10)
    assert result.retry_after_used is True


def test_result_is_bson_millisecond_aligned_without_early_rounding():
    _, _, RetryScheduled, _ = _retry_api()

    result = _plan(uniform=lambda low, high: 1.234567)

    assert isinstance(result, RetryScheduled)
    assert result.next_retry_at.microsecond % 1000 == 0
    assert result.next_retry_at >= NOW + timedelta(seconds=1.234567)
    assert result.next_retry_at <= NOW + timedelta(seconds=1.235)
    assert result.next_retry_at > NOW

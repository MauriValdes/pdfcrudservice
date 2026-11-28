"""Pure retry timing policy for one processing cycle."""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from enum import Enum
import math
from numbers import Real
import random
import re
from typing import Callable, TypeAlias


_CYCLE_WINDOW = timedelta(minutes=10)
_MAX_LOCAL_BACKOFF_SECONDS = 30.0
_BACKOFF_BASE_SECONDS = {
    1: 2.0,
    2: 4.0,
    3: 8.0,
    4: 16.0,
}
_MICROSECONDS_PER_SECOND = 1_000_000
_MICROSECONDS_PER_MILLISECOND = 1_000
_ASCII_INTEGER = re.compile(r"[0-9]+\Z", re.ASCII)


class RetryExhaustionReason(str, Enum):
    """Reason no further retry can be scheduled in this cycle."""

    ATTEMPTS_EXHAUSTED = "attempts_exhausted"
    WINDOW_EXPIRED = "window_expired"


@dataclass(frozen=True, slots=True)
class RetryScheduled:
    """A retry timestamp computed within the cycle's time window."""

    next_retry_at: datetime
    retry_after_used: bool


@dataclass(frozen=True, slots=True)
class RetryExhausted:
    """A retry cannot be scheduled because its budget is exhausted."""

    reason: RetryExhaustionReason


RetryPlan: TypeAlias = RetryScheduled | RetryExhausted


def _normalize_utc(value: datetime, name: str) -> datetime:
    """Normalize aware dates to UTC and interpret naive BSON dates as UTC."""
    if not isinstance(value, datetime):
        raise TypeError(f"{name} must be a datetime")

    try:
        if value.tzinfo is None or value.utcoffset() is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a valid datetime") from error


def _timedelta_microseconds(value: timedelta) -> int:
    """Return an exact integer duration without floating-point conversion."""
    return (
        (value.days * 24 * 60 * 60 + value.seconds)
        * _MICROSECONDS_PER_SECOND
        + value.microseconds
    )


def _ceil_millisecond(value: datetime) -> datetime:
    """Round a UTC instant upward to BSON millisecond precision."""
    remainder = value.microsecond % _MICROSECONDS_PER_MILLISECOND
    if remainder == 0:
        return value
    increment = _MICROSECONDS_PER_MILLISECOND - remainder
    return value + timedelta(microseconds=increment)


def _bounded_timestamp(
    now: datetime,
    delay_microseconds: int,
    cycle_end: datetime,
    remaining_microseconds: int,
) -> datetime:
    """Add a bounded delay, round upward, and never pass the cycle end."""
    if delay_microseconds >= remaining_microseconds:
        return cycle_end

    target = now + timedelta(microseconds=delay_microseconds)
    rounded_target = _ceil_millisecond(target)
    return min(rounded_target, cycle_end)


def _local_backoff_microseconds(
    attempt_count: int,
    uniform: Callable[[float, float], float],
) -> int:
    """Sample and validate local jitter, returning a non-early duration."""
    base_seconds = _BACKOFF_BASE_SECONDS[attempt_count]
    lower_bound = base_seconds / 2
    upper_bound = min(base_seconds, _MAX_LOCAL_BACKOFF_SECONDS)

    sample = uniform(lower_bound, upper_bound)
    if isinstance(sample, bool) or not isinstance(sample, Real):
        raise TypeError("uniform must return a finite real number")
    try:
        sample_seconds = float(sample)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError("uniform returned an invalid backoff") from error
    if (
        not math.isfinite(sample_seconds)
        or sample_seconds < lower_bound
        or sample_seconds > upper_bound
    ):
        raise ValueError("uniform returned a backoff outside its bounds")

    return math.ceil(sample_seconds * _MICROSECONDS_PER_SECOND)


def _retry_after_timestamp(
    retry_after: str | None,
    *,
    now: datetime,
    cycle_end: datetime,
    remaining_microseconds: int,
) -> datetime | None:
    """Parse Retry-After safely, returning its cycle-bounded UTC timestamp."""
    if retry_after is None:
        return None

    value = retry_after.strip()
    if not value:
        return None

    if _ASCII_INTEGER.fullmatch(value):
        significant_digits = value.lstrip("0") or "0"
        largest_in_window = (
            remaining_microseconds + _MICROSECONDS_PER_SECOND - 1
        ) // _MICROSECONDS_PER_SECOND
        largest_digits = str(largest_in_window)

        # The remaining retry window is at most ten minutes. Compare decimal
        # strings first so arbitrarily long headers are never converted to int.
        if (
            len(significant_digits) > len(largest_digits)
            or (
                len(significant_digits) == len(largest_digits)
                and significant_digits >= largest_digits
            )
        ):
            return cycle_end

        seconds = int(significant_digits)
        delay_microseconds = seconds * _MICROSECONDS_PER_SECOND
        return _bounded_timestamp(
            now,
            delay_microseconds,
            cycle_end,
            remaining_microseconds,
        )

    try:
        parsed_date = parsedate_to_datetime(value)
        retry_at = _normalize_utc(parsed_date, "Retry-After")
    except (OverflowError, TypeError, ValueError, IndexError):
        return None

    if retry_at <= now:
        retry_at = now
    elif retry_at >= cycle_end:
        return cycle_end

    return min(_ceil_millisecond(retry_at), cycle_end)


def plan_retry(
    *,
    attempt_count: int,
    cycle_started_at: datetime,
    now: datetime,
    retry_after: str | None = None,
    uniform: Callable[[float, float], float] = random.uniform,
) -> RetryPlan:
    """Plan a UTC retry instant or report exhausted attempt/time budget.

    Naive datetimes are interpreted as UTC, matching BSON dates returned by
    MongoDB with the default timezone-naive driver configuration.
    """
    if type(attempt_count) is not int or not 1 <= attempt_count <= 5:
        raise ValueError("attempt_count must be an integer from 1 through 5")
    if retry_after is not None and not isinstance(retry_after, str):
        raise TypeError("retry_after must be a string or None")
    if not callable(uniform):
        raise TypeError("uniform must be callable")

    normalized_start = _normalize_utc(cycle_started_at, "cycle_started_at")
    normalized_now = _normalize_utc(now, "now")
    if normalized_start > normalized_now:
        raise ValueError("cycle_started_at must not be later than now")

    # Compare original instants first, then align the persisted cycle anchor
    # with BSON's millisecond precision before deriving its fixed deadline.
    normalized_start = normalized_start.replace(
        microsecond=(
            normalized_start.microsecond // _MICROSECONDS_PER_MILLISECOND
        )
        * _MICROSECONDS_PER_MILLISECOND
    )
    try:
        cycle_end = normalized_start + _CYCLE_WINDOW
    except OverflowError as error:
        raise ValueError("cycle end is outside the supported datetime range") from error

    if attempt_count == 5:
        return RetryExhausted(RetryExhaustionReason.ATTEMPTS_EXHAUSTED)
    if normalized_now >= cycle_end:
        return RetryExhausted(RetryExhaustionReason.WINDOW_EXPIRED)

    remaining_microseconds = _timedelta_microseconds(cycle_end - normalized_now)
    local_delay_microseconds = _local_backoff_microseconds(
        attempt_count,
        uniform,
    )
    local_retry_at = _bounded_timestamp(
        normalized_now,
        local_delay_microseconds,
        cycle_end,
        remaining_microseconds,
    )
    retry_after_at = _retry_after_timestamp(
        retry_after,
        now=normalized_now,
        cycle_end=cycle_end,
        remaining_microseconds=remaining_microseconds,
    )

    if retry_after_at is None:
        return RetryScheduled(
            next_retry_at=local_retry_at,
            retry_after_used=False,
        )

    effective_retry_at = max(local_retry_at, retry_after_at)
    return RetryScheduled(
        next_retry_at=effective_retry_at,
        retry_after_used=effective_retry_at > local_retry_at,
    )

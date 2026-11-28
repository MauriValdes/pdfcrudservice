"""RED contract tests for the process-local MS2 circuit breaker."""

import asyncio
import math

import pytest

from src.worker.circuit_breaker import (
    Admitted,
    CircuitBreaker,
    CircuitBreakerState,
    Rejected,
    RejectionReason,
)
from src.worker.error_classifier import (
    CircuitBreakerSignal,
    classify_ms2_error,
)
from src.worker.ms2_client import (
    MS2HTTPError,
    MS2InvalidResponseError,
    MS2TransportError,
    TransportKind,
)


class FakeMonotonicClock:
    def __init__(self, value=0.0):
        self.value = value

    def __call__(self):
        return self.value


def _category(error):
    category = classify_ms2_error(error)
    assert category is not None
    return category


def _eligible_category():
    return _category(MS2HTTPError(503, "transient", None))


def _neutral_category():
    return _category(MS2HTTPError(400, "permanent_input", None))


def _admit(breaker):
    admission = breaker.admit()
    assert isinstance(admission, Admitted)
    return admission.permit


def _record_failure(breaker, category=None):
    permit = _admit(breaker)
    assert breaker.record_error(permit, category or _eligible_category())
    return permit


def _open_breaker(clock):
    breaker = CircuitBreaker(monotonic_clock=clock)
    for _ in range(5):
        _record_failure(breaker)
    assert breaker.state is CircuitBreakerState.OPEN
    return breaker


def test_initial_state_and_instances_are_independent():
    clock = FakeMonotonicClock()
    first = CircuitBreaker(monotonic_clock=clock)
    second = CircuitBreaker(monotonic_clock=clock)

    assert first.state is CircuitBreakerState.CLOSED
    assert second.state is CircuitBreakerState.CLOSED
    for _ in range(5):
        _record_failure(first)

    assert first.state is CircuitBreakerState.OPEN
    assert second.state is CircuitBreakerState.CLOSED


@pytest.mark.parametrize(
    ("failure_count", "expected_state"),
    [
        (4, CircuitBreakerState.CLOSED),
        (5, CircuitBreakerState.OPEN),
    ],
)
def test_opens_at_five_eligible_failures(failure_count, expected_state):
    breaker = CircuitBreaker(monotonic_clock=FakeMonotonicClock())

    for _ in range(failure_count):
        _record_failure(breaker)

    assert breaker.state is expected_state


@pytest.mark.parametrize(
    ("fifth_failure_at", "expected_state"),
    [
        (59.999, CircuitBreakerState.OPEN),
        (60.0, CircuitBreakerState.CLOSED),
    ],
)
def test_sliding_window_expires_events_at_exactly_sixty_seconds(
    fifth_failure_at, expected_state
):
    clock = FakeMonotonicClock()
    breaker = CircuitBreaker(monotonic_clock=clock)

    for event_time in (0.0, 1.0, 2.0, 3.0):
        clock.value = event_time
        _record_failure(breaker)

    clock.value = fifth_failure_at
    _record_failure(breaker)

    assert breaker.state is expected_state


def test_closed_successes_and_neutrals_do_not_clear_recent_failures():
    clock = FakeMonotonicClock()
    breaker = CircuitBreaker(monotonic_clock=clock)

    _record_failure(breaker)
    _record_failure(breaker)

    success_permit = _admit(breaker)
    assert breaker.record_success(success_permit)
    neutral_permit = _admit(breaker)
    assert breaker.record_error(neutral_permit, _neutral_category())

    _record_failure(breaker)
    _record_failure(breaker)
    assert breaker.state is CircuitBreakerState.CLOSED
    _record_failure(breaker)
    assert breaker.state is CircuitBreakerState.OPEN


@pytest.mark.parametrize(
    ("error", "eligible"),
    [
        pytest.param(
            MS2HTTPError(429, "transient", None), True, id="http-429"
        ),
        pytest.param(
            MS2HTTPError(503, "transient", None), True, id="http-503"
        ),
        pytest.param(
            MS2TransportError(
                "timeout",
                transport_kind=TransportKind.TIMEOUT,
            ),
            True,
            id="timeout",
        ),
        pytest.param(
            MS2TransportError(
                "connection",
                transport_kind=TransportKind.CONNECTION,
            ),
            True,
            id="connection",
        ),
        pytest.param(
            MS2InvalidResponseError("invalid response"),
            True,
            id="invalid-response",
        ),
        pytest.param(
            MS2HTTPError(400, "permanent_input", None),
            False,
            id="input-error",
        ),
        pytest.param(
            MS2HTTPError(401, "unclassified", None),
            False,
            id="technical-error",
        ),
    ],
)
def test_real_classifier_categories_control_closed_failure_count(
    error, eligible
):
    category = _category(error)
    assert category.closed_failure_eligible is eligible
    assert category.half_open_signal is (
        CircuitBreakerSignal.FAILURE
        if eligible
        else CircuitBreakerSignal.NEUTRAL
    )

    breaker = CircuitBreaker(monotonic_clock=FakeMonotonicClock())
    permit = _admit(breaker)
    assert breaker.record_error(permit, category)
    for _ in range(4):
        _record_failure(breaker)

    assert breaker.state is (
        CircuitBreakerState.OPEN
        if eligible
        else CircuitBreakerState.CLOSED
    )


@pytest.mark.parametrize(
    ("elapsed", "admitted"),
    [
        (29.999, False),
        (30.0, True),
    ],
)
def test_open_minimum_and_state_query_transition_at_thirty_seconds(
    elapsed, admitted
):
    clock = FakeMonotonicClock(10.0)
    breaker = _open_breaker(clock)
    clock.value = 10.0 + elapsed

    assert breaker.state is CircuitBreakerState.OPEN
    decision = breaker.admit()
    if admitted:
        assert isinstance(decision, Admitted)
        assert breaker.state is CircuitBreakerState.HALF_OPEN
    else:
        assert isinstance(decision, Rejected)
        assert decision.state is CircuitBreakerState.OPEN
        assert decision.reason is RejectionReason.OPEN
        assert decision.retry_after_seconds == pytest.approx(0.001)
        assert breaker.state is CircuitBreakerState.OPEN


def test_open_rejection_reports_monotonic_time_remaining():
    clock = FakeMonotonicClock()
    breaker = _open_breaker(clock)
    clock.value = 12.5

    decision = breaker.admit()

    assert isinstance(decision, Rejected)
    assert decision.reason is RejectionReason.OPEN
    assert decision.retry_after_seconds == pytest.approx(17.5)
    assert breaker.state is CircuitBreakerState.OPEN


def test_half_open_grants_one_probe_and_rejects_others_without_a_deadline():
    clock = FakeMonotonicClock()
    breaker = _open_breaker(clock)
    clock.value = 30.0

    probe = breaker.admit()
    rejected = breaker.admit()

    assert isinstance(probe, Admitted)
    assert isinstance(rejected, Rejected)
    assert rejected.state is CircuitBreakerState.HALF_OPEN
    assert rejected.reason is RejectionReason.HALF_OPEN_BUSY
    assert rejected.retry_after_seconds is None


def test_half_open_success_closes_and_resets_the_failure_history():
    clock = FakeMonotonicClock()
    breaker = _open_breaker(clock)
    clock.value = 30.0
    probe = _admit(breaker)

    assert breaker.record_success(probe)
    assert breaker.state is CircuitBreakerState.CLOSED
    for _ in range(4):
        _record_failure(breaker)
    assert breaker.state is CircuitBreakerState.CLOSED
    _record_failure(breaker)
    assert breaker.state is CircuitBreakerState.OPEN


def test_half_open_failure_reopens_and_restarts_the_thirty_second_period():
    clock = FakeMonotonicClock()
    breaker = _open_breaker(clock)
    clock.value = 30.0
    probe = _admit(breaker)
    clock.value = 37.0

    assert breaker.record_error(probe, _eligible_category())
    assert breaker.state is CircuitBreakerState.OPEN

    clock.value = 66.999
    just_before_reopen = breaker.admit()
    assert isinstance(just_before_reopen, Rejected)
    assert just_before_reopen.reason is RejectionReason.OPEN
    assert just_before_reopen.retry_after_seconds == pytest.approx(0.001)

    clock.value = 67.0
    at_reopen = breaker.admit()
    assert isinstance(at_reopen, Admitted)
    assert breaker.state is CircuitBreakerState.HALF_OPEN


def test_half_open_neutral_and_release_free_the_probe_without_closing():
    clock = FakeMonotonicClock()
    breaker = _open_breaker(clock)
    clock.value = 30.0
    first_probe = _admit(breaker)

    assert breaker.record_error(first_probe, _neutral_category())
    assert breaker.state is CircuitBreakerState.HALF_OPEN
    assert breaker.record_success(first_probe) is False

    second_probe = _admit(breaker)
    assert breaker.release(second_probe)
    assert breaker.state is CircuitBreakerState.HALF_OPEN
    assert breaker.release(second_probe) is False
    assert isinstance(breaker.admit(), Admitted)


@pytest.mark.parametrize("resolution", ["success", "error", "release"])
def test_duplicate_resolutions_are_rejected(resolution):
    breaker = CircuitBreaker(monotonic_clock=FakeMonotonicClock())
    permit = _admit(breaker)

    if resolution == "success":
        resolve = lambda: breaker.record_success(permit)
    elif resolution == "error":
        resolve = lambda: breaker.record_error(permit, _eligible_category())
    else:
        resolve = lambda: breaker.release(permit)

    assert resolve() is True
    assert resolve() is False


def test_foreign_permit_is_rejected_without_consuming_its_owner_permit():
    clock = FakeMonotonicClock()
    owner = CircuitBreaker(monotonic_clock=clock)
    foreign = CircuitBreaker(monotonic_clock=clock)
    permit = _admit(owner)

    assert foreign.record_success(permit) is False
    assert owner.record_success(permit) is True
    assert owner.state is CircuitBreakerState.CLOSED


def test_late_results_from_prior_generations_cannot_change_later_states():
    clock = FakeMonotonicClock()
    breaker = CircuitBreaker(monotonic_clock=clock)
    old_permits = [breaker.admit().permit for _ in range(5)]
    category = _eligible_category()
    for permit in old_permits[:4]:
        assert breaker.record_error(permit, category)
    assert breaker.record_error(old_permits[4], category)
    assert breaker.state is CircuitBreakerState.OPEN

    assert breaker.record_success(old_permits[0]) is False
    assert breaker.state is CircuitBreakerState.OPEN

    clock.value = 30.0
    probe = _admit(breaker)
    assert breaker.record_error(old_permits[1], category) is False
    assert breaker.state is CircuitBreakerState.HALF_OPEN
    busy = breaker.admit()
    assert isinstance(busy, Rejected)
    assert busy.reason is RejectionReason.HALF_OPEN_BUSY

    assert breaker.record_success(probe)
    assert breaker.state is CircuitBreakerState.CLOSED
    assert breaker.record_error(old_permits[2], category) is False
    assert breaker.state is CircuitBreakerState.CLOSED
    assert breaker.release(old_permits[3]) is False


@pytest.mark.asyncio
async def test_concurrent_half_open_admissions_grant_exactly_one_probe():
    clock = FakeMonotonicClock()
    breaker = _open_breaker(clock)
    clock.value = 30.0

    async def request_admission():
        return breaker.admit()

    decisions = await asyncio.gather(
        *(request_admission() for _ in range(12))
    )

    assert sum(isinstance(item, Admitted) for item in decisions) == 1
    rejected = [item for item in decisions if isinstance(item, Rejected)]
    assert len(rejected) == 11
    assert all(
        item.reason is RejectionReason.HALF_OPEN_BUSY for item in rejected
    )


@pytest.mark.parametrize(
    ("bad_value", "error_type"),
    [
        (True, TypeError),
        (math.nan, ValueError),
        (math.inf, ValueError),
    ],
)
def test_invalid_clock_read_does_not_consume_permit_or_mutate_state(
    bad_value, error_type
):
    clock = FakeMonotonicClock()
    breaker = CircuitBreaker(monotonic_clock=clock)
    permit = _admit(breaker)
    clock.value = bad_value

    with pytest.raises(error_type):
        breaker.record_error(permit, _eligible_category())
    assert breaker.state is CircuitBreakerState.CLOSED

    clock.value = 1.0
    assert breaker.record_error(permit, _eligible_category())
    assert breaker.state is CircuitBreakerState.CLOSED


def test_regressive_clock_read_does_not_consume_permit_or_change_state():
    clock = FakeMonotonicClock(5.0)
    breaker = CircuitBreaker(monotonic_clock=clock)
    permit = _admit(breaker)
    clock.value = 4.0

    with pytest.raises(ValueError):
        breaker.record_error(permit, _eligible_category())
    assert breaker.state is CircuitBreakerState.CLOSED

    clock.value = 6.0
    assert breaker.record_error(permit, _eligible_category())


def test_release_does_not_read_the_clock():
    clock = FakeMonotonicClock()
    breaker = CircuitBreaker(monotonic_clock=clock)
    permit = _admit(breaker)
    clock.value = math.nan

    assert breaker.release(permit)
    assert breaker.state is CircuitBreakerState.CLOSED


def test_invalid_permit_and_category_types_raise_without_consuming_valid_permit():
    breaker = CircuitBreaker(monotonic_clock=FakeMonotonicClock())
    permit = _admit(breaker)

    with pytest.raises(TypeError):
        breaker.record_success(object())
    with pytest.raises(TypeError):
        breaker.release(object())
    with pytest.raises(TypeError):
        breaker.record_error(permit, object())

    assert breaker.record_error(permit, _eligible_category())


def test_non_callable_clock_and_invalid_initial_clock_values_are_rejected():
    with pytest.raises(TypeError):
        CircuitBreaker(monotonic_clock=None)
    with pytest.raises(ValueError):
        CircuitBreaker(monotonic_clock=lambda: math.nan)
    with pytest.raises(ValueError):
        CircuitBreaker(monotonic_clock=lambda: math.inf)

"""Process-local circuit breaker for calls from the worker to MS2.

Instances are intended to be used synchronously from a single asyncio event
loop. Methods contain no ``await`` points, so each state transition is atomic
with respect to other coroutines on that loop. Instances are not thread-safe.
"""

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
import math
from numbers import Real
import time
from typing import Callable, TypeAlias

from src.worker.error_classifier import CircuitBreakerSignal, ErrorCategory


_FAILURE_THRESHOLD = 5
_FAILURE_WINDOW_SECONDS = 60.0
_OPEN_DURATION_SECONDS = 30.0


class CircuitBreakerState(str, Enum):
    """Current state of a local circuit breaker."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class RejectionReason(str, Enum):
    """Why an admission request was rejected."""

    OPEN = "open"
    HALF_OPEN_BUSY = "half_open_busy"


@dataclass(frozen=True, slots=True)
class AdmissionPermit:
    """Opaque, immutable token granting one call admission.

    Instances are issued by :class:`CircuitBreaker`; resolution additionally
    checks the breaker’s active-permit registry, not just these token fields.
    """

    _owner: object = field(repr=False)
    _generation: int = field(repr=False)
    _nonce: object = field(repr=False)


@dataclass(frozen=True, slots=True)
class Admitted:
    """An admission decision that includes its one-use permit."""

    permit: AdmissionPermit


@dataclass(frozen=True, slots=True)
class Rejected:
    """A rejection decision and any known monotonic wait duration."""

    state: CircuitBreakerState
    reason: RejectionReason
    retry_after_seconds: float | None


Admission: TypeAlias = Admitted | Rejected


class _PermitKind(str, Enum):
    CLOSED_CALL = "closed_call"
    HALF_OPEN_PROBE = "half_open_probe"


class CircuitBreaker:
    """Synchronous, process-local admission and outcome state machine."""

    def __init__(
        self,
        *,
        monotonic_clock: Callable[[], Real] = time.monotonic,
    ) -> None:
        if not callable(monotonic_clock):
            raise TypeError("monotonic_clock must be callable")

        self._clock = monotonic_clock
        self._state = CircuitBreakerState.CLOSED
        self._generation = 0
        self._owner = object()
        self._active: dict[object, tuple[AdmissionPermit, _PermitKind]] = {}
        self._failure_times: deque[float] = deque()
        self._opened_at: float | None = None
        self._last_clock_value: float | None = None

        # Validate the injected source at construction, but retain no clock
        # value if it is invalid.
        self._read_clock()

    @property
    def state(self) -> CircuitBreakerState:
        """Return the stored state without advancing time or mutating it."""
        return self._state

    def admit(self) -> Admission:
        """Request permission to make one MS2 call."""
        if self._state is CircuitBreakerState.OPEN:
            now = self._read_clock()
            assert self._opened_at is not None
            elapsed = now - self._opened_at
            if elapsed < _OPEN_DURATION_SECONDS:
                return Rejected(
                    state=CircuitBreakerState.OPEN,
                    reason=RejectionReason.OPEN,
                    retry_after_seconds=_OPEN_DURATION_SECONDS - elapsed,
                )

            self._advance_generation()
            self._state = CircuitBreakerState.HALF_OPEN
            self._opened_at = None

        if self._state is CircuitBreakerState.HALF_OPEN:
            if any(
                kind is _PermitKind.HALF_OPEN_PROBE
                for _, kind in self._active.values()
            ):
                return Rejected(
                    state=CircuitBreakerState.HALF_OPEN,
                    reason=RejectionReason.HALF_OPEN_BUSY,
                    retry_after_seconds=None,
                )
            return Admitted(self._issue_permit(_PermitKind.HALF_OPEN_PROBE))

        return Admitted(self._issue_permit(_PermitKind.CLOSED_CALL))

    def record_success(self, permit: AdmissionPermit) -> bool:
        """Resolve an admitted call as a valid success."""
        self._validate_permit_type(permit)
        record = self._active_record(permit)
        if record is None:
            return False

        _, kind = record
        self._consume_permit(permit)
        if kind is _PermitKind.HALF_OPEN_PROBE:
            self._advance_generation()
            self._state = CircuitBreakerState.CLOSED
            self._opened_at = None
            self._failure_times.clear()
        return True

    def record_error(
        self,
        permit: AdmissionPermit,
        category: ErrorCategory,
    ) -> bool:
        """Resolve an admitted call using the classifier’s existing signal."""
        self._validate_permit_type(permit)
        if not isinstance(category, ErrorCategory):
            raise TypeError("category must be an ErrorCategory")

        record = self._active_record(permit)
        if record is None:
            return False

        _, kind = record
        if kind is _PermitKind.CLOSED_CALL:
            if not category.closed_failure_eligible:
                self._consume_permit(permit)
                return True

            now = self._read_clock()
            self._expire_old_failures(now)
            self._consume_permit(permit)
            self._failure_times.append(now)
            if len(self._failure_times) >= _FAILURE_THRESHOLD:
                self._open(now)
            return True

        if category.half_open_signal is CircuitBreakerSignal.NEUTRAL:
            self._consume_permit(permit)
            return True
        if category.half_open_signal is not CircuitBreakerSignal.FAILURE:
            raise ValueError("category has an invalid HALF-OPEN signal")

        now = self._read_clock()
        self._consume_permit(permit)
        self._open(now)
        return True

    def release(self, permit: AdmissionPermit) -> bool:
        """Release a permit neutrally, without consulting the clock."""
        self._validate_permit_type(permit)
        if self._active_record(permit) is None:
            return False
        self._consume_permit(permit)
        return True

    def _read_clock(self) -> float:
        value = self._clock()
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError("monotonic clock must return a real number")
        try:
            numeric_value = float(value)
        except (OverflowError, ValueError) as exc:
            raise ValueError("monotonic clock must return a finite number") from exc
        if not math.isfinite(numeric_value):
            raise ValueError("monotonic clock must return a finite number")
        if (
            self._last_clock_value is not None
            and numeric_value < self._last_clock_value
        ):
            raise ValueError("monotonic clock moved backwards")

        # Commit the observation only after all clock validation succeeds.
        self._last_clock_value = numeric_value
        return numeric_value

    @staticmethod
    def _validate_permit_type(permit: AdmissionPermit) -> None:
        if not isinstance(permit, AdmissionPermit):
            raise TypeError("permit must be an AdmissionPermit")

    def _active_record(
        self, permit: AdmissionPermit
    ) -> tuple[AdmissionPermit, _PermitKind] | None:
        if permit._owner is not self._owner:
            return None
        record = self._active.get(permit._nonce)
        if record is None or record[0] is not permit:
            return None
        if permit._generation != self._generation:
            return None
        return record

    def _issue_permit(self, kind: _PermitKind) -> AdmissionPermit:
        nonce = object()
        permit = AdmissionPermit(
            _owner=self._owner,
            _generation=self._generation,
            _nonce=nonce,
        )
        self._active[nonce] = (permit, kind)
        return permit

    def _consume_permit(self, permit: AdmissionPermit) -> None:
        self._active.pop(permit._nonce, None)

    def _expire_old_failures(self, now: float) -> None:
        # The exact window is (now - 60, now]; events on its left boundary
        # expire.
        cutoff = now - _FAILURE_WINDOW_SECONDS
        while self._failure_times and self._failure_times[0] <= cutoff:
            self._failure_times.popleft()

    def _advance_generation(self) -> None:
        self._generation += 1
        # In-flight HTTP calls continue, but their now-unregistered permits
        # cannot resolve into this or any later breaker generation.
        self._active.clear()

    def _open(self, now: float) -> None:
        self._advance_generation()
        self._state = CircuitBreakerState.OPEN
        self._opened_at = now
        self._failure_times.clear()

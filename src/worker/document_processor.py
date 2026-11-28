"""Basic orchestration for processing one already-claimed document."""

import asyncio
import logging
import math
from datetime import datetime, timedelta, timezone
from enum import Enum
from numbers import Real

from bson import ObjectId
from pymongo.errors import (
    AutoReconnect,
    ConnectionFailure,
    NetworkTimeout,
    ServerSelectionTimeoutError,
    WTimeoutError,
    WriteConcernError,
)

from src.core.exceptions import DatabaseUnavailableError
from src.models.document import DocumentModel
from src.repository.gridfs_repository import GridFSRepository
from src.repository.processing_repository import (
    ProcessingRepository,
    RetryTimestampExpiredError,
)
from src.worker.circuit_breaker import (
    Admitted,
    CircuitBreaker,
    Rejected,
    RejectionReason,
)
from src.worker.error_classifier import DocumentAction, classify_ms2_error
from src.worker.heartbeat import run_with_heartbeat
from src.worker.ms2_client import (
    MS2Client,
    MS2HTTPError,
    MS2InvalidResponseError,
    MS2TransportError,
)
from src.worker.pdf_reader import read_claimed_pdf
from src.worker.retry_policy import RetryExhausted, RetryScheduled, plan_retry


_logger = logging.getLogger(__name__)
_MS2_ERRORS = (MS2HTTPError, MS2TransportError, MS2InvalidResponseError)
_AMBIGUOUS_DRIVER_ERRORS = (
    AutoReconnect,
    ConnectionFailure,
    NetworkTimeout,
    ServerSelectionTimeoutError,
    WriteConcernError,
    WTimeoutError,
)


class _TransitionResolution(Enum):
    ACKNOWLEDGED = "ACKNOWLEDGED"
    GUARD_REJECTED = "GUARD_REJECTED"
    OBSERVED_UNCONFIRMED = "OBSERVED_UNCONFIRMED"
    UNKNOWN = "UNKNOWN"


def _is_ambiguous_transition_error(error: Exception) -> bool:
    """Classify only persistence failures that can hide an applied write."""
    if isinstance(error, _AMBIGUOUS_DRIVER_ERRORS):
        return True
    if not isinstance(error, DatabaseUnavailableError):
        return False

    cause = error.__cause__
    seen = set()
    while cause is not None and id(cause) not in seen:
        if isinstance(cause, _AMBIGUOUS_DRIVER_ERRORS):
            return True
        seen.add(id(cause))
        cause = cause.__cause__
    return False


def _bson_utc_milliseconds(value: datetime) -> datetime:
    """Normalize a datetime as it is represented by a BSON date."""
    if not isinstance(value, datetime):
        raise ValueError("expected datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        value = value.replace(tzinfo=timezone.utc)
    normalized = value.astimezone(timezone.utc)
    return normalized.replace(
        microsecond=(normalized.microsecond // 1000) * 1000
    )


def _utc_now() -> datetime:
    """Return the current timezone-aware UTC instant."""
    return datetime.now(timezone.utc)


class ProcessingResult(Enum):
    """Non-persistent outcome of one processing invocation."""

    COMPLETED = "completed"
    ATTEMPT_NOT_AUTHORIZED = "attempt_not_authorized"
    RESULT_NOT_CONFIRMED = "result_not_confirmed"
    FAILED_PERMANENT = "failed_permanent"
    FAILED_EXHAUSTED = "failed_exhausted"
    RETRY_SCHEDULED = "retry_scheduled"
    FAILED_TECHNICAL = "failed_technical"
    CIRCUIT_BREAKER_RELEASED = "circuit_breaker_released"


async def process_claimed_document(
    document: DocumentModel,
    *,
    processing_repository: ProcessingRepository,
    gridfs_repository: GridFSRepository,
    ms2_client: MS2Client,
    circuit_breaker: CircuitBreaker,
    circuit_breaker_busy_retry_seconds: float,
) -> ProcessingResult:
    """Process one claimed document with at most one authorized MS2 extraction.

    The caller is responsible for preventing concurrent invocations for the
    same claim. When deciding exhaustion, no other processing for the claim or
    acceptable authorized result may be pending for this acquisition.
    Cancellation cannot establish whether remote effects occurred.
    """
    if isinstance(circuit_breaker_busy_retry_seconds, bool) or not isinstance(
        circuit_breaker_busy_retry_seconds, Real
    ):
        raise TypeError("circuit_breaker_busy_retry_seconds must be a real number")
    try:
        busy_interval = float(circuit_breaker_busy_retry_seconds)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(
            "circuit_breaker_busy_retry_seconds must be finite and positive"
        ) from error
    if not math.isfinite(busy_interval) or busy_interval <= 0:
        raise ValueError(
            "circuit_breaker_busy_retry_seconds must be finite and positive"
        )

    snapshot = document.model_copy(deep=True)

    document_id = snapshot.id
    worker_id = snapshot.worker_id
    fencing_token = snapshot.fencing_token
    processing_cycle_id = snapshot.processing_cycle_id
    file_id = snapshot.file_id
    filename = snapshot.filename

    if not isinstance(document_id, str):
        raise TypeError("document.id must be a string")
    if not ObjectId.is_valid(document_id):
        raise ValueError("document.id must be a valid ObjectId")
    if not isinstance(worker_id, str):
        raise TypeError("document.worker_id must be a string")
    if not worker_id:
        raise ValueError("document.worker_id must not be empty")
    if not isinstance(fencing_token, str):
        raise TypeError("document.fencing_token must be a string")
    if not fencing_token:
        raise ValueError("document.fencing_token must not be empty")
    if not isinstance(processing_cycle_id, str):
        raise TypeError("document.processing_cycle_id must be a string")
    if not processing_cycle_id:
        raise ValueError("document.processing_cycle_id must not be empty")
    if not isinstance(file_id, ObjectId):
        raise ValueError("document.file_id must be an ObjectId")
    if not isinstance(filename, str):
        raise TypeError("document.filename must be a string")
    if not filename:
        raise ValueError("document.filename must not be empty")

    def matches_identity(observed: dict[str, object], status: str) -> bool:
        return (
            observed.get("_id") == ObjectId(document_id)
            and observed.get("status") == status
            and observed.get("processing_cycle_id") == processing_cycle_id
        )

    def matches_attempt_count(
        observed: dict[str, object],
        expected_attempt_count: int,
    ) -> bool:
        actual = observed.get("attempt_count", 0)
        return (
            isinstance(actual, int)
            and not isinstance(actual, bool)
            and actual == expected_attempt_count
        )

    def matches_terminal(
        observed: dict[str, object] | None,
        *,
        status: str,
        attempt_count: int,
        failure_code: str | None = None,
        content: str | None = None,
        page_count: int | None = None,
    ) -> bool:
        if not isinstance(observed, dict):
            return False
        if (
            not matches_identity(observed, status)
            or observed.get("worker_id") != worker_id
            or observed.get("fencing_token") != fencing_token
            or not matches_attempt_count(observed, attempt_count)
            or "next_retry_at" in observed
        ):
            return False
        if failure_code is not None:
            return (
                observed.get("last_error") == failure_code
                and "content" not in observed
                and "page_count" not in observed
            )
        return (
            observed.get("content") == content
            and observed.get("page_count") == page_count
            and "last_error" not in observed
        )

    def matches_exhaustion(
        observed: dict[str, object] | None,
        expected_attempt_count: int,
    ) -> bool:
        failure_code = (
            "attempts_exhausted"
            if expected_attempt_count == 5
            else "cycle_window_exhausted"
        )
        return matches_terminal(
            observed,
            status="failed",
            attempt_count=expected_attempt_count,
            failure_code=failure_code,
        )

    def matches_pending_release(
        observed: dict[str, object] | None,
        *,
        expected_attempt_count: int,
        requested_retry_at: datetime,
        failure_code: str | None,
    ) -> bool:
        if not isinstance(observed, dict):
            return False
        cycle_started_at = snapshot.processing_cycle_started_at
        if not isinstance(cycle_started_at, datetime):
            return False
        try:
            cycle_deadline = _bson_utc_milliseconds(
                cycle_started_at + timedelta(minutes=10)
            )
            effective_retry_at = min(
                _bson_utc_milliseconds(requested_retry_at),
                cycle_deadline,
            )
            observed_retry_at = _bson_utc_milliseconds(
                observed["next_retry_at"]
            )
        except (KeyError, OverflowError, TypeError, ValueError):
            return False

        return (
            matches_identity(observed, "pending")
            and matches_attempt_count(observed, expected_attempt_count)
            and "worker_id" not in observed
            and "fencing_token" not in observed
            and "content" not in observed
            and "page_count" not in observed
            and observed_retry_at == effective_retry_at
            and (
                failure_code is None
                or observed.get("last_error") == failure_code
            )
        )

    async def resolve_transition(
        transition_operation: str,
        write,
        expected_state,
    ) -> _TransitionResolution:
        try:
            acknowledged = await write()
        except Exception as primary_error:
            if not _is_ambiguous_transition_error(primary_error):
                raise

            resolution = _TransitionResolution.UNKNOWN
            try:
                observed = await processing_repository.observe_transition_state(
                    document_id,
                    timeout_seconds=0.25,
                )
            except asyncio.CancelledError:
                raise
            except Exception as observation_error:
                if not (
                    isinstance(
                        observation_error,
                        (TimeoutError, DatabaseUnavailableError),
                    )
                    or isinstance(observation_error, _AMBIGUOUS_DRIVER_ERRORS)
                ):
                    raise
                observed = None

            if observed is not None and expected_state(observed):
                resolution = _TransitionResolution.OBSERVED_UNCONFIRMED

            _logger.warning(
                "Ambiguous processing transition resolved",
                extra={
                    "transition_resolution": resolution.value,
                    "transition_operation": transition_operation,
                    "document_id": document_id,
                    "exception_type": type(primary_error).__name__,
                },
            )
            return resolution

        return (
            _TransitionResolution.ACKNOWLEDGED
            if acknowledged
            else _TransitionResolution.GUARD_REJECTED
        )

    async def operation() -> ProcessingResult:
        permit = None
        permit_pending = False
        primary_error = False
        try:
            admission = circuit_breaker.admit()
            if isinstance(admission, Rejected):
                if admission.reason is RejectionReason.OPEN:
                    delay = max(
                        admission.retry_after_seconds or 0.0,
                        busy_interval,
                    )
                elif admission.reason is RejectionReason.HALF_OPEN_BUSY:
                    delay = busy_interval
                else:
                    raise TypeError("Circuit Breaker returned an unsupported rejection")

                now = _utc_now()
                if now.tzinfo is None or now.utcoffset() is None:
                    raise ValueError("_utc_now must return a timezone-aware datetime")
                retry_at = now.astimezone(timezone.utc) + timedelta(seconds=delay)
                # MongoDB stores BSON dates at millisecond precision. Ceil rather
                # than truncate so the persisted timestamp remains in the future.
                remainder = retry_at.microsecond % 1000
                if remainder:
                    retry_at += timedelta(microseconds=1000 - remainder)
                retry_at = retry_at.replace(tzinfo=timezone.utc)
                try:
                    resolution = await resolve_transition(
                        "release_for_circuit_open",
                        lambda: processing_repository.release_for_circuit_open(
                            document_id,
                            worker_id,
                            fencing_token,
                            processing_cycle_id,
                            snapshot.attempt_count,
                            retry_at,
                        ),
                        lambda observed: matches_pending_release(
                            observed,
                            expected_attempt_count=snapshot.attempt_count,
                            requested_retry_at=retry_at,
                            failure_code=None,
                        ),
                    )
                except RetryTimestampExpiredError:
                    resolution = _TransitionResolution.GUARD_REJECTED

                if resolution is _TransitionResolution.ACKNOWLEDGED:
                    return ProcessingResult.CIRCUIT_BREAKER_RELEASED
                if resolution in (
                    _TransitionResolution.OBSERVED_UNCONFIRMED,
                    _TransitionResolution.UNKNOWN,
                ):
                    return ProcessingResult.RESULT_NOT_CONFIRMED

                exhaustion_resolution = await resolve_transition(
                    "fail_exhausted_processing",
                    lambda: processing_repository.fail_exhausted_processing(
                        document_id,
                        worker_id,
                        fencing_token,
                        processing_cycle_id,
                    ),
                    lambda observed: matches_exhaustion(
                        observed,
                        snapshot.attempt_count,
                    ),
                )
                return (
                    ProcessingResult.FAILED_EXHAUSTED
                    if exhaustion_resolution
                    is _TransitionResolution.ACKNOWLEDGED
                    else ProcessingResult.RESULT_NOT_CONFIRMED
                )

            if not isinstance(admission, Admitted):
                raise TypeError("Circuit Breaker returned an unsupported admission")
            permit = admission.permit
            permit_pending = True

            pdf_bytes = await read_claimed_pdf(
                snapshot,
                processing_repository=processing_repository,
                gridfs_repository=gridfs_repository,
            )

            attempt_number = await processing_repository.record_ms2_attempt(
                document_id,
                worker_id,
                fencing_token,
                processing_cycle_id,
            )
            if attempt_number is None:
                circuit_breaker.release(permit)
                permit_pending = False
                exhaustion_resolution = await resolve_transition(
                    "fail_exhausted_processing",
                    lambda: processing_repository.fail_exhausted_processing(
                        document_id,
                        worker_id,
                        fencing_token,
                        processing_cycle_id,
                    ),
                    lambda observed: matches_exhaustion(
                        observed,
                        snapshot.attempt_count,
                    ),
                )
                if (
                    exhaustion_resolution
                    is _TransitionResolution.ACKNOWLEDGED
                ):
                    return ProcessingResult.FAILED_EXHAUSTED
                if (
                    exhaustion_resolution
                    is _TransitionResolution.GUARD_REJECTED
                ):
                    return ProcessingResult.ATTEMPT_NOT_AUTHORIZED
                return ProcessingResult.RESULT_NOT_CONFIRMED

            try:
                result = await ms2_client.extract(pdf_bytes, filename)
            except _MS2_ERRORS as error:
                category = classify_ms2_error(error)
                if category is None:
                    raise
                circuit_breaker.record_error(permit, category)
                permit_pending = False
                return await persist_classified_error(
                    error,
                    category,
                    attempt_number,
                )
            else:
                circuit_breaker.record_success(permit)
                permit_pending = False

            resolution = await resolve_transition(
                "complete_processing",
                lambda: processing_repository.complete_processing(
                    document_id,
                    worker_id,
                    fencing_token,
                    processing_cycle_id,
                    attempt_number,
                    result.content,
                    result.page_count,
                ),
                lambda observed: matches_terminal(
                    observed,
                    status="completed",
                    attempt_count=attempt_number,
                    content=result.content,
                    page_count=result.page_count,
                ),
            )
            if resolution is _TransitionResolution.ACKNOWLEDGED:
                return ProcessingResult.COMPLETED
            return ProcessingResult.RESULT_NOT_CONFIRMED
        except BaseException:
            primary_error = True
            raise
        finally:
            if permit_pending:
                try:
                    circuit_breaker.release(permit)
                except BaseException:
                    if not primary_error:
                        raise

    async def persist_classified_error(
        error: Exception,
        category,
        attempt_number: int,
    ) -> ProcessingResult:
        if category.alert_required:
            fields = (
                f"event_id=worker.ms2.technical_failure "
                f"document_id={document_id} error_code={category.code}"
            )
            if category.http_status is not None:
                fields += f" http_status={category.http_status}"
            _logger.error(fields)

        if category.document_action in (
            DocumentAction.FAIL_INPUT,
            DocumentAction.FAIL_TECHNICAL,
        ):
            resolution = await resolve_transition(
                "fail_processing",
                lambda: processing_repository.fail_processing(
                    document_id,
                    worker_id,
                    fencing_token,
                    processing_cycle_id,
                    attempt_number,
                    category.code,
                ),
                lambda observed: matches_terminal(
                    observed,
                    status="failed",
                    attempt_count=attempt_number,
                    failure_code=category.code,
                ),
            )
            if resolution is not _TransitionResolution.ACKNOWLEDGED:
                return ProcessingResult.RESULT_NOT_CONFIRMED
            if category.document_action is DocumentAction.FAIL_INPUT:
                return ProcessingResult.FAILED_PERMANENT
            return ProcessingResult.FAILED_TECHNICAL

        if category.document_action is DocumentAction.RETRY:
            now = _utc_now()
            retry_plan = plan_retry(
                attempt_count=attempt_number,
                cycle_started_at=snapshot.processing_cycle_started_at,
                now=now,
                retry_after=(
                    error.retry_after if isinstance(error, MS2HTTPError) else None
                ),
            )
            if isinstance(retry_plan, RetryExhausted):
                resolution = await resolve_transition(
                    "fail_exhausted_processing",
                    lambda: processing_repository.fail_exhausted_processing(
                        document_id,
                        worker_id,
                        fencing_token,
                        processing_cycle_id,
                    ),
                    lambda observed: matches_exhaustion(
                        observed,
                        attempt_number,
                    ),
                )
                return (
                    ProcessingResult.FAILED_EXHAUSTED
                    if resolution is _TransitionResolution.ACKNOWLEDGED
                    else ProcessingResult.RESULT_NOT_CONFIRMED
                )

            if isinstance(retry_plan, RetryScheduled):
                try:
                    resolution = await resolve_transition(
                        "schedule_transient_retry",
                        lambda: processing_repository.schedule_transient_retry(
                            document_id,
                            worker_id,
                            fencing_token,
                            processing_cycle_id,
                            attempt_number,
                            category.code,
                            retry_plan.next_retry_at,
                        ),
                        lambda observed: matches_pending_release(
                            observed,
                            expected_attempt_count=attempt_number,
                            requested_retry_at=retry_plan.next_retry_at,
                            failure_code=category.code,
                        ),
                    )
                except RetryTimestampExpiredError:
                    resolution = _TransitionResolution.GUARD_REJECTED

                if resolution is _TransitionResolution.ACKNOWLEDGED:
                    return ProcessingResult.RETRY_SCHEDULED
                if resolution in (
                    _TransitionResolution.OBSERVED_UNCONFIRMED,
                    _TransitionResolution.UNKNOWN,
                ):
                    return ProcessingResult.RESULT_NOT_CONFIRMED

                exhaustion_resolution = await resolve_transition(
                    "fail_exhausted_processing",
                    lambda: processing_repository.fail_exhausted_processing(
                        document_id,
                        worker_id,
                        fencing_token,
                        processing_cycle_id,
                    ),
                    lambda observed: matches_exhaustion(
                        observed,
                        attempt_number,
                    ),
                )
                return (
                    ProcessingResult.FAILED_EXHAUSTED
                    if exhaustion_resolution
                    is _TransitionResolution.ACKNOWLEDGED
                    else ProcessingResult.RESULT_NOT_CONFIRMED
                )

            raise TypeError("plan_retry returned an unsupported result")

        raise TypeError("MS2 classifier returned an unsupported action")

    return await run_with_heartbeat(
        snapshot,
        processing_repository=processing_repository,
        operation=operation,
    )

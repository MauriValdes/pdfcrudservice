"""Pruebas unitarias RED del reclamo y renovación de leases de procesamiento."""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest
from bson import ObjectId
from pymongo import ASCENDING, ReturnDocument
from pymongo.errors import (
    AutoReconnect,
    ConnectionFailure,
    NetworkTimeout,
    ServerSelectionTimeoutError,
)

from src.core.exceptions import DatabaseUnavailableError
from src.models.document import DocumentModel, DocumentStatus
from src.repository import processing_repository
from src.repository.processing_repository import ProcessingRepository


class FrozenDateTime:
    @classmethod
    def now(cls, tz=None):
        assert tz is timezone.utc
        return NOW


NOW = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)


@pytest.fixture
def repo_with_mocks():
    collection = MagicMock()
    collection.find_one_and_update = AsyncMock()
    collection.find_one = AsyncMock()
    collection.find = MagicMock()
    collection.update_one = AsyncMock()
    collection.update_many = AsyncMock()
    collection.replace_one = AsyncMock()
    collection.aggregate = MagicMock()
    db = MagicMock()
    db.__getitem__.return_value = collection
    repository = ProcessingRepository(db)
    return repository


@pytest.fixture
def fixed_clock(monkeypatch):
    monkeypatch.setattr(
        processing_repository,
        "datetime",
        FrozenDateTime,
        raising=False,
    )
    return NOW


def _call_parts(repository):
    call = repository.collection.find_one_and_update.call_args
    assert call is not None
    query = call.args[0]
    update = call.args[1]
    sort = call.kwargs["sort"]
    return query, update, sort, call.kwargs


def _set_stage(update):
    assert isinstance(update, list)
    assert len(update) == 1
    return update[0]["$set"]


def _literal_value(expression):
    if isinstance(expression, dict) and "$literal" in expression:
        return expression["$literal"]
    return expression


def _claimed_document(
    *,
    file_id=None,
    processing_cycle_id=None,
    attempt_count=4,
    fencing_token="previous-token",
):
    document = {
        "_id": ObjectId(),
        "filename": "claimed.pdf",
        "checksum": "abc123",
        "size_bytes": 128,
        "status": DocumentStatus.PROCESSING.value,
        "created_at": NOW,
        "updated_at": NOW,
        "file_id": file_id or ObjectId(),
        "worker_id": "worker-a",
        "lease_until": NOW + timedelta(seconds=60),
        "heartbeat_at": NOW,
        "fencing_token": fencing_token,
        "attempt_count": attempt_count,
    }
    if processing_cycle_id is not ...:
        document["processing_cycle_id"] = processing_cycle_id
    return document


def _renewal_call_parts(repository):
    call = repository.collection.update_one.call_args
    assert call is not None
    query = call.args[0]
    update = call.args[1]
    return query, update, call.kwargs


def _pipeline_set(update):
    assert isinstance(update, list)
    assert len(update) == 1
    return update[0]["$set"]


@pytest.mark.asyncio
async def test_claim_filters_pending_objectid_and_all_retry_time_cases(
    repo_with_mocks, fixed_clock
):
    repo = repo_with_mocks
    repo.collection.find_one_and_update.return_value = None

    await repo.claim_pending("worker-a")

    query, _, _, _ = _call_parts(repo)
    assert query["status"] == DocumentStatus.PENDING.value
    assert query["$expr"] == {
        "$eq": [
            {"$type": "$file_id"},
            "objectId",
        ]
    }
    assert query["$or"] == [
        {"next_retry_at": {"$exists": False}},
        {"next_retry_at": None},
        {"next_retry_at": {"$lte": NOW}},
    ]
    repo.collection.find_one_and_update.assert_awaited_once()


@pytest.mark.asyncio
async def test_claim_sorts_by_creation_then_id(repo_with_mocks, fixed_clock):
    repo = repo_with_mocks
    repo.collection.find_one_and_update.return_value = None

    await repo.claim_pending("worker-a")

    _, _, sort, _ = _call_parts(repo)
    assert sort == [("created_at", ASCENDING), ("_id", ASCENDING)]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("lease_duration", "expected_seconds"),
    [
        (timedelta(seconds=60), 60),
        (timedelta(seconds=23), 23),
    ],
)
async def test_claim_sets_processing_owner_lease_and_consistent_timestamps(
    repo_with_mocks, fixed_clock, lease_duration, expected_seconds
):
    repo = repo_with_mocks
    repo.collection.find_one_and_update.return_value = None

    await repo.claim_pending("worker-a", lease_duration=lease_duration)

    _, update, _, kwargs = _call_parts(repo)
    fields = _set_stage(update)
    assert fields["status"] == DocumentStatus.PROCESSING.value
    assert _literal_value(fields["worker_id"]) == "worker-a"
    assert fields["lease_until"] == NOW + timedelta(seconds=expected_seconds)
    assert fields["heartbeat_at"] == NOW
    assert fields["updated_at"] == NOW
    assert kwargs["return_document"] == ReturnDocument.AFTER


@pytest.mark.asyncio
async def test_claim_encodes_dollar_prefixed_worker_id_as_literal(
    repo_with_mocks, fixed_clock
):
    repo = repo_with_mocks
    repo.collection.find_one_and_update.return_value = None

    await repo.claim_pending("$workerField")

    _, update, _, _ = _call_parts(repo)
    assert _set_stage(update)["worker_id"] == {"$literal": "$workerField"}


@pytest.mark.asyncio
async def test_claim_uses_sixty_second_default_lease(repo_with_mocks, fixed_clock):
    repo = repo_with_mocks
    repo.collection.find_one_and_update.return_value = None

    await repo.claim_pending("worker-a")

    _, update, _, _ = _call_parts(repo)
    assert _set_stage(update)["lease_until"] == NOW + timedelta(seconds=60)


@pytest.mark.asyncio
async def test_each_claim_generates_a_new_string_fencing_token(
    repo_with_mocks, fixed_clock
):
    repo = repo_with_mocks
    repo.collection.find_one_and_update.return_value = None

    await repo.claim_pending("worker-a")
    await repo.claim_pending("worker-a")

    tokens = [
        _literal_value(_set_stage(call.args[1])["fencing_token"])
        for call in repo.collection.find_one_and_update.call_args_list
    ]
    assert len(tokens) == 2
    assert all(isinstance(token, str) for token in tokens)
    assert UUID(tokens[0]) != UUID(tokens[1])


@pytest.mark.asyncio
async def test_claim_initializes_cycle_atomically_when_missing_or_null(
    repo_with_mocks, fixed_clock
):
    repo = repo_with_mocks
    repo.collection.find_one_and_update.return_value = None

    await repo.claim_pending("worker-a")

    _, update, _, _ = _call_parts(repo)
    expression = _set_stage(update)["processing_cycle_id"]
    assert isinstance(expression, dict)
    assert "$ifNull" in expression
    source, initial_cycle = expression["$ifNull"]
    assert source == "$processing_cycle_id"
    assert isinstance(_literal_value(initial_cycle), str)
    UUID(_literal_value(initial_cycle))


@pytest.mark.asyncio
async def test_claim_does_not_change_attempt_count(repo_with_mocks, fixed_clock):
    repo = repo_with_mocks
    repo.collection.find_one_and_update.return_value = None

    await repo.claim_pending("worker-a")

    _, update, _, _ = _call_parts(repo)
    assert "attempt_count" not in _set_stage(update)


@pytest.mark.asyncio
async def test_claim_returns_the_updated_document_as_document_model(
    repo_with_mocks, fixed_clock
):
    repo = repo_with_mocks
    claimed = _claimed_document(processing_cycle_id="cycle-1", attempt_count=3)
    repo.collection.find_one_and_update.return_value = claimed

    result = await repo.claim_pending("worker-a")

    assert isinstance(result, DocumentModel)
    assert result.id == str(claimed["_id"])
    assert result.status == DocumentStatus.PROCESSING
    assert result.worker_id == "worker-a"
    assert result.processing_cycle_id == "cycle-1"
    assert result.attempt_count == 3


@pytest.mark.asyncio
async def test_claim_returns_none_when_no_eligible_document_exists(
    repo_with_mocks, fixed_clock
):
    repo = repo_with_mocks
    repo.collection.find_one_and_update.return_value = None

    result = await repo.claim_pending("worker-a")

    assert result is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "connection_error",
    [
        ServerSelectionTimeoutError("unavailable"),
        AutoReconnect("unavailable"),
        ConnectionFailure("unavailable"),
        NetworkTimeout("unavailable"),
    ],
)
async def test_claim_translates_mongo_connectivity_errors(
    repo_with_mocks, fixed_clock, connection_error
):
    repo = repo_with_mocks
    repo.collection.find_one_and_update.side_effect = connection_error

    with pytest.raises(DatabaseUnavailableError):
        await repo.claim_pending("worker-a")


@pytest.mark.asyncio
async def test_claim_is_independent_of_upload_ownership_fields(
    repo_with_mocks, fixed_clock
):
    repo = repo_with_mocks
    repo.collection.find_one_and_update.return_value = None

    await repo.claim_pending("worker-a")

    query, update, _, _ = _call_parts(repo)
    update_text = repr(update)
    query_text = repr(query)
    for upload_field in (
        "upload_owner_id",
        "upload_lease_until",
        "upload_fencing_token",
    ):
        assert upload_field not in query_text
        assert upload_field not in update_text


@pytest.mark.asyncio
async def test_claim_uses_one_atomic_operation_without_pre_read(
    repo_with_mocks, fixed_clock
):
    repo = repo_with_mocks
    repo.collection.find_one_and_update.return_value = None

    await repo.claim_pending("worker-a")

    repo.collection.find_one_and_update.assert_awaited_once()
    repo.collection.find_one.assert_not_called()
    repo.collection.find.assert_not_called()
    repo.collection.update_one.assert_not_called()
    repo.collection.update_many.assert_not_called()
    repo.collection.replace_one.assert_not_called()
    repo.collection.aggregate.assert_not_called()


@pytest.mark.asyncio
async def test_renewal_atomically_filters_current_processing_owner_and_live_lease(
    repo_with_mocks, fixed_clock
):
    repo = repo_with_mocks
    document_id = str(ObjectId())
    repo.collection.update_one.return_value = MagicMock(matched_count=1)

    result = await repo.renew_processing_lease(
        document_id,
        "worker-a",
        "fencing-token-a",
    )

    assert result is True
    query, _, kwargs = _renewal_call_parts(repo)
    assert query == {
        "_id": ObjectId(document_id),
        "status": DocumentStatus.PROCESSING.value,
        "worker_id": "worker-a",
        "fencing_token": "fencing-token-a",
        "lease_until": {"$gte": NOW},
    }
    assert isinstance(query["_id"], ObjectId)
    assert query["lease_until"] == {"$gte": NOW}
    assert kwargs == {"upsert": False}


@pytest.mark.asyncio
async def test_renewal_pipeline_maximizes_only_the_three_lease_timestamps(
    repo_with_mocks, fixed_clock
):
    repo = repo_with_mocks
    repo.collection.update_one.return_value = MagicMock(matched_count=1)

    await repo.renew_processing_lease(
        str(ObjectId()),
        "worker-a",
        "fencing-token-a",
    )

    _, update, _ = _renewal_call_parts(repo)
    fields = _pipeline_set(update)
    assert set(fields) == {"heartbeat_at", "lease_until", "updated_at"}
    expected_values = {
        "heartbeat_at": NOW,
        "lease_until": NOW + timedelta(seconds=60),
        "updated_at": NOW,
    }
    for field, expected_value in expected_values.items():
        assert fields[field]["$max"][0] == f"${field}"
        value = fields[field]["$max"][1]
        if isinstance(value, dict) and "$literal" in value:
            value = value["$literal"]
        assert value == expected_value


@pytest.mark.asyncio
async def test_renewal_returns_false_when_mongo_matches_zero_documents(
    repo_with_mocks, fixed_clock
):
    repo = repo_with_mocks
    repo.collection.update_one.return_value = MagicMock(matched_count=0)

    result = await repo.renew_processing_lease(
        str(ObjectId()),
        "worker-a",
        "fencing-token-a",
    )

    assert result is False
    repo.collection.update_one.assert_awaited_once()


@pytest.mark.asyncio
async def test_renewal_rejects_invalid_document_id_without_mongodb_operations(
    repo_with_mocks, fixed_clock
):
    repo = repo_with_mocks

    with pytest.raises(ValueError):
        await repo.renew_processing_lease(
            "identificador-invalido",
            "worker-a",
            "fencing-token-a",
        )

    repo.collection.update_one.assert_not_awaited()
    repo.collection.find_one_and_update.assert_not_awaited()
    repo.collection.find_one.assert_not_awaited()
    repo.collection.find.assert_not_called()
    repo.collection.update_many.assert_not_awaited()
    repo.collection.replace_one.assert_not_awaited()
    repo.collection.aggregate.assert_not_called()


@pytest.mark.parametrize(
    "rejection_reason",
    [
        "wrong-status",
        "wrong-worker-id",
        "wrong-fencing-token",
        "expired-lease",
        "document-does-not-exist",
    ],
)
@pytest.mark.asyncio
async def test_renewal_reports_false_for_every_conditional_rejection(
    repo_with_mocks, fixed_clock, rejection_reason
):
    repo = repo_with_mocks
    repo.collection.update_one.return_value = MagicMock(matched_count=0)

    result = await repo.renew_processing_lease(
        str(ObjectId()),
        "worker-a",
        "fencing-token-a",
    )

    assert result is False, rejection_reason
    repo.collection.update_one.assert_awaited_once()


@pytest.mark.parametrize(
    "connection_error",
    [
        ServerSelectionTimeoutError("unavailable"),
        AutoReconnect("unavailable"),
        ConnectionFailure("unavailable"),
        NetworkTimeout("unavailable"),
    ],
)
@pytest.mark.asyncio
async def test_renewal_translates_mongo_connectivity_errors(
    repo_with_mocks, fixed_clock, connection_error
):
    repo = repo_with_mocks
    repo.collection.update_one.side_effect = connection_error

    with pytest.raises(DatabaseUnavailableError):
        await repo.renew_processing_lease(
            str(ObjectId()),
            "worker-a",
            "fencing-token-a",
        )


@pytest.mark.asyncio
async def test_renewal_propagates_unexpected_errors(repo_with_mocks, fixed_clock):
    repo = repo_with_mocks
    repo.collection.update_one.side_effect = RuntimeError("unexpected update failure")

    with pytest.raises(RuntimeError, match="unexpected update failure"):
        await repo.renew_processing_lease(
            str(ObjectId()),
            "worker-a",
            "fencing-token-a",
        )


@pytest.mark.asyncio
async def test_renewal_does_not_read_or_mutate_processing_or_upload_ownership_fields(
    repo_with_mocks, fixed_clock
):
    repo = repo_with_mocks
    repo.collection.update_one.return_value = MagicMock(matched_count=1)

    await repo.renew_processing_lease(
        str(ObjectId()),
        "worker-a",
        "fencing-token-a",
    )

    query, update, _ = _renewal_call_parts(repo)
    update_fields = _pipeline_set(update)
    assert not {
        "processing_cycle_id",
        "attempt_count",
        "worker_id",
        "fencing_token",
    }.intersection(update_fields)
    assert not any(field.startswith("upload_") for field in update_fields)
    query_and_update = repr((query, update))
    assert "upload_owner_id" not in query_and_update
    assert "upload_lease_until" not in query_and_update
    assert "upload_fencing_token" not in query_and_update

    repo.collection.update_one.assert_awaited_once()
    repo.collection.find_one_and_update.assert_not_awaited()
    repo.collection.find_one.assert_not_awaited()
    repo.collection.find.assert_not_called()
    repo.collection.update_many.assert_not_awaited()
    repo.collection.replace_one.assert_not_awaited()
    repo.collection.aggregate.assert_not_called()


@pytest.mark.asyncio
async def test_recover_expired_processing_returns_updated_document_and_atomic_contract(
    repo_with_mocks, fixed_clock
):
    repo = repo_with_mocks
    previous_token = str(UUID(int=1))
    raw_document = _claimed_document(
        processing_cycle_id="cycle-preserved",
        attempt_count=7,
        fencing_token=previous_token,
    )
    raw_document.update(
        {
            "status": DocumentStatus.PROCESSING.value,
            "next_retry_at": NOW + timedelta(minutes=2),
            "upload_owner_id": "upload-owner",
            "upload_lease_until": NOW + timedelta(minutes=10),
            "upload_fencing_token": "upload-token",
        }
    )
    repo.collection.find_one_and_update.return_value = raw_document
    worker_id = "new-recovery-worker"

    result = await repo.recover_expired_processing(worker_id)

    assert isinstance(result, DocumentModel)
    assert result.id == str(raw_document["_id"])
    query, update, sort, kwargs = _call_parts(repo)
    assert query == {
        "status": DocumentStatus.PROCESSING.value,
        "lease_until": {"$lt": NOW},
        "$or": [
            {"worker_id": {"$exists": False}},
            {"worker_id": None},
            {"worker_id": {"$ne": worker_id}},
        ],
    }
    assert sort == [("lease_until", ASCENDING), ("_id", ASCENDING)]
    assert kwargs == {
        "sort": [("lease_until", ASCENDING), ("_id", ASCENDING)],
        "return_document": ReturnDocument.AFTER,
        "upsert": False,
    }

    fields = _set_stage(update)
    assert set(fields) == {
        "worker_id",
        "lease_until",
        "heartbeat_at",
        "fencing_token",
    }
    assert _literal_value(fields["worker_id"]) == worker_id
    assert fields["lease_until"] == NOW + timedelta(seconds=60)
    assert fields["heartbeat_at"] == NOW
    new_token = UUID(_literal_value(fields["fencing_token"]))
    assert str(new_token) != previous_token
    assert DocumentStatus(result.status) is DocumentStatus.PROCESSING

    protected_fields = {
        "updated_at",
        "processing_cycle_id",
        "attempt_count",
        "next_retry_at",
        "upload_owner_id",
        "upload_lease_until",
        "upload_fencing_token",
    }
    assert not protected_fields.intersection(fields)
    assert repo.collection.find_one_and_update.await_count == 1


@pytest.mark.asyncio
async def test_recover_expired_processing_returns_none_without_eligible_document(
    repo_with_mocks, fixed_clock
):
    repo = repo_with_mocks
    repo.collection.find_one_and_update.return_value = None

    result = await repo.recover_expired_processing("recovery-worker")

    assert result is None
    repo.collection.find_one_and_update.assert_awaited_once()


@pytest.mark.parametrize(
    "connection_error",
    [
        ServerSelectionTimeoutError("unavailable"),
        AutoReconnect("unavailable"),
        ConnectionFailure("unavailable"),
        NetworkTimeout("unavailable"),
    ],
)
@pytest.mark.asyncio
async def test_recover_expired_processing_translates_connectivity_errors(
    repo_with_mocks, fixed_clock, connection_error
):
    repo = repo_with_mocks
    repo.collection.find_one_and_update.side_effect = connection_error

    with pytest.raises(DatabaseUnavailableError):
        await repo.recover_expired_processing("recovery-worker")


@pytest.mark.asyncio
async def test_recover_expired_processing_propagates_unexpected_errors(
    repo_with_mocks, fixed_clock
):
    repo = repo_with_mocks
    repo.collection.find_one_and_update.side_effect = RuntimeError(
        "unexpected recovery failure"
    )

    with pytest.raises(RuntimeError, match="unexpected recovery failure"):
        await repo.recover_expired_processing("recovery-worker")


@pytest.mark.asyncio
async def test_recover_expired_processing_uses_only_one_atomic_operation(
    repo_with_mocks, fixed_clock
):
    repo = repo_with_mocks
    repo.collection.find_one_and_update.return_value = None

    await repo.recover_expired_processing("recovery-worker")

    repo.collection.find_one_and_update.assert_awaited_once()
    repo.collection.find_one.assert_not_awaited()
    repo.collection.find.assert_not_called()
    repo.collection.update_one.assert_not_awaited()
    repo.collection.update_many.assert_not_awaited()
    repo.collection.replace_one.assert_not_awaited()
    repo.collection.aggregate.assert_not_called()


@pytest.mark.asyncio
async def test_recover_expired_processing_protects_dollar_prefixed_worker_id(
    repo_with_mocks, fixed_clock
):
    repo = repo_with_mocks
    repo.collection.find_one_and_update.return_value = _claimed_document(
        fencing_token=str(UUID(int=1))
    )
    worker_id = "$recovery-worker"

    await repo.recover_expired_processing(worker_id)

    _, update, _, _ = _call_parts(repo)
    worker_expression = _set_stage(update)["worker_id"]
    assert worker_expression == {"$literal": worker_id}

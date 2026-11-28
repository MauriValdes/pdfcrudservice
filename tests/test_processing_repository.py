"""Pruebas unitarias RED del reclamo y renovación de leases de procesamiento."""

import asyncio
from datetime import datetime, timedelta, timezone, tzinfo
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest
from bson import Int64, ObjectId
from pymongo import ASCENDING, ReturnDocument
from pymongo.errors import (
    AutoReconnect,
    ConnectionFailure,
    DocumentTooLarge,
    NetworkTimeout,
    ServerSelectionTimeoutError,
    WTimeoutError,
    WriteConcernError,
)
from bson.errors import InvalidDocument

from src.core.exceptions import DatabaseUnavailableError
from src.models.document import DocumentModel, DocumentStatus
from src.repository import processing_repository
from src.repository.processing_repository import ProcessingRepository


class FrozenDateTime:
    @classmethod
    def now(cls, tz=None):
        assert tz is timezone.utc
        return NOW


class NoOffsetTZ(tzinfo):
    def utcoffset(self, dt):
        return None


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


def _new_cycle_expression():
    no_cycle_id = {
        "$in": [
            {"$type": "$processing_cycle_id"},
            ["missing", "null"],
        ]
    }
    no_cycle_started_at = {
        "$in": [
            {"$type": "$processing_cycle_started_at"},
            ["missing", "null"],
        ]
    }
    no_prior_processing_metadata = {
        "$and": [
            {
                "$in": [
                    {"$type": f"${field}"},
                    ["missing", "null"],
                ]
            }
            for field in (
                "last_attempt_at",
                "last_error",
                "completed_at",
                "failed_at",
                "next_retry_at",
                "worker_id",
                "lease_until",
                "heartbeat_at",
                "fencing_token",
            )
        ]
    }
    valid_initial_attempt_count = {
        "$cond": [
            {
                "$in": [
                    {"$type": "$attempt_count"},
                    ["missing", "int", "long"],
                ]
            },
            {"$eq": [{"$ifNull": ["$attempt_count", 0]}, 0]},
            False,
        ]
    }
    return {
        "$and": [
            no_cycle_id,
            no_cycle_started_at,
            valid_initial_attempt_count,
            no_prior_processing_metadata,
        ]
    }


def _existing_cycle_expression(now=NOW):
    return {
        "$and": [
            {"$eq": [{"$type": "$processing_cycle_id"}, "string"]},
            {"$ne": ["$processing_cycle_id", ""]},
            {"$eq": [{"$type": "$processing_cycle_started_at"}, "date"]},
            {"$lte": ["$processing_cycle_started_at", now]},
        ]
    }


def _cycle_eligibility_expression(now=NOW):
    return {
        "$or": [
            _new_cycle_expression(),
            _existing_cycle_expression(now),
        ]
    }


def _attempt_count_expression():
    return {
        "$cond": [
            {"$eq": [{"$type": "$attempt_count"}, "missing"]},
            True,
            {
                "$cond": [
                    {
                        "$in": [
                            {"$type": "$attempt_count"},
                            ["int", "long"],
                        ]
                    },
                    {
                        "$and": [
                            {"$gte": ["$attempt_count", 0]},
                            {"$lte": ["$attempt_count", 4]},
                        ]
                    },
                    False,
                ]
            },
        ]
    }


def _attempt_query(
    document_id,
    *,
    now=NOW,
    worker_id="worker-a",
    fencing_token="fencing-a",
    processing_cycle_id="cycle-a",
):
    return {
        "_id": document_id,
        "status": DocumentStatus.PROCESSING.value,
        "worker_id": worker_id,
        "fencing_token": fencing_token,
        "processing_cycle_id": processing_cycle_id,
        "$expr": {
            "$and": [
                {"$eq": [{"$type": "$status"}, "string"]},
                {"$eq": [{"$type": "$worker_id"}, "string"]},
                {"$eq": [{"$type": "$fencing_token"}, "string"]},
                {"$eq": [{"$type": "$lease_until"}, "date"]},
                {"$gte": ["$lease_until", now]},
                {
                    "$and": [
                        _existing_cycle_expression(now),
                        {
                            "$gt": [
                                "$processing_cycle_started_at",
                                now - timedelta(minutes=10),
                            ]
                        },
                    ]
                },
                _attempt_count_expression(),
                {
                    "$in": [
                        {"$type": "$last_attempt_at"},
                        ["missing", "null", "date"],
                    ]
                },
                {
                    "$in": [
                        {"$type": "$updated_at"},
                        ["missing", "null", "date"],
                    ]
                },
            ]
        },
    }


def _completion_query(
    document_id,
    *,
    now=NOW,
    worker_id="worker-a",
    fencing_token="fencing-a",
    processing_cycle_id="cycle-a",
    attempt_number=2,
):
    return {
        "_id": document_id,
        "status": "processing",
        "worker_id": worker_id,
        "fencing_token": fencing_token,
        "processing_cycle_id": processing_cycle_id,
        "$expr": {
            "$and": [
                {"$eq": [{"$type": "$status"}, "string"]},
                {"$eq": [{"$type": "$worker_id"}, "string"]},
                {"$eq": [{"$type": "$fencing_token"}, "string"]},
                {"$eq": [{"$type": "$lease_until"}, "date"]},
                {"$gte": ["$lease_until", now]},
                _existing_cycle_expression(now),
                {
                    "$in": [
                        {"$type": "$attempt_count"},
                        ["int", "long"],
                    ]
                },
                {"$eq": ["$attempt_count", attempt_number]},
                {"$eq": [{"$type": "$last_attempt_at"}, "date"]},
                {
                    "$in": [
                        {"$type": "$updated_at"},
                        ["missing", "null", "date"],
                    ]
                },
                {
                    "$in": [
                        {"$type": "$completed_at"},
                        ["missing", "null"],
                    ]
                },
                {
                    "$in": [
                        {"$type": "$next_retry_at"},
                        ["missing", "null", "date"],
                    ]
                },
                {
                    "$in": [
                        {"$type": "$failed_at"},
                        ["missing", "null"],
                    ]
                },
                {
                    "$in": [
                        {"$type": "$original_cleanup_status"},
                        ["missing", "null"],
                    ]
                },
                {
                    "$in": [
                        {"$type": "$original_deleted_at"},
                        ["missing", "null"],
                    ]
                },
                {"$eq": [{"$type": "$file_id"}, "objectId"]},
            ]
        },
    }


def _claim_update_parts(update):
    assert isinstance(update, list)
    assert len(update) == 1
    assert "$replaceWith" in update[0], (
        "claim_pending debe usar una actualización condicional conjunta "
        "de los dos campos del ciclo"
    )
    root = update[0]["$replaceWith"]["$let"]
    merge_objects = root["in"]["$mergeObjects"]
    return (
        root["vars"]["initialize_cycle"],
        merge_objects[1]["$cond"],
        merge_objects[2],
        merge_objects,
    )


def _literal_value(expression):
    if isinstance(expression, dict) and "$literal" in expression:
        return expression["$literal"]
    return expression


def _claimed_document(
    *,
    file_id=None,
    processing_cycle_id="cycle-existing",
    processing_cycle_started_at=NOW,
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
    if processing_cycle_started_at is not ...:
        document["processing_cycle_started_at"] = processing_cycle_started_at
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
        "$and": [
            {"$eq": [{"$type": "$file_id"}, "objectId"]},
            _cycle_eligibility_expression(),
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
    _, _, fields, merge_objects = _claim_update_parts(update)
    assert fields["status"] == DocumentStatus.PROCESSING.value
    assert _literal_value(fields["worker_id"]) == "worker-a"
    assert _literal_value(fields["lease_until"]) == NOW + timedelta(
        seconds=expected_seconds
    )
    assert _literal_value(fields["heartbeat_at"]) == NOW
    assert _literal_value(fields["updated_at"]) == NOW
    assert kwargs["return_document"] == ReturnDocument.AFTER
    assert kwargs["upsert"] is False
    assert "$$ROOT" in merge_objects


@pytest.mark.asyncio
async def test_claim_encodes_dollar_prefixed_worker_id_as_literal(
    repo_with_mocks, fixed_clock
):
    repo = repo_with_mocks
    repo.collection.find_one_and_update.return_value = None

    await repo.claim_pending("$workerField")

    _, update, _, _ = _call_parts(repo)
    _, _, fields, _ = _claim_update_parts(update)
    assert fields["worker_id"] == {"$literal": "$workerField"}


@pytest.mark.asyncio
async def test_claim_uses_sixty_second_default_lease(repo_with_mocks, fixed_clock):
    repo = repo_with_mocks
    repo.collection.find_one_and_update.return_value = None

    await repo.claim_pending("worker-a")

    _, update, _, _ = _call_parts(repo)
    _, _, fields, _ = _claim_update_parts(update)
    assert fields["lease_until"] == {"$literal": NOW + timedelta(seconds=60)}


@pytest.mark.asyncio
async def test_each_claim_generates_a_new_string_fencing_token(
    repo_with_mocks, fixed_clock
):
    repo = repo_with_mocks
    repo.collection.find_one_and_update.return_value = None

    await repo.claim_pending("worker-a")
    await repo.claim_pending("worker-a")

    tokens = [
        _literal_value(_claim_update_parts(call.args[1])[2]["fencing_token"])
        for call in repo.collection.find_one_and_update.call_args_list
    ]
    assert len(tokens) == 2
    assert all(isinstance(token, str) for token in tokens)
    assert UUID(tokens[0]) != UUID(tokens[1])


@pytest.mark.asyncio
async def test_claim_initializes_cycle_pair_with_one_original_document_condition(
    repo_with_mocks, fixed_clock
):
    repo = repo_with_mocks
    repo.collection.find_one_and_update.return_value = None

    await repo.claim_pending("worker-a")

    _, update, _, _ = _call_parts(repo)
    condition, cycle_pair_condition, claim_fields, merge_objects = (
        _claim_update_parts(update)
    )
    assert condition == _new_cycle_expression()
    assert cycle_pair_condition[0] == "$$initialize_cycle"
    initial_pair = cycle_pair_condition[1]
    initial_cycle = _literal_value(initial_pair["processing_cycle_id"])
    initial_started_at = _literal_value(
        initial_pair["processing_cycle_started_at"]
    )
    UUID(initial_cycle)
    assert initial_started_at == NOW
    assert cycle_pair_condition[2] == {}
    assert merge_objects[0] == "$$ROOT"
    assert set(initial_pair) == {
        "processing_cycle_id",
        "processing_cycle_started_at",
    }
    assert "attempt_count" not in claim_fields
    assert "attempt_count" not in repr(merge_objects)


@pytest.mark.asyncio
async def test_claim_pipeline_preserves_existing_cycle_pair_from_original_document(
    repo_with_mocks, fixed_clock
):
    repo = repo_with_mocks
    repo.collection.find_one_and_update.return_value = None

    await repo.claim_pending("worker-a")

    _, update, _, _ = _call_parts(repo)
    condition, cycle_pair_condition, _, merge_objects = _claim_update_parts(update)

    assert condition == _new_cycle_expression()
    assert cycle_pair_condition[0] == "$$initialize_cycle"
    assert set(cycle_pair_condition[1]) == {
        "processing_cycle_id",
        "processing_cycle_started_at",
    }
    assert cycle_pair_condition[2] == {}
    assert merge_objects[0] == "$$ROOT"
    assert "$processing_cycle_id" not in repr(cycle_pair_condition[1])
    assert "$processing_cycle_started_at" not in repr(cycle_pair_condition[1])


@pytest.mark.asyncio
async def test_claim_captures_utc_clock_once_for_cycle_and_lease(monkeypatch, repo_with_mocks):
    calls = []

    class CountingClock:
        @classmethod
        def now(cls, tz=None):
            calls.append(tz)
            assert tz is timezone.utc
            return NOW

    monkeypatch.setattr(processing_repository, "datetime", CountingClock)
    repo = repo_with_mocks
    repo.collection.find_one_and_update.return_value = None

    await repo.claim_pending("worker-a")

    assert calls == [timezone.utc]
    _, update, _, _ = _call_parts(repo)
    _, cycle_pair_condition, claim_fields, _ = _claim_update_parts(update)
    initial_pair = cycle_pair_condition[1]
    assert _literal_value(initial_pair["processing_cycle_started_at"]) == NOW
    assert _literal_value(claim_fields["heartbeat_at"]) == NOW
    assert _literal_value(claim_fields["lease_until"]) == NOW + timedelta(seconds=60)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "evidence_field",
    [
        "last_attempt_at",
        "last_error",
        "completed_at",
        "failed_at",
        "next_retry_at",
        "worker_id",
        "lease_until",
        "heartbeat_at",
        "fencing_token",
    ],
)
async def test_new_cycle_filter_rejects_prior_processing_evidence(
    repo_with_mocks, fixed_clock, evidence_field
):
    repo = repo_with_mocks
    repo.collection.find_one_and_update.return_value = None

    await repo.claim_pending("worker-a")

    query, _, _, _ = _call_parts(repo)
    encoded_query = repr(query["$expr"])
    assert evidence_field in encoded_query


@pytest.mark.asyncio
async def test_claim_preserves_unrelated_fields_and_upload_metadata(repo_with_mocks, fixed_clock):
    repo = repo_with_mocks
    repo.collection.find_one_and_update.return_value = None

    await repo.claim_pending("worker-a")

    _, update, _, _ = _call_parts(repo)
    _, _, claim_fields, merge_objects = _claim_update_parts(update)
    assert merge_objects[0] == "$$ROOT"
    assert set(claim_fields) == {
        "status",
        "worker_id",
        "lease_until",
        "heartbeat_at",
        "updated_at",
        "fencing_token",
    }
    assert not any(key.startswith("upload_") for key in claim_fields)


@pytest.mark.asyncio
async def test_claim_does_not_change_attempt_count(repo_with_mocks, fixed_clock):
    repo = repo_with_mocks
    repo.collection.find_one_and_update.return_value = None

    await repo.claim_pending("worker-a")

    _, update, _, _ = _call_parts(repo)
    _, _, claim_fields, merge_objects = _claim_update_parts(update)
    assert "attempt_count" not in claim_fields
    assert all("attempt_count" not in repr(part) for part in merge_objects[1:])


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
        "processing_cycle_started_at",
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
        "$expr": _existing_cycle_expression(),
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
        "processing_cycle_started_at",
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


@pytest.mark.asyncio
async def test_record_ms2_attempt_builds_exact_atomic_filter_and_pipeline(monkeypatch, repo_with_mocks):
    calls = []
    raw_now = NOW.replace(microsecond=123456)
    normalized_now = NOW.replace(microsecond=123000)

    class CountingDateTime:
        @classmethod
        def now(cls, tz=None):
            calls.append(tz)
            assert tz is timezone.utc
            return raw_now

    monkeypatch.setattr(processing_repository, "datetime", CountingDateTime)
    repo = repo_with_mocks
    repo.collection.find_one_and_update.return_value = {
        "attempt_count": Int64(3),
    }
    document_id = ObjectId()

    result = await repo.record_ms2_attempt(
        str(document_id),
        "worker-a",
        "fencing-a",
        "cycle-a",
    )

    assert type(result) is int
    assert result == 3
    assert calls == [timezone.utc]
    call = repo.collection.find_one_and_update.call_args
    assert call is not None
    query, update = call.args
    assert query == _attempt_query(document_id, now=normalized_now)
    assert update == [
        {
            "$set": {
                "attempt_count": {
                    "$add": [{"$ifNull": ["$attempt_count", 0]}, 1]
                },
                "last_attempt_at": {
                    "$max": [
                        {
                            "$ifNull": [
                                "$last_attempt_at",
                                {"$literal": normalized_now},
                            ]
                        },
                        {"$literal": normalized_now},
                    ]
                },
                "updated_at": {
                    "$max": [
                        {
                            "$ifNull": [
                                "$updated_at",
                                {"$literal": normalized_now},
                            ]
                        },
                        {"$literal": normalized_now},
                    ]
                },
            }
        }
    ]
    assert call.kwargs == {
        "return_document": ReturnDocument.AFTER,
        "upsert": False,
    }
    repo.collection.find_one_and_update.assert_awaited_once()
    repo.collection.find_one.assert_not_awaited()
    repo.collection.find.assert_not_called()
    repo.collection.update_one.assert_not_awaited()
    repo.collection.update_many.assert_not_awaited()
    repo.collection.replace_one.assert_not_awaited()
    repo.collection.aggregate.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("persisted_count", "expected_count"),
    [
        (1, 1),
        (3, 3),
        (Int64(4), 4),
        (5, 5),
    ],
)
async def test_record_ms2_attempt_returns_confirmed_python_integer(
    repo_with_mocks, fixed_clock, persisted_count, expected_count
):
    repo = repo_with_mocks
    repo.collection.find_one_and_update.return_value = {
        "attempt_count": persisted_count,
    }

    result = await repo.record_ms2_attempt(
        str(ObjectId()),
        "worker-a",
        "fencing-a",
        "cycle-a",
    )

    assert type(result) is int
    assert result == expected_count
    repo.collection.find_one_and_update.assert_awaited_once()


@pytest.mark.asyncio
async def test_record_ms2_attempt_returns_none_without_diagnostic_reads(
    repo_with_mocks, fixed_clock
):
    repo = repo_with_mocks
    repo.collection.find_one_and_update.return_value = None

    result = await repo.record_ms2_attempt(
        str(ObjectId()),
        "worker-a",
        "fencing-a",
        "cycle-a",
    )

    assert result is None
    repo.collection.find_one_and_update.assert_awaited_once()
    repo.collection.find_one.assert_not_awaited()
    repo.collection.find.assert_not_called()
    repo.collection.update_one.assert_not_awaited()
    repo.collection.update_many.assert_not_awaited()
    repo.collection.replace_one.assert_not_awaited()
    repo.collection.aggregate.assert_not_called()


@pytest.mark.parametrize("document_id", ["", "invalid-object-id", "123456789012"])
@pytest.mark.asyncio
async def test_record_ms2_attempt_rejects_invalid_document_id_before_mongodb(
    repo_with_mocks, fixed_clock, document_id
):
    repo = repo_with_mocks

    with pytest.raises(ValueError):
        await repo.record_ms2_attempt(
            document_id,
            "worker-a",
            "fencing-a",
            "cycle-a",
        )

    repo.collection.find_one_and_update.assert_not_awaited()
    repo.collection.find_one.assert_not_awaited()
    repo.collection.find.assert_not_called()
    repo.collection.update_one.assert_not_awaited()
    repo.collection.update_many.assert_not_awaited()
    repo.collection.replace_one.assert_not_awaited()
    repo.collection.aggregate.assert_not_called()


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
async def test_record_ms2_attempt_translates_connectivity_errors_once(
    repo_with_mocks, fixed_clock, connection_error
):
    repo = repo_with_mocks
    repo.collection.find_one_and_update.side_effect = connection_error

    with pytest.raises(DatabaseUnavailableError):
        await repo.record_ms2_attempt(
            str(ObjectId()),
            "worker-a",
            "fencing-a",
            "cycle-a",
        )

    repo.collection.find_one_and_update.assert_awaited_once()
    repo.collection.find_one.assert_not_awaited()
    repo.collection.update_one.assert_not_awaited()
    repo.collection.update_many.assert_not_awaited()


@pytest.mark.parametrize(
    "unexpected_error",
    [
        WriteConcernError("write concern was not confirmed"),
        WTimeoutError("write concern timed out"),
        RuntimeError("unexpected attempt write failure"),
    ],
)
@pytest.mark.asyncio
async def test_record_ms2_attempt_propagates_ambiguous_and_unexpected_errors_once(
    repo_with_mocks, fixed_clock, unexpected_error
):
    repo = repo_with_mocks
    repo.collection.find_one_and_update.side_effect = unexpected_error

    with pytest.raises(type(unexpected_error)) as raised:
        await repo.record_ms2_attempt(
            str(ObjectId()),
            "worker-a",
            "fencing-a",
            "cycle-a",
        )

    assert raised.value is unexpected_error
    repo.collection.find_one_and_update.assert_awaited_once()
    repo.collection.find_one.assert_not_awaited()
    repo.collection.update_one.assert_not_awaited()
    repo.collection.update_many.assert_not_awaited()


@pytest.mark.asyncio
async def test_record_ms2_attempt_propagates_cancellation_without_retry(
    repo_with_mocks, fixed_clock
):
    repo = repo_with_mocks
    cancellation = asyncio.CancelledError()
    repo.collection.find_one_and_update.side_effect = cancellation

    with pytest.raises(asyncio.CancelledError):
        await repo.record_ms2_attempt(
            str(ObjectId()),
            "worker-a",
            "fencing-a",
            "cycle-a",
        )

    repo.collection.find_one_and_update.assert_awaited_once()
    repo.collection.find_one.assert_not_awaited()
    repo.collection.update_one.assert_not_awaited()
    repo.collection.update_many.assert_not_awaited()


@pytest.mark.asyncio
async def test_complete_processing_builds_exact_atomic_update_and_returns_match(
    monkeypatch, repo_with_mocks
):
    calls = []
    raw_now = NOW.replace(microsecond=654321)
    normalized_now = NOW.replace(microsecond=654000)

    class CountingDateTime:
        @classmethod
        def now(cls, tz=None):
            calls.append(tz)
            assert tz is timezone.utc
            return raw_now

    monkeypatch.setattr(processing_repository, "datetime", CountingDateTime)
    repo = repo_with_mocks
    repo.collection.update_one.return_value.matched_count = 1
    document_id = ObjectId()
    content = "$$ROOT\n$literal — café\nsecond line"

    result = await repo.complete_processing(
        str(document_id),
        "worker-a",
        "fencing-a",
        "cycle-a",
        2,
        content,
        3,
    )

    assert result is True
    assert calls == [timezone.utc]
    call = repo.collection.update_one.call_args
    assert call is not None
    query, update = call.args
    assert query == _completion_query(
        document_id,
        now=normalized_now,
        attempt_number=2,
    )
    assert update == [
        {
            "$set": {
                "content": {"$literal": content},
                "page_count": {"$literal": 3},
                "status": {"$literal": "completed"},
                "completed_at": {"$literal": normalized_now},
                "original_cleanup_status": {"$literal": "pending"},
                "original_deleted_at": {"$literal": None},
                "updated_at": {
                    "$max": [
                        {
                            "$ifNull": [
                                "$updated_at",
                                {"$literal": normalized_now},
                            ]
                        },
                        {"$literal": normalized_now},
                    ]
                },
            }
        },
        {
            "$unset": [
                "lease_until",
                "next_retry_at",
                "error_message",
                "last_error",
                "failed_at",
            ]
        },
    ]
    assert call.kwargs == {"upsert": False}
    repo.collection.update_one.assert_awaited_once()
    repo.collection.find_one_and_update.assert_not_awaited()
    repo.collection.find_one.assert_not_awaited()
    repo.collection.find.assert_not_called()
    repo.collection.update_many.assert_not_awaited()
    repo.collection.replace_one.assert_not_awaited()
    repo.collection.aggregate.assert_not_called()


@pytest.mark.parametrize(
    ("matched_count", "expected"),
    [(0, False), (1, True)],
)
@pytest.mark.asyncio
async def test_complete_processing_returns_only_update_matched_count(
    repo_with_mocks, fixed_clock, matched_count, expected
):
    repo = repo_with_mocks
    repo.collection.update_one.return_value.matched_count = matched_count

    result = await repo.complete_processing(
        str(ObjectId()),
        "worker-a",
        "fencing-a",
        "cycle-a",
        1,
        "",
        0,
    )

    assert result is expected
    repo.collection.update_one.assert_awaited_once()
    repo.collection.find_one.assert_not_awaited()
    repo.collection.find_one_and_update.assert_not_awaited()


@pytest.mark.asyncio
async def test_complete_processing_accepts_maximum_bson_int64_page_count(
    repo_with_mocks, fixed_clock
):
    repo = repo_with_mocks
    repo.collection.update_one.return_value.matched_count = 1

    result = await repo.complete_processing(
        str(ObjectId()),
        "worker-a",
        "fencing-a",
        "cycle-a",
        5,
        "",
        2**63 - 1,
    )

    assert result is True
    repo.collection.update_one.assert_awaited_once()


@pytest.mark.parametrize(
    ("argument_index", "invalid_value"),
    [
        (0, ""),
        (0, "invalid-object-id"),
        (0, "123456789012"),
        (0, None),
        (1, ""),
        (1, None),
        (1, 7),
        (2, ""),
        (2, None),
        (2, 7),
        (3, ""),
        (3, None),
        (3, 7),
        (4, 0),
        (4, 6),
        (4, True),
        (4, False),
        (4, 1.0),
        (4, "2"),
        (4, None),
        (5, None),
        (5, b"text"),
        (5, ["text"]),
        (6, -1),
        (6, True),
        (6, False),
        (6, 1.0),
        (6, "1"),
        (6, None),
        (6, 2**63),
    ],
)
@pytest.mark.asyncio
async def test_complete_processing_rejects_invalid_arguments_before_mongodb(
    repo_with_mocks, fixed_clock, argument_index, invalid_value
):
    repo = repo_with_mocks
    arguments = [
        str(ObjectId()),
        "worker-a",
        "fencing-a",
        "cycle-a",
        1,
        "",
        0,
    ]
    arguments[argument_index] = invalid_value

    with pytest.raises(ValueError):
        await repo.complete_processing(*arguments)

    repo.collection.update_one.assert_not_awaited()
    repo.collection.find_one_and_update.assert_not_awaited()
    repo.collection.find_one.assert_not_awaited()
    repo.collection.find.assert_not_called()
    repo.collection.update_many.assert_not_awaited()
    repo.collection.replace_one.assert_not_awaited()
    repo.collection.aggregate.assert_not_called()


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
async def test_complete_processing_translates_connectivity_errors_once(
    repo_with_mocks, fixed_clock, connection_error
):
    repo = repo_with_mocks
    repo.collection.update_one.side_effect = connection_error

    with pytest.raises(DatabaseUnavailableError):
        await repo.complete_processing(
            str(ObjectId()),
            "worker-a",
            "fencing-a",
            "cycle-a",
            1,
            "result",
            1,
        )

    repo.collection.update_one.assert_awaited_once()
    repo.collection.find_one.assert_not_awaited()
    repo.collection.find_one_and_update.assert_not_awaited()
    repo.collection.update_many.assert_not_awaited()


@pytest.mark.parametrize(
    "write_error",
    [
        WriteConcernError("write concern was not confirmed"),
        WTimeoutError("write concern timed out"),
        RuntimeError("unexpected completion write failure"),
        DocumentTooLarge("BSON document exceeds maximum size"),
        InvalidDocument("content cannot be encoded as BSON"),
    ],
)
@pytest.mark.asyncio
async def test_complete_processing_propagates_write_and_bson_errors_once(
    repo_with_mocks, fixed_clock, write_error
):
    repo = repo_with_mocks
    repo.collection.update_one.side_effect = write_error

    with pytest.raises(type(write_error)) as raised:
        await repo.complete_processing(
            str(ObjectId()),
            "worker-a",
            "fencing-a",
            "cycle-a",
            1,
            "result",
            1,
        )

    assert raised.value is write_error
    repo.collection.update_one.assert_awaited_once()
    repo.collection.find_one.assert_not_awaited()
    repo.collection.find_one_and_update.assert_not_awaited()
    repo.collection.update_many.assert_not_awaited()


@pytest.mark.asyncio
async def test_complete_processing_propagates_cancellation_without_retry(
    repo_with_mocks, fixed_clock
):
    repo = repo_with_mocks
    cancellation = asyncio.CancelledError()
    repo.collection.update_one.side_effect = cancellation

    with pytest.raises(asyncio.CancelledError):
        await repo.complete_processing(
            str(ObjectId()),
            "worker-a",
            "fencing-a",
            "cycle-a",
            1,
            "result",
            1,
        )

    repo.collection.update_one.assert_awaited_once()
    repo.collection.find_one.assert_not_awaited()
    repo.collection.find_one_and_update.assert_not_awaited()
    repo.collection.update_many.assert_not_awaited()


_RED_FAILURE_MESSAGES = {
    "ms2_invalid_input": "El archivo PDF no es válido.",
    "pdf_too_large": "El archivo PDF supera el tamaño máximo permitido.",
    "pdf_unprocessable": "No se pudo procesar el archivo PDF.",
    "ms2_http_401": (
        "No se pudo completar el procesamiento debido a un problema técnico "
        "del servicio."
    ),
    "ms2_http_403": (
        "No se pudo completar el procesamiento debido a un problema técnico "
        "del servicio."
    ),
    "ms2_http_404": (
        "No se pudo completar el procesamiento debido a un problema técnico "
        "del servicio."
    ),
    "ms2_http_unclassified": (
        "No se pudo completar el procesamiento debido a un problema técnico "
        "del servicio."
    ),
}
_RED_EXHAUSTION_MESSAGES = {
    "attempts_exhausted": "Se agotaron los intentos de procesamiento.",
    "cycle_window_exhausted": "Se agotó el tiempo disponible para el procesamiento.",
}


def _terminal_common_guards(now):
    return [
        {"$eq": [{"$type": "$status"}, "string"]},
        {"$eq": [{"$type": "$worker_id"}, "string"]},
        {"$eq": [{"$type": "$fencing_token"}, "string"]},
        {"$eq": [{"$type": "$lease_until"}, "date"]},
        {"$gte": ["$lease_until", now]},
        _existing_cycle_expression(now),
        {"$in": [{"$type": "$updated_at"}, ["missing", "null", "date"]]},
        {"$in": [{"$type": "$completed_at"}, ["missing", "null"]]},
        {"$in": [{"$type": "$failed_at"}, ["missing", "null"]]},
        {"$in": [{"$type": "$next_retry_at"}, ["missing", "null", "date"]]},
    ]


def _terminal_arguments(document_id=None):
    return (
        str(document_id or ObjectId()),
        "worker-a",
        "fencing-a",
        "cycle-a",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure_code", "public_message"),
    list(_RED_FAILURE_MESSAGES.items()),
)
async def test_fail_processing_builds_atomic_literal_safe_update(
    monkeypatch, repo_with_mocks, failure_code, public_message
):
    calls = []
    raw_now = NOW.replace(microsecond=987654)
    normalized_now = NOW.replace(microsecond=987000)

    class CountingDateTime:
        @classmethod
        def now(cls, tz=None):
            calls.append(tz)
            assert tz is timezone.utc
            return raw_now

    monkeypatch.setattr(processing_repository, "datetime", CountingDateTime)
    repo = repo_with_mocks
    repo.collection.update_one.return_value.matched_count = 1
    document_id = ObjectId()

    result = await repo.fail_processing(
        *_terminal_arguments(document_id), 3, failure_code
    )

    assert result is True
    assert calls == [timezone.utc]
    query, update = repo.collection.update_one.call_args.args
    assert query == {
        "_id": document_id,
        "status": "processing",
        "worker_id": "worker-a",
        "fencing_token": "fencing-a",
        "processing_cycle_id": "cycle-a",
        "$expr": {
            "$and": [
                *_terminal_common_guards(normalized_now),
                {"$in": [{"$type": "$attempt_count"}, ["int", "long"]]},
                {"$eq": ["$attempt_count", 3]},
                {"$eq": [{"$type": "$last_attempt_at"}, "date"]},
            ]
        },
    }
    assert update == [
        {
            "$set": {
                "status": {"$literal": "failed"},
                "error_message": {"$literal": public_message},
                "last_error": {"$literal": failure_code},
                "failed_at": {"$literal": normalized_now},
                "updated_at": {
                    "$max": [
                        {
                            "$ifNull": [
                                "$updated_at",
                                {"$literal": normalized_now},
                            ]
                        },
                        {"$literal": normalized_now},
                    ]
                },
            }
        },
        {
            "$unset": [
                "lease_until",
                "next_retry_at",
                "completed_at",
                "content",
                "page_count",
            ]
        },
    ]
    assert repo.collection.update_one.call_args.kwargs == {"upsert": False}
    repo.collection.update_one.assert_awaited_once()
    repo.collection.find_one.assert_not_awaited()
    repo.collection.find_one_and_update.assert_not_awaited()
    repo.collection.find.assert_not_called()
    repo.collection.update_many.assert_not_awaited()
    repo.collection.replace_one.assert_not_awaited()
    repo.collection.aggregate.assert_not_called()


@pytest.mark.asyncio
async def test_fail_exhausted_processing_builds_atomic_priority_pipeline(
    monkeypatch, repo_with_mocks
):
    calls = []
    raw_now = NOW.replace(microsecond=345678)
    normalized_now = NOW.replace(microsecond=345000)

    class CountingDateTime:
        @classmethod
        def now(cls, tz=None):
            calls.append(tz)
            assert tz is timezone.utc
            return raw_now

    monkeypatch.setattr(processing_repository, "datetime", CountingDateTime)
    repo = repo_with_mocks
    repo.collection.update_one.return_value.matched_count = 1
    document_id = ObjectId()
    attempts_exhausted = {
        "$and": [
            {"$in": [{"$type": "$attempt_count"}, ["int", "long"]]},
            {"$eq": ["$attempt_count", 5]},
        ]
    }
    valid_attempt_state = {
        "$or": [
            {
                "$and": [
                    {"$eq": [{"$type": "$attempt_count"}, "missing"]},
                    {
                        "$in": [
                            {"$type": "$last_attempt_at"},
                            ["missing", "null"],
                        ]
                    },
                ]
            },
            {
                "$and": [
                    {"$in": [{"$type": "$attempt_count"}, ["int", "long"]]},
                    {"$eq": ["$attempt_count", 0]},
                    {
                        "$in": [
                            {"$type": "$last_attempt_at"},
                            ["missing", "null"],
                        ]
                    },
                ]
            },
            {
                "$and": [
                    {"$in": [{"$type": "$attempt_count"}, ["int", "long"]]},
                    {"$gte": ["$attempt_count", 1]},
                    {"$lte": ["$attempt_count", 5]},
                    {"$eq": [{"$type": "$last_attempt_at"}, "date"]},
                ]
            },
        ]
    }
    window_exhausted = {
        "$lte": [
            "$processing_cycle_started_at",
            normalized_now - timedelta(minutes=10),
        ]
    }

    result = await repo.fail_exhausted_processing(*_terminal_arguments(document_id))

    assert result is True
    assert calls == [timezone.utc]
    query, update = repo.collection.update_one.call_args.args
    assert query == {
        "_id": document_id,
        "status": "processing",
        "worker_id": "worker-a",
        "fencing_token": "fencing-a",
        "processing_cycle_id": "cycle-a",
        "$expr": {
            "$and": [
                *_terminal_common_guards(normalized_now),
                valid_attempt_state,
                {"$or": [attempts_exhausted, window_exhausted]},
            ]
        },
    }
    assert update == [
        {
            "$set": {
                "status": {"$literal": "failed"},
                "error_message": {
                    "$cond": [
                        attempts_exhausted,
                        {"$literal": _RED_EXHAUSTION_MESSAGES["attempts_exhausted"]},
                        {
                            "$literal": _RED_EXHAUSTION_MESSAGES[
                                "cycle_window_exhausted"
                            ]
                        },
                    ]
                },
                "last_error": {
                    "$cond": [
                        attempts_exhausted,
                        {"$literal": "attempts_exhausted"},
                        {"$literal": "cycle_window_exhausted"},
                    ]
                },
                "failed_at": {"$literal": normalized_now},
                "updated_at": {
                    "$max": [
                        {
                            "$ifNull": [
                                "$updated_at",
                                {"$literal": normalized_now},
                            ]
                        },
                        {"$literal": normalized_now},
                    ]
                },
            }
        },
        {
            "$unset": [
                "lease_until",
                "next_retry_at",
                "completed_at",
                "content",
                "page_count",
            ]
        },
    ]
    assert repo.collection.update_one.call_args.kwargs == {"upsert": False}
    repo.collection.update_one.assert_awaited_once()
    repo.collection.find_one.assert_not_awaited()
    repo.collection.find_one_and_update.assert_not_awaited()


@pytest.mark.parametrize(
    ("attempt_number", "failure_code"),
    [
        (0, "ms2_invalid_input"),
        (6, "ms2_invalid_input"),
        (True, "ms2_invalid_input"),
        (1.0, "ms2_invalid_input"),
        (1, "attempts_exhausted"),
        (1, "cycle_window_exhausted"),
        (1, "ms2_invalid_response"),
        (1, "$where"),
        (1, ["ms2_invalid_input"]),
        (1, RuntimeError("do not persist exception text")),
    ],
)
@pytest.mark.asyncio
async def test_fail_processing_rejects_unapproved_arguments_without_mongodb(
    repo_with_mocks, fixed_clock, attempt_number, failure_code
):
    repo = repo_with_mocks
    with pytest.raises(ValueError):
        await repo.fail_processing(
            *_terminal_arguments(), attempt_number, failure_code
        )
    repo.collection.update_one.assert_not_awaited()
    repo.collection.find_one.assert_not_awaited()
    repo.collection.find_one_and_update.assert_not_awaited()


@pytest.mark.parametrize(
    "operation",
    ["fail_processing", "fail_exhausted_processing"],
)
@pytest.mark.parametrize(
    ("argument_index", "invalid_value"),
    [
        (0, None),
        (0, "not-an-object-id"),
        (1, ""),
        (1, 3),
        (2, ""),
        (2, None),
        (3, ""),
        (3, []),
    ],
)
@pytest.mark.asyncio
async def test_terminal_failure_rejects_invalid_shared_arguments_without_mongodb(
    repo_with_mocks, fixed_clock, operation, argument_index, invalid_value
):
    repo = repo_with_mocks
    arguments = list(_terminal_arguments())
    if operation == "fail_processing":
        arguments.extend([1, "ms2_invalid_input"])
    arguments[argument_index] = invalid_value

    with pytest.raises(ValueError):
        await getattr(repo, operation)(*arguments)
    repo.collection.update_one.assert_not_awaited()
    repo.collection.find_one.assert_not_awaited()
    repo.collection.find_one_and_update.assert_not_awaited()
    repo.collection.find.assert_not_called()
    repo.collection.update_many.assert_not_awaited()
    repo.collection.replace_one.assert_not_awaited()
    repo.collection.aggregate.assert_not_called()


@pytest.mark.parametrize(
    ("operation", "arguments"),
    [
        ("fail_processing", lambda: (*_terminal_arguments(), 1, "ms2_invalid_input")),
        ("fail_exhausted_processing", lambda: _terminal_arguments()),
    ],
)
@pytest.mark.asyncio
async def test_terminal_failure_operations_return_false_without_extra_operations(
    repo_with_mocks, fixed_clock, operation, arguments
):
    repo = repo_with_mocks
    repo.collection.update_one.return_value.matched_count = 0

    result = await getattr(repo, operation)(*arguments())

    assert result is False
    repo.collection.update_one.assert_awaited_once()
    assert repo.collection.update_one.call_args.kwargs == {"upsert": False}
    repo.collection.find_one.assert_not_awaited()
    repo.collection.find_one_and_update.assert_not_awaited()
    repo.collection.find.assert_not_called()
    repo.collection.update_many.assert_not_awaited()
    repo.collection.replace_one.assert_not_awaited()
    repo.collection.aggregate.assert_not_called()


@pytest.mark.parametrize(
    ("operation", "arguments"),
    [
        ("fail_processing", lambda: (*_terminal_arguments(), 1, "ms2_invalid_input")),
        ("fail_exhausted_processing", lambda: _terminal_arguments()),
    ],
)
@pytest.mark.parametrize(
    "write_error",
    [
        ServerSelectionTimeoutError("unavailable"),
        AutoReconnect("unavailable"),
        ConnectionFailure("unavailable"),
        NetworkTimeout("unavailable"),
        WriteConcernError("write concern was not confirmed"),
        WTimeoutError("write concern timed out"),
        RuntimeError("unexpected terminal write failure"),
        asyncio.CancelledError(),
    ],
)
@pytest.mark.asyncio
async def test_terminal_failure_operations_preserve_error_contract_without_retry(
    repo_with_mocks, fixed_clock, operation, arguments, write_error
):
    repo = repo_with_mocks
    repo.collection.update_one.side_effect = write_error
    method = getattr(repo, operation)

    if isinstance(
        write_error,
        (
            ServerSelectionTimeoutError,
            AutoReconnect,
            ConnectionFailure,
            NetworkTimeout,
        ),
    ):
        with pytest.raises(DatabaseUnavailableError) as raised:
            await method(*arguments())
        assert raised.value.__cause__ is write_error
    else:
        with pytest.raises(type(write_error)) as raised:
            await method(*arguments())
        assert raised.value is write_error

    repo.collection.update_one.assert_awaited_once()
    repo.collection.find_one.assert_not_awaited()
    repo.collection.find_one_and_update.assert_not_awaited()
    repo.collection.update_many.assert_not_awaited()


_TRANSIENT_RETRY_CODES = (
    "ms2_http_429",
    "ms2_http_500",
    "ms2_http_502",
    "ms2_http_503",
    "ms2_http_504",
    "ms2_timeout",
    "ms2_connection_error",
    "ms2_protocol_error",
    "ms2_invalid_response",
)


class _NoEffectiveOffset(tzinfo):
    def utcoffset(self, dt):
        return None

    def dst(self, dt):
        return None


def _retry_arguments(document_id=None):
    return (
        *_terminal_arguments(document_id),
        3,
        "ms2_http_503",
        NOW + timedelta(seconds=8),
    )


def _open_retry_arguments(document_id=None):
    return (
        *_terminal_arguments(document_id),
        2,
        NOW + timedelta(seconds=8),
    )


@pytest.mark.asyncio
async def test_schedule_transient_retry_builds_one_atomic_bounded_release(
    monkeypatch, repo_with_mocks
):
    calls = []
    raw_now = NOW.replace(microsecond=987654)
    now = NOW.replace(microsecond=987000)
    proposed_retry_utc = (NOW + timedelta(seconds=8)).replace(microsecond=456789)
    proposed_retry = proposed_retry_utc.astimezone(
        timezone(timedelta(hours=-4))
    )
    normalized_retry = proposed_retry.replace(microsecond=456000)
    normalized_retry = normalized_retry.astimezone(timezone.utc)

    class CountingDateTime:
        @classmethod
        def now(cls, tz=None):
            calls.append(tz)
            assert tz is timezone.utc
            return raw_now

    monkeypatch.setattr(processing_repository, "datetime", CountingDateTime)
    repo = repo_with_mocks
    repo.collection.update_one.return_value.matched_count = 1
    document_id = ObjectId()

    result = await repo.schedule_transient_retry(
        *_terminal_arguments(document_id),
        3,
        "ms2_http_503",
        proposed_retry,
    )

    assert result is True
    assert calls == [timezone.utc]
    query, update = repo.collection.update_one.call_args.args
    assert query["_id"] == document_id
    assert query["status"] == "processing"
    assert query["worker_id"] == "worker-a"
    assert query["fencing_token"] == "fencing-a"
    assert query["processing_cycle_id"] == "cycle-a"
    guards = query["$expr"]["$and"]
    assert _terminal_common_guards(now) == guards[: len(_terminal_common_guards(now))]
    assert {"$gt": ["$processing_cycle_started_at", now - timedelta(minutes=10)]} in guards
    assert {"$eq": ["$attempt_count", 3]} in guards
    assert {"$lt": ["$attempt_count", 5]} in guards
    assert {"$eq": [{"$type": "$last_attempt_at"}, "date"]} in guards
    assert {
        "$in": [{"$type": "$heartbeat_at"}, ["missing", "null", "date"]]
    } in guards
    assert update[0]["$set"]["status"] == {"$literal": "pending"}
    assert update[0]["$set"]["last_error"] == {"$literal": "ms2_http_503"}
    assert update[0]["$set"]["next_retry_at"] == {
        "$min": [
            {"$literal": normalized_retry},
            {"$add": ["$processing_cycle_started_at", 600000]},
        ]
    }
    assert update[0]["$set"]["updated_at"] == {
        "$max": [
            {"$ifNull": ["$updated_at", {"$literal": now}]},
            {"$literal": now},
        ]
    }
    assert update[1]["$unset"] == [
        "worker_id",
        "fencing_token",
        "lease_until",
        "heartbeat_at",
        "error_message",
        "completed_at",
        "failed_at",
        "content",
        "page_count",
    ]
    assert repo.collection.update_one.call_args.kwargs == {"upsert": False}
    repo.collection.update_one.assert_awaited_once()
    repo.collection.find_one.assert_not_awaited()
    repo.collection.find_one_and_update.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_code", _TRANSIENT_RETRY_CODES)
async def test_schedule_transient_retry_accepts_only_closed_transient_codes(
    repo_with_mocks, fixed_clock, failure_code
):
    repo = repo_with_mocks
    repo.collection.update_one.return_value.matched_count = 1

    result = await repo.schedule_transient_retry(
        *_terminal_arguments(),
        3,
        failure_code,
        NOW + timedelta(seconds=8),
    )

    assert result is True
    update = repo.collection.update_one.call_args.args[1]
    assert update[0]["$set"]["last_error"] == {"$literal": failure_code}


@pytest.mark.parametrize(
    ("attempt_number", "failure_code"),
    [
        (0, "ms2_http_503"),
        (6, "ms2_http_503"),
        (True, "ms2_http_503"),
        (3.0, "ms2_http_503"),
        (3, "ms2_invalid_input"),
        (3, "ms2_http_401"),
        (3, "ms2_http_403"),
        (3, "ms2_http_404"),
        (3, "ms2_http_unclassified"),
        (3, "attempts_exhausted"),
        (3, "cycle_window_exhausted"),
        (3, "$where"),
        (3, ["ms2_http_503"]),
        (3, RuntimeError("must not persist exception text")),
    ],
)
@pytest.mark.asyncio
async def test_schedule_transient_retry_rejects_invalid_arguments_before_mongodb(
    repo_with_mocks, fixed_clock, attempt_number, failure_code
):
    repo = repo_with_mocks
    with pytest.raises(ValueError):
        await repo.schedule_transient_retry(
            *_terminal_arguments(),
            attempt_number,
            failure_code,
            NOW + timedelta(seconds=8),
        )
    repo.collection.update_one.assert_not_awaited()


@pytest.mark.parametrize(
    ("operation", "next_retry_at"),
    [
        (operation, next_retry_at)
        for operation in ("schedule_transient_retry", "release_for_circuit_open")
        for next_retry_at in (
            "not-a-datetime",
            NOW.replace(tzinfo=None),
            NOW.replace(tzinfo=_NoEffectiveOffset()),
            NOW,
            NOW - timedelta(milliseconds=1),
            NOW + timedelta(microseconds=999),
        )
    ],
)
@pytest.mark.asyncio
async def test_retry_operations_reject_invalid_retry_instants_before_mongodb(
    repo_with_mocks, fixed_clock, operation, next_retry_at
):
    repo = repo_with_mocks
    with pytest.raises(ValueError):
        if operation == "schedule_transient_retry":
            await repo.schedule_transient_retry(
                *_terminal_arguments(),
                3,
                "ms2_http_503",
                next_retry_at,
            )
        else:
            await repo.release_for_circuit_open(
                *_terminal_arguments(),
                2,
                next_retry_at,
            )
    repo.collection.update_one.assert_not_awaited()


@pytest.mark.parametrize(
    "operation",
    ["schedule_transient_retry", "release_for_circuit_open"],
)
@pytest.mark.parametrize(
    "next_retry_at",
    [
        NOW,
        NOW - timedelta(milliseconds=1),
        NOW + timedelta(microseconds=999),
    ],
    ids=["equal-now", "past", "rounds-down-to-now"],
)
@pytest.mark.asyncio
async def test_retry_operations_raise_specific_error_for_expired_valid_timestamp(
    repo_with_mocks, fixed_clock, operation, next_retry_at
):
    repo = repo_with_mocks
    arguments = list(
        _retry_arguments()
        if operation == "schedule_transient_retry"
        else _open_retry_arguments()
    )
    arguments[-1] = next_retry_at

    with pytest.raises(ValueError) as raised:
        await getattr(repo, operation)(*arguments)

    expired_error = getattr(
        processing_repository, "RetryTimestampExpiredError", None
    )
    assert expired_error is not None
    assert issubclass(expired_error, ValueError)
    assert isinstance(raised.value, expired_error)
    repo.collection.update_one.assert_not_awaited()


@pytest.mark.parametrize(
    ("operation", "next_retry_at"),
    [
        ("schedule_transient_retry", "not-a-datetime"),
        ("schedule_transient_retry", NOW.replace(tzinfo=None)),
        ("schedule_transient_retry", NOW.replace(tzinfo=_NoEffectiveOffset())),
        ("release_for_circuit_open", "not-a-datetime"),
        ("release_for_circuit_open", NOW.replace(tzinfo=None)),
        ("release_for_circuit_open", NOW.replace(tzinfo=_NoEffectiveOffset())),
    ],
)
@pytest.mark.asyncio
async def test_invalid_retry_timestamp_is_not_expiry_specific_error(
    repo_with_mocks, fixed_clock, operation, next_retry_at
):
    repo = repo_with_mocks
    arguments = list(
        _retry_arguments()
        if operation == "schedule_transient_retry"
        else _open_retry_arguments()
    )
    arguments[-1] = next_retry_at

    with pytest.raises(ValueError) as raised:
        await getattr(repo, operation)(*arguments)

    expired_error = getattr(
        processing_repository, "RetryTimestampExpiredError", None
    )
    if expired_error is not None:
        assert not isinstance(raised.value, expired_error)
    repo.collection.update_one.assert_not_awaited()


@pytest.mark.parametrize(
    ("operation", "argument_index", "invalid_value"),
    [
        (operation, argument_index, invalid_value)
        for operation in ("schedule_transient_retry", "release_for_circuit_open")
        for argument_index, invalid_value in (
            (0, None),
            (0, "not-an-object-id"),
            (1, ""),
            (1, 3),
            (2, ""),
            (2, None),
            (3, ""),
            (3, []),
        )
    ],
)
@pytest.mark.asyncio
async def test_retry_operations_reject_invalid_shared_arguments(
    repo_with_mocks, fixed_clock, operation, argument_index, invalid_value
):
    repo = repo_with_mocks
    arguments = list(
        _retry_arguments()
        if operation == "schedule_transient_retry"
        else _open_retry_arguments()
    )
    arguments[argument_index] = invalid_value

    with pytest.raises(ValueError):
        await getattr(repo, operation)(*arguments)

    repo.collection.update_one.assert_not_awaited()


@pytest.mark.parametrize(
    ("operation", "arguments"),
    [
        ("schedule_transient_retry", _retry_arguments),
        ("release_for_circuit_open", _open_retry_arguments),
    ],
)
@pytest.mark.asyncio
async def test_retry_operations_return_false_for_unmatched_atomic_write(
    repo_with_mocks, fixed_clock, operation, arguments
):
    repo = repo_with_mocks
    repo.collection.update_one.return_value.matched_count = 0

    result = await getattr(repo, operation)(*arguments())

    assert result is False
    repo.collection.update_one.assert_awaited_once()
    assert repo.collection.update_one.call_args.kwargs == {"upsert": False}
    repo.collection.find_one.assert_not_awaited()
    repo.collection.find_one_and_update.assert_not_awaited()


@pytest.mark.asyncio
async def test_release_for_circuit_open_preserves_attempt_state_and_last_error(
    monkeypatch, repo_with_mocks
):
    raw_now = NOW.replace(microsecond=765432)
    now = NOW.replace(microsecond=765000)

    class CountingDateTime:
        calls = 0

        @classmethod
        def now(cls, tz=None):
            cls.calls += 1
            assert tz is timezone.utc
            return raw_now

    monkeypatch.setattr(processing_repository, "datetime", CountingDateTime)
    repo = repo_with_mocks
    repo.collection.update_one.return_value.matched_count = 1

    result = await repo.release_for_circuit_open(
        *_terminal_arguments(),
        2,
        NOW + timedelta(seconds=8),
    )

    assert result is True
    assert CountingDateTime.calls == 1
    query, update = repo.collection.update_one.call_args.args
    guards = query["$expr"]["$and"]
    assert {"$eq": ["$attempt_count", 2]} in guards
    assert {"$lte": ["$attempt_count", 4]} in guards
    assert {"$eq": [{"$type": "$last_attempt_at"}, "date"]} in guards
    assert {"$gt": ["$processing_cycle_started_at", now - timedelta(minutes=10)]} in guards
    assert "last_error" not in update[0]["$set"]
    assert "attempt_count" not in update[0]["$set"]
    assert "last_attempt_at" not in update[0]["$set"]
    assert update[0]["$set"]["next_retry_at"]["$min"][1] == {
        "$add": ["$processing_cycle_started_at", 600000]
    }
    assert repo.collection.update_one.call_args.kwargs == {"upsert": False}
    repo.collection.update_one.assert_awaited_once()


@pytest.mark.parametrize(
    "operation",
    [
        "fail_processing",
        "fail_exhausted_processing",
        "schedule_transient_retry",
        "release_for_circuit_open",
    ],
)
@pytest.mark.asyncio
async def test_noncompletion_operations_do_not_register_original_cleanup(
    repo_with_mocks, fixed_clock, operation
):
    repo = repo_with_mocks
    repo.collection.update_one.return_value.matched_count = 1
    arguments = _terminal_arguments()

    if operation == "fail_processing":
        await repo.fail_processing(*arguments, 3, "ms2_invalid_input")
    elif operation == "fail_exhausted_processing":
        await repo.fail_exhausted_processing(*arguments)
    elif operation == "schedule_transient_retry":
        await repo.schedule_transient_retry(
            *arguments,
            3,
            "ms2_http_503",
            NOW + timedelta(seconds=8),
        )
    else:
        await repo.release_for_circuit_open(
            *arguments,
            2,
            NOW + timedelta(seconds=8),
        )

    update = repo.collection.update_one.call_args.args[1]
    set_fields = set().union(*(stage.get("$set", {}) for stage in update))
    unset_fields = {
        field
        for stage in update
        for field in stage.get("$unset", [])
    }
    assert not {
        "original_cleanup_status",
        "original_deleted_at",
    } & (set_fields | unset_fields)


@pytest.mark.parametrize("expected_attempt_count", [-1, 6, True, 1.0, "2"])
@pytest.mark.asyncio
async def test_release_for_circuit_open_rejects_invalid_expected_counter(
    repo_with_mocks, fixed_clock, expected_attempt_count
):
    repo = repo_with_mocks
    with pytest.raises(ValueError):
        await repo.release_for_circuit_open(
            *_terminal_arguments(),
            expected_attempt_count,
            NOW + timedelta(seconds=8),
        )
    repo.collection.update_one.assert_not_awaited()


@pytest.mark.parametrize(
    ("operation", "arguments"),
    [
        ("schedule_transient_retry", lambda: _retry_arguments()),
        ("release_for_circuit_open", lambda: _open_retry_arguments()),
    ],
)
@pytest.mark.parametrize(
    "write_error",
    [
        ServerSelectionTimeoutError("unavailable"),
        AutoReconnect("unavailable"),
        ConnectionFailure("unavailable"),
        NetworkTimeout("unavailable"),
        WriteConcernError("ambiguous acknowledgement"),
        RuntimeError("unexpected write failure"),
        asyncio.CancelledError(),
    ],
)
@pytest.mark.asyncio
async def test_retry_operations_preserve_single_write_error_contract(
    repo_with_mocks, fixed_clock, operation, arguments, write_error
):
    repo = repo_with_mocks
    repo.collection.update_one.side_effect = write_error

    if isinstance(
        write_error,
        (
            ServerSelectionTimeoutError,
            AutoReconnect,
            ConnectionFailure,
            NetworkTimeout,
        ),
    ):
        with pytest.raises(DatabaseUnavailableError) as raised:
            await getattr(repo, operation)(*arguments())
        assert raised.value.__cause__ is write_error
    else:
        with pytest.raises(type(write_error)) as raised:
            await getattr(repo, operation)(*arguments())
        assert raised.value is write_error

    repo.collection.update_one.assert_awaited_once()
    assert repo.collection.update_one.call_args.kwargs == {"upsert": False}
    repo.collection.find_one.assert_not_awaited()
    repo.collection.find_one_and_update.assert_not_awaited()


def _cleanup_identity():
    return (
        str(ObjectId()),
        "cleanup-cycle",
        datetime(2026, 1, 2, 3, 4, 5, 654321, tzinfo=timezone.utc),
        ObjectId(),
    )


@pytest.mark.asyncio
async def test_validate_original_cleanup_candidate_uses_exact_identity_and_read_only_queries(
    repo_with_mocks,
):
    repo = repo_with_mocks
    document_id, cycle_id, completed_at, file_id = _cleanup_identity()
    document_object_id = ObjectId(document_id)
    normalized_completed_at = completed_at.replace(microsecond=654000)
    repo.collection.find_one.side_effect = [{"_id": document_object_id}, None]

    result = await repo.validate_original_cleanup_candidate(
        document_id,
        cycle_id,
        completed_at,
        file_id,
    )

    assert result is True
    candidate_query, projection = repo.collection.find_one.await_args_list[0].args
    assert candidate_query == {
        "_id": document_object_id,
        "status": "completed",
        "processing_cycle_id": cycle_id,
        "completed_at": normalized_completed_at,
        "file_id": file_id,
        "original_cleanup_status": "pending",
        "$expr": {
            "$and": [
                {"$eq": [{"$type": "$status"}, "string"]},
                {"$eq": [{"$type": "$processing_cycle_id"}, "string"]},
                {"$eq": [{"$type": "$completed_at"}, "date"]},
                {"$eq": [{"$type": "$file_id"}, "objectId"]},
                {"$eq": [{"$type": "$content"}, "string"]},
                {"$in": [{"$type": "$page_count"}, ["int", "long"]]},
                {"$gte": ["$page_count", 0]},
                {
                    "$eq": [
                        {"$type": "$original_cleanup_status"},
                        "string",
                    ]
                },
                {
                    "$in": [
                        {"$type": "$original_deleted_at"},
                        ["missing", "null"],
                    ]
                },
            ]
        },
    }
    assert projection == {"_id": 1}
    reference_query = repo.collection.find_one.await_args_list[1].args[0]
    assert reference_query == {
        "_id": {"$ne": document_object_id},
        "file_id": {"$in": [file_id]},
    }
    assert repo.collection.find_one.await_count == 2
    repo.collection.update_one.assert_not_awaited()
    repo.collection.update_many.assert_not_awaited()
    repo.collection.find_one_and_update.assert_not_awaited()


@pytest.mark.parametrize("reference", ["scalar", "array"])
@pytest.mark.asyncio
async def test_validate_original_cleanup_candidate_rejects_shared_reference(
    repo_with_mocks, reference
):
    repo = repo_with_mocks
    document_id, cycle_id, completed_at, file_id = _cleanup_identity()
    external_document = {"_id": ObjectId()}
    if reference == "array":
        external_document["file_id"] = [file_id]
    else:
        external_document["file_id"] = file_id
    repo.collection.find_one.side_effect = [
        {"_id": ObjectId(document_id)},
        external_document,
    ]

    result = await repo.validate_original_cleanup_candidate(
        document_id,
        cycle_id,
        completed_at,
        file_id,
    )

    assert result is False
    assert repo.collection.find_one.await_count == 2
    repo.collection.update_one.assert_not_awaited()


@pytest.mark.parametrize(
    ("document_id", "cycle_id", "completed_at", "file_id"),
    [
        ("invalid-object-id", "cleanup-cycle", NOW, ObjectId()),
        (str(ObjectId()), "", NOW, ObjectId()),
        (str(ObjectId()), "cleanup-cycle", NOW.replace(tzinfo=None), ObjectId()),
        (str(ObjectId()), "cleanup-cycle", NOW, "not-an-object-id"),
    ],
)
@pytest.mark.asyncio
async def test_validate_original_cleanup_candidate_rejects_invalid_arguments_before_mongo(
    repo_with_mocks, document_id, cycle_id, completed_at, file_id
):
    repo = repo_with_mocks

    with pytest.raises(ValueError):
        await repo.validate_original_cleanup_candidate(
            document_id,
            cycle_id,
            completed_at,
            file_id,
        )

    repo.collection.find_one.assert_not_awaited()
    repo.collection.update_one.assert_not_awaited()


@pytest.mark.asyncio
async def test_validate_original_cleanup_candidate_translates_mongo_unavailability(
    repo_with_mocks,
):
    repo = repo_with_mocks
    repo.collection.find_one.side_effect = AutoReconnect("MongoDB unavailable")
    document_id, cycle_id, completed_at, file_id = _cleanup_identity()

    with pytest.raises(DatabaseUnavailableError):
        await repo.validate_original_cleanup_candidate(
            document_id,
            cycle_id,
            completed_at,
            file_id,
        )

    repo.collection.find_one.assert_awaited_once()
    repo.collection.update_one.assert_not_awaited()


@pytest.mark.asyncio
async def test_validate_original_cleanup_candidate_propagates_cancellation(
    repo_with_mocks,
):
    repo = repo_with_mocks
    cancellation = asyncio.CancelledError()
    repo.collection.find_one.side_effect = cancellation
    document_id, cycle_id, completed_at, file_id = _cleanup_identity()

    with pytest.raises(asyncio.CancelledError) as raised:
        await repo.validate_original_cleanup_candidate(
            document_id,
            cycle_id,
            completed_at,
            file_id,
        )

    assert raised.value is cancellation
    repo.collection.find_one.assert_awaited_once()


@pytest.mark.asyncio
async def test_confirm_original_cleanup_builds_conditional_monotonic_single_write(
    repo_with_mocks,
):
    repo = repo_with_mocks
    document_id, cycle_id, completed_at, file_id = _cleanup_identity()
    normalized_completed_at = completed_at.replace(microsecond=654000)
    confirmed_at = datetime(
        2026, 1, 2, 3, 5, 6, 789456, tzinfo=timezone(timedelta(hours=2))
    )
    normalized_confirmed_at = confirmed_at.astimezone(timezone.utc).replace(
        microsecond=789000
    )
    repo.collection.update_one.return_value.matched_count = 1

    result = await repo.confirm_original_cleanup(
        document_id,
        cycle_id,
        completed_at,
        file_id,
        confirmed_at,
    )

    assert result is processing_repository.CleanupConfirmation.CONFIRMED
    query, update = repo.collection.update_one.call_args.args
    assert query == {
        "_id": ObjectId(document_id),
        "status": "completed",
        "processing_cycle_id": cycle_id,
        "completed_at": normalized_completed_at,
        "file_id": file_id,
        "original_cleanup_status": "pending",
        "$expr": {
            "$and": [
                {"$eq": [{"$type": "$status"}, "string"]},
                {"$eq": [{"$type": "$processing_cycle_id"}, "string"]},
                {"$eq": [{"$type": "$completed_at"}, "date"]},
                {"$eq": [{"$type": "$file_id"}, "objectId"]},
                {"$eq": [{"$type": "$content"}, "string"]},
                {"$in": [{"$type": "$page_count"}, ["int", "long"]]},
                {"$gte": ["$page_count", 0]},
                {
                    "$eq": [
                        {"$type": "$original_cleanup_status"},
                        "string",
                    ]
                },
                {
                    "$in": [
                        {"$type": "$original_deleted_at"},
                        ["missing", "null"],
                    ]
                },
                {
                    "$in": [
                        {"$type": "$updated_at"},
                        ["missing", "null", "date"],
                    ]
                },
            ]
        },
    }
    assert update == [
        {
            "$set": {
                "original_cleanup_status": {"$literal": "completed"},
                "original_deleted_at": {"$literal": normalized_confirmed_at},
                "updated_at": {
                    "$max": [
                        {
                            "$ifNull": [
                                "$updated_at",
                                {"$literal": normalized_confirmed_at},
                            ]
                        },
                        {"$literal": normalized_confirmed_at},
                    ]
                },
            }
        }
    ]
    assert repo.collection.update_one.call_args.kwargs == {"upsert": False}
    repo.collection.update_one.assert_awaited_once()
    repo.collection.find_one.assert_not_awaited()
    repo.collection.update_many.assert_not_awaited()


@pytest.mark.asyncio
async def test_confirm_original_cleanup_recognizes_same_completed_identity_without_rewrite(
    repo_with_mocks,
):
    repo = repo_with_mocks
    document_id, cycle_id, completed_at, file_id = _cleanup_identity()
    normalized_completed_at = completed_at.replace(microsecond=654000)
    repo.collection.update_one.return_value.matched_count = 0
    document_object_id = ObjectId(document_id)
    repo.collection.find_one.return_value = {"_id": document_object_id}

    result = await repo.confirm_original_cleanup(
        document_id,
        cycle_id,
        completed_at,
        file_id,
        NOW,
    )

    assert result is processing_repository.CleanupConfirmation.ALREADY_CONFIRMED
    repo.collection.update_one.assert_awaited_once()
    repo.collection.find_one.assert_awaited_once()
    assert repo.collection.find_one.await_args.args[0] == {
        "_id": document_object_id,
        "status": "completed",
        "processing_cycle_id": cycle_id,
        "completed_at": normalized_completed_at,
        "file_id": file_id,
        "original_cleanup_status": "completed",
        "$expr": {
            "$and": [
                {"$eq": [{"$type": "$status"}, "string"]},
                {"$eq": [{"$type": "$processing_cycle_id"}, "string"]},
                {"$eq": [{"$type": "$completed_at"}, "date"]},
                {"$eq": [{"$type": "$file_id"}, "objectId"]},
                {"$eq": [{"$type": "$content"}, "string"]},
                {"$in": [{"$type": "$page_count"}, ["int", "long"]]},
                {"$gte": ["$page_count", 0]},
                {
                    "$eq": [
                        {"$type": "$original_cleanup_status"},
                        "string",
                    ]
                },
                {"$eq": [{"$type": "$original_deleted_at"}, "date"]},
            ]
        },
    }
    assert repo.collection.find_one.await_args.args[1] == {"_id": 1}


@pytest.mark.asyncio
async def test_confirm_original_cleanup_returns_stale_for_nonmatching_identity(
    repo_with_mocks,
):
    repo = repo_with_mocks
    repo.collection.update_one.return_value.matched_count = 0
    repo.collection.find_one.return_value = None

    result = await repo.confirm_original_cleanup(
        *_cleanup_identity(),
        NOW,
    )

    assert result is processing_repository.CleanupConfirmation.STALE
    repo.collection.update_one.assert_awaited_once()
    repo.collection.find_one.assert_awaited_once()


@pytest.mark.parametrize(
    ("document_id", "cycle_id", "completed_at", "file_id", "confirmed_at"),
    [
        ("invalid-object-id", "cycle", NOW, ObjectId(), NOW),
        (str(ObjectId()), "", NOW, ObjectId(), NOW),
        (str(ObjectId()), "cycle", NOW.replace(tzinfo=None), ObjectId(), NOW),
        (str(ObjectId()), "cycle", NOW, "not-an-object-id", NOW),
        (str(ObjectId()), "cycle", NOW, ObjectId(), NOW.replace(tzinfo=None)),
        (
            str(ObjectId()),
            "cycle",
            NOW,
            ObjectId(),
            NOW.replace(tzinfo=NoOffsetTZ()),
        ),
    ],
)
@pytest.mark.asyncio
async def test_confirm_original_cleanup_rejects_invalid_arguments_before_mongo(
    repo_with_mocks,
    document_id,
    cycle_id,
    completed_at,
    file_id,
    confirmed_at,
):
    repo = repo_with_mocks

    with pytest.raises(ValueError):
        await repo.confirm_original_cleanup(
            document_id,
            cycle_id,
            completed_at,
            file_id,
            confirmed_at,
        )

    repo.collection.find_one.assert_not_awaited()
    repo.collection.update_one.assert_not_awaited()


@pytest.mark.asyncio
async def test_confirm_original_cleanup_translates_reconciliation_connectivity_error(
    repo_with_mocks,
):
    repo = repo_with_mocks
    repo.collection.update_one.return_value.matched_count = 0
    read_error = AutoReconnect("reconciliation unavailable")
    repo.collection.find_one.side_effect = read_error

    with pytest.raises(DatabaseUnavailableError) as raised:
        await repo.confirm_original_cleanup(*_cleanup_identity(), NOW)

    assert raised.value.__cause__ is read_error
    repo.collection.update_one.assert_awaited_once()
    repo.collection.find_one.assert_awaited_once()


@pytest.mark.asyncio
async def test_confirm_original_cleanup_propagates_ambiguous_write_error_for_retry(
    repo_with_mocks,
):
    repo = repo_with_mocks
    write_error = AutoReconnect("write response lost")
    repo.collection.update_one.side_effect = write_error
    identity = _cleanup_identity()

    with pytest.raises(DatabaseUnavailableError) as raised:
        await repo.confirm_original_cleanup(*identity, NOW)

    assert raised.value.__cause__ is write_error
    repo.collection.update_one.assert_awaited_once()
    repo.collection.find_one.assert_not_awaited()


@pytest.mark.parametrize(
    "write_error",
    [
        WriteConcernError("write concern was not confirmed"),
        RuntimeError("unexpected cleanup confirmation error"),
        asyncio.CancelledError(),
    ],
)
@pytest.mark.asyncio
async def test_confirm_original_cleanup_preserves_nonconnectivity_errors_and_cancellation(
    repo_with_mocks, write_error
):
    repo = repo_with_mocks
    repo.collection.update_one.side_effect = write_error
    identity = _cleanup_identity()

    with pytest.raises(type(write_error)) as raised:
        await repo.confirm_original_cleanup(*identity, NOW)

    assert raised.value is write_error
    repo.collection.update_one.assert_awaited_once()
    repo.collection.find_one.assert_not_awaited()


@pytest.mark.asyncio
async def test_confirm_original_cleanup_reconciles_ambiguous_write_on_later_call(
    repo_with_mocks,
):
    repo = repo_with_mocks
    document_id, cycle_id, completed_at, file_id = _cleanup_identity()
    write_error = AutoReconnect("first confirmation response lost")
    repo.collection.update_one.side_effect = [
        write_error,
        MagicMock(matched_count=0),
    ]
    repo.collection.find_one.return_value = {
        "_id": ObjectId(document_id),
        "status": "completed",
        "processing_cycle_id": cycle_id,
        "completed_at": completed_at.replace(microsecond=654000),
        "file_id": file_id,
        "content": "preserved",
        "page_count": 1,
        "original_cleanup_status": "completed",
        "original_deleted_at": NOW,
    }
    arguments = (document_id, cycle_id, completed_at, file_id, NOW)

    with pytest.raises(DatabaseUnavailableError):
        await repo.confirm_original_cleanup(*arguments)

    result = await repo.confirm_original_cleanup(*arguments)

    assert result is processing_repository.CleanupConfirmation.ALREADY_CONFIRMED
    assert repo.collection.update_one.await_count == 2
    repo.collection.find_one.assert_awaited_once()

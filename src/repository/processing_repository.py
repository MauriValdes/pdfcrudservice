from datetime import datetime, timedelta, timezone
from uuid import uuid4

from bson import ObjectId
from pymongo import ASCENDING, ReturnDocument
from pymongo.errors import (
    AutoReconnect,
    ConnectionFailure,
    NetworkTimeout,
    ServerSelectionTimeoutError,
)

from src.config.config import settings
from src.core.exceptions import DatabaseUnavailableError
from src.models.document import DocumentModel


_UNAVAILABLE_ERRORS = (
    ServerSelectionTimeoutError,
    AutoReconnect,
    ConnectionFailure,
    NetworkTimeout,
)

_PRIOR_PROCESSING_METADATA_FIELDS = (
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


def _new_cycle_expression() -> dict:
    """Return the aggregation expression for an uninitialized processing cycle."""
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
            for field in _PRIOR_PROCESSING_METADATA_FIELDS
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


def _existing_cycle_expression(now: datetime) -> dict:
    """Return the aggregation expression for a coherent, started cycle."""
    return {
        "$and": [
            {"$eq": [{"$type": "$processing_cycle_id"}, "string"]},
            {"$ne": ["$processing_cycle_id", ""]},
            {"$eq": [{"$type": "$processing_cycle_started_at"}, "date"]},
            {"$lte": ["$processing_cycle_started_at", now]},
        ]
    }


def _cycle_eligibility_expression(now: datetime) -> dict:
    """Accept either a valid new-cycle candidate or a coherent existing cycle."""
    return {
        "$or": [
            _new_cycle_expression(),
            _existing_cycle_expression(now),
        ]
    }


class ProcessingRepository:
    """Persistencia de las operaciones de propiedad de los trabajos."""

    def __init__(self, db):
        self.db = db
        self.collection = db[settings.mongo_collection]

    async def claim_pending(
        self,
        worker_id: str,
        *,
        lease_duration: timedelta = timedelta(seconds=60),
    ) -> DocumentModel | None:
        now = datetime.now(timezone.utc)
        fencing_token = str(uuid4())
        initial_cycle_id = str(uuid4())

        query = {
            "status": "pending",
            "$expr": {
                "$and": [
                    {"$eq": [{"$type": "$file_id"}, "objectId"]},
                    _cycle_eligibility_expression(now),
                ]
            },
            "$or": [
                {"next_retry_at": {"$exists": False}},
                {"next_retry_at": None},
                {"next_retry_at": {"$lte": now}},
            ],
        }
        claim_fields = {
            "status": "processing",
            "worker_id": {"$literal": worker_id},
            "lease_until": {"$literal": now + lease_duration},
            "heartbeat_at": now,
            "updated_at": now,
            "fencing_token": {"$literal": fencing_token},
        }
        update = [
            {
                "$replaceWith": {
                    "$let": {
                        "vars": {
                            "initialize_cycle": _new_cycle_expression(),
                        },
                        "in": {
                            "$mergeObjects": [
                                "$$ROOT",
                                {
                                    "$cond": [
                                        "$$initialize_cycle",
                                        {
                                            "processing_cycle_id": {
                                                "$literal": initial_cycle_id,
                                            },
                                            "processing_cycle_started_at": {
                                                "$literal": now,
                                            },
                                        },
                                        {},
                                    ]
                                },
                                claim_fields,
                            ]
                        },
                    }
                }
            }
        ]

        try:
            document_data = await self.collection.find_one_and_update(
                query,
                update,
                sort=[("created_at", ASCENDING), ("_id", ASCENDING)],
                return_document=ReturnDocument.AFTER,
                upsert=False,
            )
        except _UNAVAILABLE_ERRORS as error:
            raise DatabaseUnavailableError() from error

        if document_data is None:
            return None

        document_data = dict(document_data)
        document_id = document_data.pop("_id", None)
        if document_id is not None:
            document_data["id"] = str(document_id)
        return DocumentModel.model_validate(document_data)

    async def recover_expired_processing(
        self,
        worker_id: str,
    ) -> DocumentModel | None:
        now = datetime.now(timezone.utc)
        fencing_token = str(uuid4())

        query = {
            "status": "processing",
            "lease_until": {"$lt": now},
            "$or": [
                {"worker_id": {"$exists": False}},
                {"worker_id": None},
                {"worker_id": {"$ne": worker_id}},
            ],
            "$expr": _existing_cycle_expression(now),
        }
        update = [
            {
                "$set": {
                    "worker_id": {"$literal": worker_id},
                    "lease_until": now + timedelta(seconds=60),
                    "heartbeat_at": now,
                    "fencing_token": {"$literal": fencing_token},
                }
            }
        ]

        try:
            document_data = await self.collection.find_one_and_update(
                query,
                update,
                sort=[("lease_until", ASCENDING), ("_id", ASCENDING)],
                return_document=ReturnDocument.AFTER,
                upsert=False,
            )
        except _UNAVAILABLE_ERRORS as error:
            raise DatabaseUnavailableError() from error

        if document_data is None:
            return None

        document_data = dict(document_data)
        document_id = document_data.pop("_id", None)
        if document_id is not None:
            document_data["id"] = str(document_id)
        return DocumentModel.model_validate(document_data)

    async def renew_processing_lease(
        self,
        document_id: str,
        worker_id: str,
        fencing_token: str,
    ) -> bool:
        if not ObjectId.is_valid(document_id):
            raise ValueError("Identificador de documento inválido")

        now = datetime.now(timezone.utc)
        object_id = ObjectId(document_id)
        renewed_lease_until = now + timedelta(seconds=60)
        query = {
            "_id": object_id,
            "status": "processing",
            "worker_id": worker_id,
            "fencing_token": fencing_token,
            "lease_until": {"$gte": now},
        }
        update = [
            {
                "$set": {
                    "heartbeat_at": {
                        "$max": ["$heartbeat_at", {"$literal": now}]
                    },
                    "lease_until": {
                        "$max": [
                            "$lease_until",
                            {"$literal": renewed_lease_until},
                        ]
                    },
                    "updated_at": {
                        "$max": ["$updated_at", {"$literal": now}]
                    },
                }
            }
        ]

        try:
            result = await self.collection.update_one(
                query,
                update,
                upsert=False,
            )
        except _UNAVAILABLE_ERRORS as error:
            raise DatabaseUnavailableError() from error

        return result.matched_count == 1

    async def record_ms2_attempt(
        self,
        document_id: str,
        worker_id: str,
        fencing_token: str,
        processing_cycle_id: str,
    ) -> int | None:
        if not ObjectId.is_valid(document_id):
            raise ValueError("Identificador de documento inválido")

        object_id = ObjectId(document_id)
        captured_now = datetime.now(timezone.utc)
        now = captured_now.replace(
            microsecond=(captured_now.microsecond // 1000) * 1000
        )
        window_start = now - timedelta(minutes=10)

        valid_attempt_count = {
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
        active_cycle = {
            "$and": [
                _existing_cycle_expression(now),
                {
                    "$gt": [
                        "$processing_cycle_started_at",
                        window_start,
                    ]
                },
            ]
        }
        query = {
            "_id": object_id,
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
                    active_cycle,
                    valid_attempt_count,
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
        update = [
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
                                    {"$literal": now},
                                ]
                            },
                            {"$literal": now},
                        ]
                    },
                    "updated_at": {
                        "$max": [
                            {
                                "$ifNull": [
                                    "$updated_at",
                                    {"$literal": now},
                                ]
                            },
                            {"$literal": now},
                        ]
                    },
                }
            }
        ]

        try:
            document_data = await self.collection.find_one_and_update(
                query,
                update,
                return_document=ReturnDocument.AFTER,
                upsert=False,
            )
        except _UNAVAILABLE_ERRORS as error:
            raise DatabaseUnavailableError() from error

        if document_data is None:
            return None

        return int(document_data["attempt_count"])

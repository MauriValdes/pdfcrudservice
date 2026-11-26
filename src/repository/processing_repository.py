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
                "$eq": [
                    {"$type": "$file_id"},
                    "objectId",
                ]
            },
            "$or": [
                {"next_retry_at": {"$exists": False}},
                {"next_retry_at": None},
                {"next_retry_at": {"$lte": now}},
            ],
        }
        update = [
            {
                "$set": {
                    "status": "processing",
                    "worker_id": {"$literal": worker_id},
                    "lease_until": now + lease_duration,
                    "heartbeat_at": now,
                    "updated_at": now,
                    "fencing_token": {"$literal": fencing_token},
                    "processing_cycle_id": {
                        "$ifNull": [
                            "$processing_cycle_id",
                            {"$literal": initial_cycle_id},
                        ]
                    },
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

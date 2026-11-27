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

_FAILURE_PUBLIC_MESSAGES = {
    "ms2_invalid_input": "El archivo PDF no es válido.",
    "pdf_too_large": "El archivo PDF supera el tamaño máximo permitido.",
    "pdf_unprocessable": "No se pudo procesar el archivo PDF.",
}

_EXHAUSTION_PUBLIC_MESSAGES = {
    "attempts_exhausted": "Se agotaron los intentos de procesamiento.",
    "cycle_window_exhausted": "Se agotó el tiempo disponible para el procesamiento.",
}

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


def _processing_terminal_query(
    object_id,
    worker_id: str,
    fencing_token: str,
    processing_cycle_id: str,
    now: datetime,
    operation_guards: list[dict],
) -> dict:
    """Compose common atomic ownership and state guards for terminal writes."""
    return {
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
                _existing_cycle_expression(now),
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
                        {"$type": "$failed_at"},
                        ["missing", "null"],
                    ]
                },
                {
                    "$in": [
                        {"$type": "$next_retry_at"},
                        ["missing", "null", "date"],
                    ]
                },
                *operation_guards,
            ]
        },
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

    async def complete_processing(
        self,
        document_id: str,
        worker_id: str,
        fencing_token: str,
        processing_cycle_id: str,
        attempt_number: int,
        content: str,
        page_count: int,
    ) -> bool:
        if not isinstance(document_id, str) or not ObjectId.is_valid(document_id):
            raise ValueError("Identificador de documento inválido")
        if not isinstance(worker_id, str) or not worker_id:
            raise ValueError("Identificador de worker inválido")
        if not isinstance(fencing_token, str) or not fencing_token:
            raise ValueError("Token de fencing inválido")
        if not isinstance(processing_cycle_id, str) or not processing_cycle_id:
            raise ValueError("Identificador de ciclo inválido")
        if type(attempt_number) is not int or not 1 <= attempt_number <= 5:
            raise ValueError("Número de intento inválido")
        if not isinstance(content, str):
            raise ValueError("El contenido debe ser texto")
        if type(page_count) is not int or not 0 <= page_count <= 2**63 - 1:
            raise ValueError("Número de páginas inválido")

        object_id = ObjectId(document_id)
        captured_now = datetime.now(timezone.utc)
        now = captured_now.replace(
            microsecond=(captured_now.microsecond // 1000) * 1000
        )

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
                ]
            },
        }
        update = [
            {
                "$set": {
                    "content": {"$literal": content},
                    "page_count": {"$literal": page_count},
                    "status": {"$literal": "completed"},
                    "completed_at": {"$literal": now},
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

        try:
            result = await self.collection.update_one(
                query,
                update,
                upsert=False,
            )
        except _UNAVAILABLE_ERRORS as error:
            raise DatabaseUnavailableError() from error

        return result.matched_count == 1

    async def fail_processing(
        self,
        document_id: str,
        worker_id: str,
        fencing_token: str,
        processing_cycle_id: str,
        attempt_number: int,
        failure_code: str,
    ) -> bool:
        """Persist a permanent failure for a previously authorized, concluded attempt."""
        if not isinstance(document_id, str) or not ObjectId.is_valid(document_id):
            raise ValueError("Identificador de documento inválido")
        if not isinstance(worker_id, str) or not worker_id:
            raise ValueError("Identificador de worker inválido")
        if not isinstance(fencing_token, str) or not fencing_token:
            raise ValueError("Token de fencing inválido")
        if not isinstance(processing_cycle_id, str) or not processing_cycle_id:
            raise ValueError("Identificador de ciclo inválido")
        if type(attempt_number) is not int or not 1 <= attempt_number <= 5:
            raise ValueError("Número de intento inválido")
        if not isinstance(failure_code, str):
            raise ValueError("Código de fallo inválido")
        if failure_code not in _FAILURE_PUBLIC_MESSAGES:
            raise ValueError("Código de fallo inválido")

        object_id = ObjectId(document_id)
        captured_now = datetime.now(timezone.utc)
        now = captured_now.replace(
            microsecond=(captured_now.microsecond // 1000) * 1000
        )
        operation_guards = [
            {"$in": [{"$type": "$attempt_count"}, ["int", "long"]]},
            {"$eq": ["$attempt_count", attempt_number]},
            {"$eq": [{"$type": "$last_attempt_at"}, "date"]},
        ]
        query = _processing_terminal_query(
            object_id,
            worker_id,
            fencing_token,
            processing_cycle_id,
            now,
            operation_guards,
        )
        update = [
            {
                "$set": {
                    "status": {"$literal": "failed"},
                    "error_message": {
                        "$literal": _FAILURE_PUBLIC_MESSAGES[failure_code]
                    },
                    "last_error": {"$literal": failure_code},
                    "failed_at": {"$literal": now},
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

        try:
            result = await self.collection.update_one(
                query,
                update,
                upsert=False,
            )
        except _UNAVAILABLE_ERRORS as error:
            raise DatabaseUnavailableError() from error

        return result.matched_count == 1

    async def fail_exhausted_processing(
        self,
        document_id: str,
        worker_id: str,
        fencing_token: str,
        processing_cycle_id: str,
    ) -> bool:
        """Fail an exhausted budget only after the caller rules out pending results.

        The repository cannot observe the state of remote HTTP calls.
        """
        if not isinstance(document_id, str) or not ObjectId.is_valid(document_id):
            raise ValueError("Identificador de documento inválido")
        if not isinstance(worker_id, str) or not worker_id:
            raise ValueError("Identificador de worker inválido")
        if not isinstance(fencing_token, str) or not fencing_token:
            raise ValueError("Token de fencing inválido")
        if not isinstance(processing_cycle_id, str) or not processing_cycle_id:
            raise ValueError("Identificador de ciclo inválido")

        object_id = ObjectId(document_id)
        captured_now = datetime.now(timezone.utc)
        now = captured_now.replace(
            microsecond=(captured_now.microsecond // 1000) * 1000
        )
        attempts_exhausted = {
            "$and": [
                {"$in": [{"$type": "$attempt_count"}, ["int", "long"]]},
                {"$eq": ["$attempt_count", 5]},
            ]
        }
        window_exhausted = {
            "$lte": [
                "$processing_cycle_started_at",
                now - timedelta(minutes=10),
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
                        {
                            "$in": [
                                {"$type": "$attempt_count"},
                                ["int", "long"],
                            ]
                        },
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
                        {
                            "$in": [
                                {"$type": "$attempt_count"},
                                ["int", "long"],
                            ]
                        },
                        {"$gte": ["$attempt_count", 1]},
                        {"$lte": ["$attempt_count", 5]},
                        {"$eq": [{"$type": "$last_attempt_at"}, "date"]},
                    ]
                },
            ]
        }
        operation_guards = [
            valid_attempt_state,
            {"$or": [attempts_exhausted, window_exhausted]},
        ]
        query = _processing_terminal_query(
            object_id,
            worker_id,
            fencing_token,
            processing_cycle_id,
            now,
            operation_guards,
        )
        update = [
            {
                "$set": {
                    "status": {"$literal": "failed"},
                    "error_message": {
                        "$cond": [
                            attempts_exhausted,
                            {
                                "$literal": _EXHAUSTION_PUBLIC_MESSAGES[
                                    "attempts_exhausted"
                                ]
                            },
                            {
                                "$literal": _EXHAUSTION_PUBLIC_MESSAGES[
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
                    "failed_at": {"$literal": now},
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

        try:
            result = await self.collection.update_one(
                query,
                update,
                upsert=False,
            )
        except _UNAVAILABLE_ERRORS as error:
            raise DatabaseUnavailableError() from error

        return result.matched_count == 1

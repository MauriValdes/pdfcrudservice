from datetime import datetime, timedelta, timezone
from enum import Enum
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

_DATETIME_TYPE = datetime


class CleanupConfirmation(Enum):
    CONFIRMED = "confirmed"
    ALREADY_CONFIRMED = "already_confirmed"
    STALE = "stale"


def _validated_cleanup_arguments(
    document_id: str,
    processing_cycle_id: str,
    completed_at: datetime,
    file_id: ObjectId,
    *,
    confirmed_at: datetime | None = None,
    validate_confirmed_at: bool = False,
) -> tuple[ObjectId, str, datetime, ObjectId, datetime | None]:
    """Validate cleanup identity and normalize supplied BSON dates."""
    if not isinstance(document_id, str) or not ObjectId.is_valid(document_id):
        raise ValueError("Identificador de documento inválido")
    if not isinstance(processing_cycle_id, str) or not processing_cycle_id:
        raise ValueError("Identificador de ciclo inválido")
    if not isinstance(file_id, ObjectId):
        raise ValueError("Identificador GridFS inválido")

    def normalize(value: datetime, field_name: str) -> datetime:
        if not isinstance(value, _DATETIME_TYPE):
            raise ValueError(f"{field_name} debe ser datetime")
        try:
            if value.tzinfo is None or value.utcoffset() is None:
                raise ValueError(f"{field_name} debe incluir timezone")
            value = value.astimezone(timezone.utc)
        except (OverflowError, TypeError, ValueError) as error:
            raise ValueError(f"{field_name} debe incluir timezone válido") from error
        return value.replace(microsecond=(value.microsecond // 1000) * 1000)

    normalized_completed_at = normalize(completed_at, "completed_at")
    normalized_confirmed_at = (
        normalize(confirmed_at, "confirmed_at")
        if validate_confirmed_at
        else None
    )
    return (
        ObjectId(document_id),
        processing_cycle_id,
        normalized_completed_at,
        file_id,
        normalized_confirmed_at,
    )


def _cleanup_identity_query(
    document_object_id: ObjectId,
    processing_cycle_id: str,
    completed_at: datetime,
    file_id: ObjectId,
    *,
    marker: str,
    confirmation: bool = False,
) -> dict:
    """Build the exact candidate or reconciliation identity filter."""
    guards = [
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
    ]
    if marker == "pending":
        guards.append(
            {
                "$in": [
                    {"$type": "$original_deleted_at"},
                    ["missing", "null"],
                ]
            }
        )
        if confirmation:
            guards.append(
                {
                    "$in": [
                        {"$type": "$updated_at"},
                        ["missing", "null", "date"],
                    ]
                }
            )
    else:
        guards.append(
            {"$eq": [{"$type": "$original_deleted_at"}, "date"]}
        )

    return {
        "_id": document_object_id,
        "status": "completed",
        "processing_cycle_id": processing_cycle_id,
        "completed_at": completed_at,
        "file_id": file_id,
        "original_cleanup_status": marker,
        "$expr": {"$and": guards},
    }

_TRANSIENT_RETRY_CODES = frozenset(
    {
        "ms2_http_429",
        "ms2_http_500",
        "ms2_http_502",
        "ms2_http_503",
        "ms2_http_504",
        "ms2_timeout",
        "ms2_connection_error",
        "ms2_protocol_error",
        "ms2_invalid_response",
    }
)

_TECHNICAL_FAILURE_PUBLIC_MESSAGE = (
    "No se pudo completar el procesamiento debido a un problema técnico "
    "del servicio."
)

_FAILURE_PUBLIC_MESSAGES = {
    "ms2_invalid_input": "El archivo PDF no es válido.",
    "pdf_too_large": "El archivo PDF supera el tamaño máximo permitido.",
    "pdf_unprocessable": "No se pudo procesar el archivo PDF.",
    "ms2_http_401": _TECHNICAL_FAILURE_PUBLIC_MESSAGE,
    "ms2_http_403": _TECHNICAL_FAILURE_PUBLIC_MESSAGE,
    "ms2_http_404": _TECHNICAL_FAILURE_PUBLIC_MESSAGE,
    "ms2_http_unclassified": _TECHNICAL_FAILURE_PUBLIC_MESSAGE,
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


def _validated_retry_arguments(
    document_id: str,
    worker_id: str,
    fencing_token: str,
    processing_cycle_id: str,
    next_retry_at: datetime,
) -> tuple[ObjectId, datetime, datetime]:
    """Validate retry ownership inputs and normalize the retry instant and clock."""
    if not isinstance(document_id, str) or not ObjectId.is_valid(document_id):
        raise ValueError("Identificador de documento inválido")
    if not isinstance(worker_id, str) or not worker_id:
        raise ValueError("Identificador de worker inválido")
    if not isinstance(fencing_token, str) or not fencing_token:
        raise ValueError("Token de fencing inválido")
    if not isinstance(processing_cycle_id, str) or not processing_cycle_id:
        raise ValueError("Identificador de ciclo inválido")
    if not isinstance(next_retry_at, _DATETIME_TYPE):
        raise ValueError("La fecha del próximo intento debe ser datetime")

    try:
        if next_retry_at.tzinfo is None or next_retry_at.utcoffset() is None:
            raise ValueError("La fecha del próximo intento debe incluir timezone")
        normalized_retry_at = next_retry_at.astimezone(timezone.utc)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(
            "La fecha del próximo intento debe incluir timezone válido"
        ) from error
    normalized_retry_at = normalized_retry_at.replace(
        microsecond=(normalized_retry_at.microsecond // 1000) * 1000
    )

    captured_now = datetime.now(timezone.utc)
    now = captured_now.replace(
        microsecond=(captured_now.microsecond // 1000) * 1000
    )
    if normalized_retry_at <= now:
        raise ValueError("La fecha del próximo intento debe ser futura")

    return ObjectId(document_id), now, normalized_retry_at


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


def _retry_release_common_guards(now: datetime) -> list[dict]:
    """Return cycle-window and heartbeat type guards for retry releases."""
    return [
        {
            "$gt": [
                "$processing_cycle_started_at",
                now - timedelta(minutes=10),
            ]
        },
        {
            "$in": [
                {"$type": "$heartbeat_at"},
                ["missing", "null", "date"],
            ]
        },
    ]


def _retry_release_update(
    now: datetime,
    normalized_next_retry_at: datetime,
    *,
    failure_code: str | None = None,
) -> list[dict]:
    """Build the shared atomic transition that releases processing ownership."""
    set_fields = {
        "status": {"$literal": "pending"},
        "next_retry_at": {
            "$min": [
                {"$literal": normalized_next_retry_at},
                {"$add": ["$processing_cycle_started_at", 600000]},
            ]
        },
        "updated_at": {
            "$max": [
                {"$ifNull": ["$updated_at", {"$literal": now}]},
                {"$literal": now},
            ]
        },
    }
    if failure_code is not None:
        set_fields["last_error"] = {"$literal": failure_code}

    return [
        {"$set": set_fields},
        {
            "$unset": [
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
        },
    ]


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
        """Persist the result and atomically register original cleanup as pending."""
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
        update = [
            {
                "$set": {
                    "content": {"$literal": content},
                    "page_count": {"$literal": page_count},
                    "status": {"$literal": "completed"},
                    "completed_at": {"$literal": now},
                    "original_cleanup_status": {"$literal": "pending"},
                    "original_deleted_at": {"$literal": None},
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

    async def schedule_transient_retry(
        self,
        document_id: str,
        worker_id: str,
        fencing_token: str,
        processing_cycle_id: str,
        attempt_number: int,
        failure_code: str,
        next_retry_at: datetime,
    ) -> bool:
        """Release after an authorized, concluded transient HTTP attempt.

        The caller must ensure that no authorized call can still return an
        acceptable result; this repository cannot observe remote HTTP state.
        """
        if type(attempt_number) is not int or not 1 <= attempt_number <= 5:
            raise ValueError("Número de intento inválido")
        if not isinstance(failure_code, str):
            raise ValueError("Código de fallo inválido")
        if failure_code not in _TRANSIENT_RETRY_CODES:
            raise ValueError("Código de fallo transitorio inválido")

        object_id, now, normalized_retry_at = _validated_retry_arguments(
            document_id,
            worker_id,
            fencing_token,
            processing_cycle_id,
            next_retry_at,
        )
        operation_guards = [
            *_retry_release_common_guards(now),
            {"$in": [{"$type": "$attempt_count"}, ["int", "long"]]},
            {"$eq": ["$attempt_count", attempt_number]},
            {"$lt": ["$attempt_count", 5]},
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
        update = _retry_release_update(
            now,
            normalized_retry_at,
            failure_code=failure_code,
        )

        try:
            result = await self.collection.update_one(
                query,
                update,
                upsert=False,
            )
        except _UNAVAILABLE_ERRORS as error:
            raise DatabaseUnavailableError() from error

        return result.matched_count == 1

    async def release_for_circuit_open(
        self,
        document_id: str,
        worker_id: str,
        fencing_token: str,
        processing_cycle_id: str,
        expected_attempt_count: int,
        next_retry_at: datetime,
    ) -> bool:
        """Release without representing or consuming another HTTP attempt.

        The caller must ensure that no authorized call can still return an
        acceptable result; this repository cannot observe remote HTTP state.
        """
        if (
            type(expected_attempt_count) is not int
            or not 0 <= expected_attempt_count <= 5
        ):
            raise ValueError("Contador esperado de intentos inválido")

        object_id, now, normalized_retry_at = _validated_retry_arguments(
            document_id,
            worker_id,
            fencing_token,
            processing_cycle_id,
            next_retry_at,
        )
        operation_guards = _retry_release_common_guards(now)
        if expected_attempt_count == 0:
            operation_guards.extend(
                [
                    {
                        "$or": [
                            {"$eq": [{"$type": "$attempt_count"}, "missing"]},
                            {
                                "$and": [
                                    {
                                        "$in": [
                                            {"$type": "$attempt_count"},
                                            ["int", "long"],
                                        ]
                                    },
                                    {"$eq": ["$attempt_count", 0]},
                                ]
                            },
                        ]
                    },
                    {
                        "$in": [
                            {"$type": "$last_attempt_at"},
                            ["missing", "null"],
                        ]
                    },
                ]
            )
        else:
            operation_guards.extend(
                [
                    {"$in": [{"$type": "$attempt_count"}, ["int", "long"]]},
                    {"$eq": ["$attempt_count", expected_attempt_count]},
                    {"$lte": ["$attempt_count", 4]},
                    {"$eq": [{"$type": "$last_attempt_at"}, "date"]},
                ]
            )

        query = _processing_terminal_query(
            object_id,
            worker_id,
            fencing_token,
            processing_cycle_id,
            now,
            operation_guards,
        )
        update = _retry_release_update(now, normalized_retry_at)

        try:
            result = await self.collection.update_one(
                query,
                update,
                upsert=False,
            )
        except _UNAVAILABLE_ERRORS as error:
            raise DatabaseUnavailableError() from error

        return result.matched_count == 1

    async def validate_original_cleanup_candidate(
        self,
        document_id: str,
        processing_cycle_id: str,
        completed_at: datetime,
        file_id: ObjectId,
    ) -> bool:
        """Check a pending cleanup candidate and point-in-time references only.

        This read-only check neither authorizes deletion nor closes races with
        concurrent writers; callers must enforce the external safety policy.
        """
        (
            document_object_id,
            normalized_cycle_id,
            normalized_completed_at,
            normalized_file_id,
            _,
        ) = _validated_cleanup_arguments(
            document_id,
            processing_cycle_id,
            completed_at,
            file_id,
        )
        query = _cleanup_identity_query(
            document_object_id,
            normalized_cycle_id,
            normalized_completed_at,
            normalized_file_id,
            marker="pending",
        )
        try:
            candidate = await self.collection.find_one(query, {"_id": 1})
        except _UNAVAILABLE_ERRORS as error:
            raise DatabaseUnavailableError() from error
        if candidate is None:
            return False

        reference_query = {
            "_id": {"$ne": document_object_id},
            "file_id": {"$in": [normalized_file_id]},
        }
        try:
            reference = await self.collection.find_one(reference_query, {"_id": 1})
        except _UNAVAILABLE_ERRORS as error:
            raise DatabaseUnavailableError() from error
        return reference is None

    async def confirm_original_cleanup(
        self,
        document_id: str,
        processing_cycle_id: str,
        completed_at: datetime,
        file_id: ObjectId,
        confirmed_at: datetime,
    ) -> CleanupConfirmation:
        """Conditionally record cleanup confirmation without claiming exactly-once.

        The physical deletion and any required durable audit authorization are
        external preconditions; this method only updates confirmation metadata.
        """
        (
            document_object_id,
            normalized_cycle_id,
            normalized_completed_at,
            normalized_file_id,
            normalized_confirmed_at,
        ) = _validated_cleanup_arguments(
            document_id,
            processing_cycle_id,
            completed_at,
            file_id,
            confirmed_at=confirmed_at,
            validate_confirmed_at=True,
        )
        query = _cleanup_identity_query(
            document_object_id,
            normalized_cycle_id,
            normalized_completed_at,
            normalized_file_id,
            marker="pending",
            confirmation=True,
        )
        update = [
            {
                "$set": {
                    "original_cleanup_status": {
                        "$literal": "completed"
                    },
                    "original_deleted_at": {
                        "$literal": normalized_confirmed_at
                    },
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
        try:
            result = await self.collection.update_one(
                query,
                update,
                upsert=False,
            )
        except _UNAVAILABLE_ERRORS as error:
            raise DatabaseUnavailableError() from error

        if result.matched_count == 1:
            return CleanupConfirmation.CONFIRMED
        if result.matched_count != 0:
            return CleanupConfirmation.STALE

        reconciliation_query = _cleanup_identity_query(
            document_object_id,
            normalized_cycle_id,
            normalized_completed_at,
            normalized_file_id,
            marker="completed",
        )
        try:
            reconciled = await self.collection.find_one(
                reconciliation_query,
                {"_id": 1},
            )
        except _UNAVAILABLE_ERRORS as error:
            raise DatabaseUnavailableError() from error
        if reconciled is not None:
            return CleanupConfirmation.ALREADY_CONFIRMED
        return CleanupConfirmation.STALE

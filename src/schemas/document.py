from datetime import datetime, timezone
from typing import ClassVar, Optional

from pydantic import BaseModel, Field, field_validator

from src.models.document import DocumentStatus


class DocumentCreate(BaseModel):
    """Esquema para crear un documento nuevo.

    Campos obligatorios al subir un PDF: nombre, checksum y tamaño.
    """
    model_config = {"from_attributes": True}

    filename: str
    checksum: str
    size_bytes: int


class DocumentUpdate(BaseModel):
    """Esquema para actualizar un documento (status, content, page_count, error)."""

    model_config = {"from_attributes": True}

    status: Optional[str] = None
    content: Optional[str] = None
    page_count: Optional[int] = None
    error_message: Optional[str] = None


class DocumentResponse(BaseModel):
    """DTO serializable para respuestas HTTP.

    Este DTO es independiente al modelo de persistencia y define
    exactamente qué campos se exponen en la API.
    """
    model_config = {"from_attributes": True}

    id: str
    filename: str
    checksum: str
    size_bytes: int
    status: str
    content: Optional[str] = None
    page_count: Optional[int] = None
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    PUBLIC_FAILURE_MESSAGES: ClassVar[frozenset[str]] = frozenset(
        {
            "El archivo PDF no es válido.",
            "El archivo PDF supera el tamaño máximo permitido.",
            "No se pudo procesar el archivo PDF.",
            "Se agotaron los intentos de procesamiento.",
            "Se agotó el tiempo disponible para el procesamiento.",
        }
    )
    GENERIC_PUBLIC_FAILURE: ClassVar[str] = "No se pudo procesar el documento PDF."

    @classmethod
    def for_public_read(cls, **data):
        """Construye la proyección GET sin alterar DTOs de flujos internos."""
        status_value = data.get("status")
        if status_value in {"uploading", "pending", "processing"}:
            data.update(content=None, page_count=None, error_message=None)
        elif status_value == "completed":
            data["error_message"] = None
        elif status_value == "failed":
            data["content"] = None
            data["page_count"] = None
            if data.get("error_message") not in cls.PUBLIC_FAILURE_MESSAGES:
                data["error_message"] = cls.GENERIC_PUBLIC_FAILURE
        else:
            data.update(content=None, page_count=None, error_message=None)

        for field_name in ("created_at", "updated_at"):
            value = data.get(field_name)
            if isinstance(value, datetime):
                if value.tzinfo is None or value.utcoffset() is None:
                    data[field_name] = value.replace(tzinfo=timezone.utc)
                else:
                    data[field_name] = value.astimezone(timezone.utc)

        return cls(**data)


class DocumentReceipt(BaseModel):
    """Ticket público devuelto al recibir un PDF."""

    model_config = {"extra": "forbid"}

    document_id: str
    status: DocumentStatus
    created_at: datetime
    status_url: str

    @field_validator("created_at", mode="before")
    @classmethod
    def normalize_created_at_to_utc(cls, value):
        """Las fechas BSON naive se interpretan como UTC."""
        if isinstance(value, datetime):
            if value.tzinfo is None or value.utcoffset() is None:
                return value.replace(tzinfo=timezone.utc)
            return value.astimezone(timezone.utc)
        return value

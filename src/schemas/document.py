from datetime import datetime, timezone
from typing import Optional

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

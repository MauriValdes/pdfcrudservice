from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


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
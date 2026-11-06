from enum import Enum
from typing import Optional

from datetime import datetime

from pydantic import BaseModel, Field, ConfigDict


class DocumentStatus(str, Enum):
    """Estado del documento con representación string para MongoDB y APIs."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class DocumentModel(BaseModel):
    """Modelo de persistencia para MongoDB con alias de campo y configuración ORM."""

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    id: Optional[str] = Field(default=None, alias="_id")
    filename: str
    checksum: str
    size_bytes: int
    status: DocumentStatus = DocumentStatus.PENDING
    content: Optional[str] = None
    page_count: Optional[int] = None
    error_message: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
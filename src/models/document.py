from enum import Enum
from typing import Optional

from datetime import datetime

from bson import ObjectId
from pydantic import BaseModel, Field, ConfigDict


class DocumentStatus(str, Enum):
    """Estado del documento con representación string para MongoDB y APIs."""
    UPLOADING = "uploading"
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class DocumentModel(BaseModel):
    """Modelo de persistencia para MongoDB con alias de campo y configuración ORM."""

    model_config = ConfigDict(
        from_attributes=True,
        populate_by_name=True,
        arbitrary_types_allowed=True,
    )

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

    # Campos internos para reservas de carga en curso (no se exponen en la API).
    file_id: Optional[ObjectId] = None
    upload_owner_id: Optional[str] = None
    upload_lease_until: Optional[datetime] = None
    upload_fencing_token: Optional[str] = None
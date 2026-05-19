from __future__ import annotations

from datetime import datetime
from enum import Enum
from uuid import UUID

from pydantic import BaseModel, Field


# ── Enums ────────────────────────────────────────────────────────────────────


class ProcessingStatus(str, Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    INDEXED = "INDEXED"
    FAILED = "FAILED"


# ── Document schemas ────────────────────────────────────────────────────────


class DocumentBase(BaseModel):
    filename: str
    content_type: str


class DocumentCreate(DocumentBase):
    """Payload for creating a document record."""


class DocumentRead(DocumentBase):
    id: UUID
    status: ProcessingStatus
    filepath: str
    filesizebytes: int
    metadata_: dict = Field(alias="metadata", default_factory=dict)
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True, "populate_by_name": True}


class DocumentUpdate(BaseModel):
    """Payload for partially updating a document."""

    status: ProcessingStatus | None = None
    metadata_: dict | None = Field(alias="metadata", default=None)

    model_config = {"populate_by_name": True}


# ── Chunk schemas ───────────────────────────────────────────────────────────


class ChunkBase(BaseModel):
    document_id: UUID
    content_type: str


class ChunkCreate(ChunkBase):
    """Payload for creating a chunk record."""


class ChunkRead(ChunkBase):
    id: UUID
    status: ProcessingStatus
    filepath: str
    filesizebytes: int
    metadata_: dict = Field(alias="metadata", default_factory=dict)
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True, "populate_by_name": True}


class ChunkUpdate(BaseModel):
    """Payload for partially updating a chunk."""

    status: ProcessingStatus | None = None
    metadata_: dict | None = Field(alias="metadata", default=None)

    model_config = {"populate_by_name": True}
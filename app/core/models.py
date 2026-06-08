from __future__ import annotations

import enum
from datetime import datetime
from uuid import uuid4

from sqlalchemy import JSON, DateTime, ForeignKey, Integer, String, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# ── Base ─────────────────────────────────────────────────────────────────────


class Base(DeclarativeBase):
    """SQLAlchemy 2.0 declarative base."""


# ── Mixins ───────────────────────────────────────────────────────────────────


class TimestampMixin:
    """Provides created_at / updated_at columns with DB-side defaults."""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


# ── Enums ────────────────────────────────────────────────────────────────────


class ProcessingStatus(enum.Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    INDEXED = "INDEXED"
    FAILED = "FAILED"


class Rating(enum.Enum):
    THUMBS_UP = "THUMBS_UP"
    THUMBS_DOWN = "THUMBS_DOWN"


# ── Models ────────────────────────────────────────────────────────────────────


class Document(TimestampMixin, Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    filename: Mapped[str] = mapped_column(String, nullable=False)
    content_type: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[ProcessingStatus] = mapped_column(
        String(16),
        nullable=False,
        default=ProcessingStatus.PENDING,
    )
    filepath: Mapped[str] = mapped_column(String, nullable=False)
    filesizebytes: Mapped[int] = mapped_column(Integer, nullable=False)
    metadata_: Mapped[dict] = mapped_column(
        "metadata", JSON, nullable=False, default=dict
    )

    # Relationships
    chunks: Mapped[list[Chunk]] = relationship(
        back_populates="document", cascade="all, delete-orphan", lazy="selectin"
    )

    def __repr__(self) -> str:
        return f"<Document {self.id} status={self.status.value}>"


class Chunk(TimestampMixin, Base):
    __tablename__ = "chunks"

    id: Mapped[str] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    document_id: Mapped[str] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    content_type: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[ProcessingStatus] = mapped_column(
        String(16),
        nullable=False,
        default=ProcessingStatus.PENDING,
    )
    filepath: Mapped[str] = mapped_column(String, nullable=False)
    filesizebytes: Mapped[int] = mapped_column(Integer, nullable=False)
    metadata_: Mapped[dict] = mapped_column(
        "metadata", JSON, nullable=False, default=dict
    )

    # Relationships
    document: Mapped[Document] = relationship(back_populates="chunks")

    def __repr__(self) -> str:
        return f"<Chunk {self.id} doc={self.document_id}>"


class QueryLogs(TimestampMixin, Base):
    __tablename__ = "query_logs"

    id: Mapped[str] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    query_text: Mapped[str] = mapped_column(String, nullable=False)
    response_text: Mapped[str] = mapped_column(String, nullable=False)
    retrieved_chunk_ids: Mapped[list[dict]] = mapped_column(
        JSON, nullable=False, default=list
    )
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    model_used: Mapped[str] = mapped_column(String, nullable=False)

    feedback: Mapped[Feedback | None] = relationship(
        back_populates="query_log",  # ← back_populates, not backref
        cascade="all, delete-orphan",
        lazy="selectin",
        uselist=False,
    )

    def __repr__(self) -> str:
        return f"<QueryLog {self.id}>"


class Feedback(TimestampMixin, Base):
    __tablename__ = "feedback"

    id: Mapped[str] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    query_log_id: Mapped[str] = mapped_column(  # ← FK column (was missing)
        UUID(as_uuid=True),
        ForeignKey("query_logs.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,  # ← enforces 1:1 at DB level
        index=True,
    )
    rating: Mapped[Rating] = mapped_column(
        String(16),
        nullable=False,
        default=Rating.THUMBS_UP,
    )
    comment: Mapped[str] = mapped_column(String, nullable=False)
    relevant_chunk_ids: Mapped[list[str]] = mapped_column(
        JSON, nullable=False, default=list
    )

    query_log: Mapped[QueryLogs] = relationship(QueryLogs, back_populates="feedback")

    def __repr__(self) -> str:
        return f"<Feedback {self.id}>"

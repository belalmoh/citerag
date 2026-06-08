from uuid import UUID

from fastapi import APIRouter, HTTPException
from fastapi.params import Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.models import Document

router = APIRouter()


@router.get("/{document_id}")
async def get_job(document_id: UUID, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Document).where(Document.id == document_id))
    document: Document | None = result.scalar_one_or_none()

    if not document or document is None:
        raise HTTPException(status_code=404, detail="Document not found")

    return {
        "document_id": str(document_id),
        "status": document.status.value,
        "filename": document.filename,
    }

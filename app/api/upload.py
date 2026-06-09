import asyncio
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.database import get_db
from app.core.schemas import DocumentRead
from app.core.models import Document, ProcessingStatus
from app.ingestion.worker import process_document

router = APIRouter()

@router.post("/", response_model=DocumentRead, status_code=201)
async def upload_document(file: UploadFile = File(...), db: AsyncSession = Depends(get_db)):
    settings = get_settings()
    content = await file.read()

    max_bytes = settings.max_file_size_mb * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds {settings.max_file_size_mb} MB limit",
        )

    upload_dir = Path("uploads")
    upload_dir.mkdir(exist_ok=True)
    file_path = upload_dir / f"{uuid4()}_{file.filename}"

    await asyncio.to_thread(file_path.write_text, str(content))

    doc = Document(
        filename=file.filename or "unknown",
        content_type=file.content_type or "application/octet-stream",
        filepath=str(file_path),
        filesizebytes=file.size or 0,
        status=ProcessingStatus.PENDING.value,
    )

    db.add(doc)
    await db.commit()
    await db.refresh(doc)

    process_document.delay(str(doc.id), str(file_path))

    return doc

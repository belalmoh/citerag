from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.models import QueryLogs, Rating, Feedback

router = APIRouter()

class FeedbackRequest(BaseModel):
    query_log_id: UUID
    rating: str
    comment: str | None = None
    relevant_chunk_ids: list[str] = []

@router.post("/", status_code=201)
async def submit_feedback(body: FeedbackRequest, db: AsyncSession = Depends(get_db)):
    log = await db.get(QueryLogs, body.query_log_id)
    
    if not log:
        raise HTTPException(status_code=404, detail="Query log not found")

    try:
        rating = Rating(body.rating)
    except:
        raise HTTPException(status_code=400, detail="Invalid rating")
    
    feedback = Feedback(
        query_log_id=body.query_log_id,
        rating=rating,
        comment=body.comment,
        relevant_chunk_ids=body.relevant_chunk_ids,
    )

    db.add(feedback)
    await db.commit()
    await db.refresh(feedback)

    return {"status": "ok"}
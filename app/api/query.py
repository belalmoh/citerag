import time

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.ingestion.embedder import Embedder
from app.retrieval.retriever import Retriever

router = APIRouter()

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

class QueryResponse(BaseModel):
    response: str
    sources: list[dict]
    query_log_id: str

@router.post("/", response_model=QueryResponse)
async def query_documents(
        body: QueryRequest,
        db: AsyncSession = Depends(get_db),
        embedder: Embedder = Depends(Embedder)
):
    start = time.monotonic()



    return {"status": "pending", "message": "Query endpoint stub"}

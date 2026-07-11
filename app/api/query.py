import time

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.dependencies import get_embedder, get_indexer
from app.core.models import QueryLogs
from app.generation.llm_client import LLMClient
from app.generation.prompt_builder import PromptBuilder
from app.generation.stream_handler import generate_rag_stream
from app.ingestion.embedder import Embedder
from app.retrieval.hybrid_search import HybridRetriever
from app.retrieval.reranker import Reranker
from app.retrieval.retriever import Retriever
from app.ingestion.indexer import Indexer

router = APIRouter()

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

class SourceItem(BaseModel):
    chunk_id: str
    filename: str
    score: float
    snippet: str

class QueryResponse(BaseModel):
    response: str
    sources: list[SourceItem]
    query_log_id: str
    model_used: str
    prompt_tokens: int
    completion_tokens: int

@router.post("/", response_model=QueryResponse)
async def query_documents(
        body: QueryRequest,
        db: AsyncSession = Depends(get_db),
        embedder: Embedder = Depends(get_embedder),
        indexer: Indexer = Depends(get_indexer)
):
    start = time.monotonic()

    # 1. Embedding the query
    [query_vec] = await embedder.embed([body.query])
    
    # 2. Retrieving from Qdrant
    hybrid = HybridRetriever(indexer)

    # 3. Rerank
    chunks = await hybrid.search(query_vector=query_vec, query_text=body.query, top_k=body.top_k)
    reranker = Reranker()
    chunks = await reranker.rerank(query=body.query, chunks=chunks, top_k=body.top_k)

    # 4. Building the prompt for the LLM
    prompt = PromptBuilder().build(question=body.query, chunks=chunks)

    # 5. Querying the LLM
    llm = LLMClient()
    result = await llm.generate(user_message=prompt.user, system_prompt=prompt.system)

    latency = int((time.monotonic() - start) * 1000)

    # 6. Log to DB
    log = QueryLogs(
        query_text=body.query,
        response_text=result.text,
        model_used=result.model,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        latency_ms=latency
    )
    db.add(log)
    await db.commit()
    await db.refresh(log)

    return QueryResponse(
        response=result.text,
        sources=[
            SourceItem(
                chunk_id=chunk.chunk_id,
                filename=chunk.filename,
                score=chunk.score,
                snippet=chunk.content[:200]  # First 200 chars
            )
            for chunk in chunks
        ],
        query_log_id=str(log.id),
        model_used=result.model,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens
    )

@router.post("/stream")
async def stream_query(
    body: QueryRequest, 
    db: AsyncSession = Depends(get_db), 
    embedder: Embedder = Depends(get_embedder), 
    indexer: Indexer = Depends(get_indexer)
):
    [query_vec] = await embedder.embed([body.query])
    hybrid = HybridRetriever(indexer)
    chunks = await hybrid.search(query_vector=query_vec, query_text=body.query, top_k=body.top_k)
    prompt = PromptBuilder().build(question=body.query, chunks=chunks)
    llm = LLMClient()
    return StreamingResponse(
        generate_rag_stream(llm=llm, prompt=prompt, chunks=chunks, query=body.query, db=db),
    )
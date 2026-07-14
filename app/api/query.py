import asyncio
import time
import logging

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
from app.retrieval.retriever import Retriever
from app.retrieval.reranker import Reranker
from app.retrieval.query_expander import QueryExpander
from app.ingestion.indexer import Indexer
from app.retrieval.retriever import SearchResult

router = APIRouter()
logger = logging.getLogger(__name__)

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

def _merge_search_results(
    results_per_query: list[list[SearchResult]],
    top_k: int,
) -> list[SearchResult]:
    """Merge multiple search result lists, deduplicating by chunk_id.

    When the same chunk appears from different expanded queries, the
    highest score wins. Results are sorted descending by score.
    """
    seen: dict[str, SearchResult] = {}
    for results in results_per_query:
        for r in results:
            if r.chunk_id not in seen or r.score > seen[r.chunk_id].score:
                seen[r.chunk_id] = r
    merged = sorted(seen.values(), key=lambda x: x.score, reverse=True)
    return merged[:top_k]

@router.post("/", response_model=QueryResponse)
async def query_documents(
        body: QueryRequest,
        db: AsyncSession = Depends(get_db),
        embedder: Embedder = Depends(get_embedder),
        indexer: Indexer = Depends(get_indexer)
):
    start = time.monotonic()
    llm = LLMClient()

    # 1. Embedding the query
    [query_vec] = await embedder.embed([body.query])
    
    # 2. Query expansion
    expander = QueryExpander(llm)
    expanded_queries = await expander.expand_query(body.query)
    logger.info("Expanded query '%s' → %d variations", body.query[:50], len(expanded_queries))

    # 3. Embed all variations in one batch all
    all_vectors = await embedder.embed(expanded_queries)

    logger.info("zipped %s", list(zip(all_vectors, expanded_queries)))

    # 4. Retrieving from Qdrant
    retriever = Retriever(indexer)
    all_results = await asyncio.gather(*[
        retriever.search(
            query_vector=vec,
            top_k=body.top_k * 2,
        )
        for vec, q in zip(all_vectors, expanded_queries)
    ])

    # 5. Rerank
    chunks = _merge_search_results(all_results, top_k=body.top_k * 2)
    reranker = Reranker()
    chunks = await reranker.rerank(query=body.query, chunks=chunks, top_k=body.top_k)

    # 6. Building the prompt for the LLM
    prompt = PromptBuilder().build(question=body.query, chunks=chunks)

    # 7. Querying the LLM
    
    result = await llm.generate(user_message=prompt.user, system_prompt=prompt.system)

    latency = int((time.monotonic() - start) * 1000)

    # 8. Log to DB
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

    expander = QueryExpander()
    expanded_queries = await expander.expand_query(body.query)
    all_vectors = await embedder.embed(expanded_queries)

    retriever = Retriever(indexer)
    all_results = await asyncio.gather(*[
        retriever.search(query_vector=vec, top_k=body.top_k * 2)
        for vec, q in zip(all_vectors, expanded_queries)
    ])

    chunks = _merge_search_results(all_results, top_k=body.top_k * 2)
    prompt = PromptBuilder().build(question=body.query, chunks=chunks)
    llm = LLMClient()
    return StreamingResponse(
        generate_rag_stream(llm=llm, prompt=prompt, chunks=chunks, query=body.query, db=db),
    )
import json
import time
import logging
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.models import QueryLogs
from app.generation.llm_client import LLMClient
from app.generation.prompt_builder import Prompt
from app.retrieval.retriever import SearchResult

logger = logging.getLogger(__name__)

def _sse(data: dict) -> str:
    """Convert data to Server-Sent Events format."""
    return f"data: {json.dumps(data)}\n\n"

async def generate_rag_stream(query: str, chunks: list[SearchResult], prompt: Prompt, llm: LLMClient, db: AsyncSession):
    
    time_start = time.monotonic()

    # 1. Send start event with sources
    yield _sse({
        "event": "start",
        "sources": [
            {"chunk_id": c.chunk_id, "filename": c.filename, "score": c.score, "snippet": c.content[:200]}
            for c in chunks
        ]
    })

    # 2. Call OpenAI with streaming and yield tokens as they arrive
    buffer: list[str] = []
    prompt_tokens = 0
    completion_tokens = 0
    model_used = llm.model

    async for chunk in llm.generate_stream(prompt.user, prompt.system):
        if "token" in chunk:
            buffer.append(chunk["token"])
            yield _sse({"type": "token", "content": chunk["token"]})
        elif "usage" in chunk:
            prompt_tokens = chunk["usage"].get("prompt_tokens", 0)
            completion_tokens = chunk["usage"].get("completion_tokens", 0) 
            model_used = chunk["usage"].get("model", llm.model)

    # 3. Log to DB
    response_text = "".join(buffer)
    latency = int((time.monotonic() - time_start) * 1000)
    log = QueryLogs(
        query_text=query,
        response_text=response_text,
        retrieved_chunk_ids=[
            {"chunk_id": c.chunk_id, "score": c.score, "filename": c.filename}
            for c in chunks
        ],
        latency_ms=latency,
        model_used=model_used,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )

    db.add(log)
    await db.commit()
    await db.refresh(log)

    # 4. Send done event
    yield _sse({
        "type": "done",
        "query_log_id": str(log.id),
        "model_used": model_used,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    })
    
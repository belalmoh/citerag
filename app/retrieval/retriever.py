import logging

from pydantic import BaseModel

from app.ingestion.indexer import Indexer

logger = logging.getLogger(__name__)

class SearchResult(BaseModel):
    chunk_id: str
    content: str
    score: float
    document_id: str
    filename: str
    chunk_index: int

class Retriever:
    def __init__(self, indexer: Indexer):
        self._indexer = indexer

    async def search(self, query_vector: list[float], top_k: int, score_threshold: float):
        from qdrant_client import models

        response = await self._indexer._client.query_points(
            collection_name=self._indexer._collection_name,
            query=query_vector,  # ← was query_vector
            limit=top_k,
            score_threshold=score_threshold,
            with_payload=True,
            search_params=models.SearchParams(
                hnsw_ef=128,
                exact=False,
            ),
        )

        results = response.points

        if not results:
            return []

        return [
            SearchResult(
                chunk_id=str(result.id),
                content=result.payload["content"],
                score=result.score,
                document_id=result.payload["document_id"],
                filename=result.payload["filename"],
                chunk_index=result.payload["chunk_index"],
            ) for result in results
        ]
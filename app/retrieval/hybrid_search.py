from __future__ import annotations

from app.ingestion.indexer import Indexer
from app.retrieval.retriever import SearchResult

class HybridRetriever:
    """A retriever that combines vector search and keyword search."""
    
    def __init__(self, indexer: Indexer):
        self.indexer = indexer

    async def search(
        self, 
        query_vector: list[float], 
        query_text: str, 
        top_k: int = 5, 
        vector_limit: int = 20, 
        keyword_limit: int = 20
    ) -> list[SearchResult]:
        from qdrant_client import models

        response = await self.indexer._client.query_points(
            collection_name=self.indexer._collection_name,
            prefetch=[
                models.Prefetch(query=query_vector, limit=vector_limit),
                models.Prefetch(query=query_text, limit=keyword_limit),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=top_k,
            with_payload=True,
        )

        results = response.points
        if not results:
            return []
        
        return [
            SearchResult(
                chunk_id=result.id,
                content=result.payload.get("content", ""),
                score=result.score,
                document_id=result.payload.get("document_id", ""),
                filename=result.payload.get("filename", ""),
                chunk_index=result.payload.get("chunk_index", 0),
            ) for result in results
        ]
import asyncio

from app.retrieval.retriever import SearchResult


class Reranker:
    def __init__(self, model_name):
        from sentence_transformers import CrossEncoder
        self._model = CrossEncoder(model_name)

    async def rerank(self, query: str, chunks: list[SearchResult], top_k: int = 5) -> list[SearchResult]:
        if not chunks:
            return []
        
        pairs = [(query, c.content) for c in chunks]
        
        scores = await asyncio.to_thread(self._model.predict, pairs)

        scored = list(zip(chunks, scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        return [
            SearchResult(
                chunk_id=c.chunk_id,
                content=c.content,
                score=s,
                document_id=c.document_id,
                filename=c.filename,
                chunk_index=c.chunk_index,
            )
            for c, s in scored[:top_k]
        ]

from __future__ import annotations

import logging
from uuid import UUID, uuid5

from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from app.ingestion.chunker import Chunk

logger = logging.getLogger(__name__)

Distance = models.Distance


class Indexer:
    def __init__(self, client: AsyncQdrantClient, collection_name: str):
        self._client = client
        self._collection_name = collection_name

    async def ensure_collection(self, dimension: int, distance: str = "Cosine"):
        """Create collection if it doesn't exist."""
        try:
            await self._client.get_collection(self._collection_name)
        except UnexpectedResponse:
            await self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config=models.VectorParams(
                    size=dimension,
                    distance=Distance(distance),
                ),
            )
            await self._client.create_payload_index(
                collection_name=self._collection_name,
                field_name="content",
                field_schema=models.PayloadSchemaType.TEXT)
            logger.info(
                "Created Qdrant collection %s (dim=%d, distance=%s)",
                self._collection_name,
                dimension,
                distance,
            )

    async def upsert_chunks(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> list[str]:
        """Upsert chunks to Qdrant. Returns list of vector IDs."""
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings length mismatch")
        if not chunks:
            return []
        vector_ids = [
            str(uuid5(UUID(chunk.metadata["document_id"]), str(chunk.index)))
            for chunk in chunks
        ]
        await self._client.upsert(
            collection_name=self._collection_name,
            points=[
                models.PointStruct(
                    id=vector_id,
                    vector=list(vec),
                    payload={
                        "content": chunk.content,
                        "document_id": chunk.metadata["document_id"],
                        "chunk_index": chunk.index,
                        "filename": chunk.metadata.get("filename", ""),
                    },
                )
                for vector_id, chunk, vec in zip(vector_ids, chunks, embeddings)
            ],
        )
        return vector_ids

    async def delete_document_chunks(self, document_id: str):
        """Delete all chunks for a given document."""
        await self._client.delete(
            collection_name=self._collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id",
                            match=models.MatchValue(value=document_id),
                        )
                    ]
                )
            ),
        )

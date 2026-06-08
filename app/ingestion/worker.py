from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from celery import Celery

from app.core.config import get_settings
from app.ingestion.chunker import Chunker
from app.ingestion.embedder import Embedder, OpenAIEmbedder
from app.ingestion.indexer import Indexer
from app.ingestion.parser import DocumentParser

logger = logging.getLogger(__name__)

celery_app = Celery("citerag")


class IngestionPipeline:
    """Orchestrates the full async pipeline: parse → chunk → embed → index."""

    def __init__(
        self,
        embedder: Embedder,
        indexer: Indexer,
        chunker: Chunker | None = None,
    ) -> None:
        self._embedder = embedder
        self._indexer = indexer
        self._chunker = chunker or Chunker()

    async def run(self, document_id: str, file_path: str) -> list[str]:
        """Execute the full ingestion pipeline for a single document.

        Args:
            document_id: The DB ``Document.id`` (UUID as string) — used
                everywhere so Postgres and Qdrant agree on identity.
            file_path: Absolute path to the uploaded file on disk.

        Returns the vector IDs of the indexed chunks.
        """
        # 1. Parse (offloaded to thread — pymupdf blocks the event loop)
        logger.info("Ingestion started: doc=%s file=%s", document_id, file_path)
        parsed = await DocumentParser(file_path).parse_async()

        # 2. Chunk (sync, pure CPU — cheap enough to run inline)
        metadata: dict = {
            "document_id": document_id,
            "filename": Path(file_path).name,
            "source": parsed.source,
        }
        chunks = self._chunker.split(parsed.text, metadata)
        if not chunks:
            logger.warning("No chunks produced for %s", file_path)
            return []
        logger.info("Chunked %s → %d chunks", file_path, len(chunks))

        # 3. Embed (async, network I/O to OpenAI)
        chunk_texts = [c.content for c in chunks]
        embeddings = await self._embedder.embed(chunk_texts)
        logger.info("Embedded %d chunks for %s", len(embeddings), file_path)

        # 4. Index (async, network I/O to Qdrant)
        await self._indexer.ensure_collection(self._embedder.dimension)
        vector_ids = await self._indexer.upsert_chunks(chunks, embeddings)
        logger.info("Indexed %d vectors for %s", len(vector_ids), file_path)

        return vector_ids


# ── Celery task ───────────────────────────────────────────────────────────


@celery_app.task(bind=True, max_retries=3, default_retry_delay=5)
def process_document(self, document_id: str, file_path: str) -> dict | None:
    """Celery task: parse → chunk → embed → index → update status.

    The sync→async boundary is here — ``asyncio.run()`` bridges
    Celery's sync worker to our async pipeline components.
    """
    settings = get_settings()

    async def _pipeline() -> list[str]:
        embedder = OpenAIEmbedder(model=settings.openai_embedding_model)
        from qdrant_client import AsyncQdrantClient

        client = AsyncQdrantClient(url=settings.qdrant_url)
        indexer = Indexer(client, settings.qdrant_collection_name)
        pipeline = IngestionPipeline(embedder, indexer)
        return await pipeline.run(document_id, file_path)

    async def _mark_status(status: str) -> None:
        """Update the Document row to INDEXED or FAILED."""
        from sqlalchemy import update
        from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

        engine = create_async_engine(settings.database_url)
        maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        async with maker() as session:
            from app.core.models import Document, ProcessingStatus
            await session.execute(
                update(Document)
                .where(Document.id == document_id)
                .values(status=ProcessingStatus(status))
            )
            await session.commit()

    try:
        vector_ids = asyncio.run(_pipeline())
        asyncio.run(_mark_status("INDEXED"))
    except Exception as exc:
        logger.error(
            "Ingestion failed for doc=%s file=%s (attempt %d/%d): %s",
            document_id, file_path, self.request.retries + 1, MAX_RETRIES, exc,
        )
        asyncio.run(_mark_status("FAILED"))
        raise self.retry(exc=exc)

    return {
        "document_id": document_id,
        "chunks_indexed": len(vector_ids),
    }

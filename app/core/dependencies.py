from __future__ import annotations

import functools

from qdrant_client import AsyncQdrantClient

from app.core.config import get_settings
from app.ingestion.embedder import OpenAIEmbedder, Embedder
from app.ingestion.indexer import Indexer


@functools.lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    settings = get_settings()
    return OpenAIEmbedder(
        model=settings.openai_embedding_model,
        base_url=settings.openai_base_url,
        api_key=settings.openai_api_key,
    )

@functools.lru_cache(maxsize=1)
def get_qdrant_client() -> AsyncQdrantClient:
    settings = get_settings()
    return AsyncQdrantClient(url=settings.qdrant_url)

def get_indexer() -> Indexer:
    settings = get_settings()
    return Indexer(get_qdrant_client(), settings.qdrant_collection_name)
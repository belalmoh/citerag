from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

# ── Known model dimensions ──────────────────────────────────────────────────

MODEL_DIMENSIONS: dict[str, int] = {
    "text-embedding-3-large": 3072,
    "text-embedding-3-small": 1536,
    "all-MiniLM-L6-v2": 384,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
}

BATCH_SIZE = 100
MAX_RETRIES = 3
INITIAL_BACKOFF = 1.0  # seconds


# ── Abstract base ─────────────────────────────────────────────────────────────


class Embedder(ABC):
    """Convert a list of text strings into a list of embedding vectors."""

    @abstractmethod
    async def embed(self, text: list[str]) -> list[list[float]]:
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension for the configured model."""
        ...


# ── OpenAI embedder ──────────────────────────────────────────────────────────


class OpenAIEmbedder(Embedder):
    """OpenAI embedding client with manual batching and retry logic."""

    def __init__(self, model: str = "text-embedding-3-large") -> None:
        from openai import AsyncOpenAI

        self.model = model
        self._dimension = MODEL_DIMENSIONS.get(model, 0)
        self._client = AsyncOpenAI()

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed(self, text: list[str]) -> list[list[float]]:
        """Embed with manual batching and exponential-backoff retry."""
        if not text:
            return []
        embeddings: list[list[float]] = []
        for batch_start in range(0, len(text), BATCH_SIZE):
            batch = text[batch_start : batch_start + BATCH_SIZE]
            embeddings.extend(await self._embed_batch_with_retry(batch))
        return embeddings

    async def _embed_batch_with_retry(self, batch: list[str]) -> list[list[float]]:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = await self._client.embeddings.create(input=batch, model=self.model)
                if not self._dimension:
                    self._dimension = len(response.data[0].embedding)
                return [item.embedding for item in response.data]
            except Exception as exc:
                if attempt == MAX_RETRIES:
                    logger.error("OpenAI embed failed after %d attempts: %s", MAX_RETRIES, exc)
                    raise
                backoff = INITIAL_BACKOFF * (2 ** (attempt - 1))
                logger.warning("OpenAI embed attempt %d failed, retrying in %.1fs: %s", attempt, backoff, exc)
                await asyncio.sleep(backoff)
        return []  # unreachable


# ── Local embedder ───────────────────────────────────────────────────────────


class LocalEmbedder(Embedder):
    """Sentence-transformers embedder for cost-sensitive / offline deployments."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        self._dimension: int = self._model.get_sentence_embedding_dimension() or MODEL_DIMENSIONS.get(model_name, 384)

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed(self, text: list[str]) -> list[list[float]]:
        """Offloads CPU-bound encode to a thread so the event loop stays responsive."""
        if not text:
            return []
        result = await asyncio.to_thread(
            self._model.encode, text, batch_size=BATCH_SIZE, show_progress_bar=False
        )
        return result.tolist()


# ── Fallback embedder ────────────────────────────────────────────────────────


class FallbackEmbedder(Embedder):
    """Tries the primary embedder first; falls back to the secondary on failure."""

    def __init__(self, primary: Embedder, fallback: Embedder) -> None:
        self.primary = primary
        self.fallback = fallback

    @property
    def dimension(self) -> int:
        return self.primary.dimension or self.fallback.dimension

    async def embed(self, text: list[str]) -> list[list[float]]:
        try:
            return await self.primary.embed(text)
        except Exception as exc:
            logger.warning("Primary embedder failed, using fallback: %s", exc)
            return await self.fallback.embed(text)

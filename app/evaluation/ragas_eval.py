from __future__ import annotations

import logging
from typing import Any

from datasets import Dataset

from app.core.config import get_settings
from app.evaluation.test_queries import TestQuery

logger = logging.getLogger(__name__)

# ── Optional ragas import ──────────────────────────────────────────────

try:
    from ragas import evaluate as ragas_evaluate
    from ragas.llms import llm_factory
    from ragas.embeddings.base import OpenAIEmbeddings as RagasOpenAIEmbeddings
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )

    _RAGAS_AVAILABLE = True
except ImportError:
    _RAGAS_AVAILABLE = False
    logger.warning("ragas package not installed — RagasEvaluator will be unavailable")


# ── Evaluation sample model ────────────────────────────────────────────


class EvalSample:
    """Results from running the RAG pipeline on a single test query.

    Attributes:
        question: The original query text.
        generated_answer: The LLM's answer given the retrieved context.
        retrieved_contexts: List of chunk text contents that were retrieved
            and fed to the LLM.
        ground_truth: The expected answer (from TestQuery.expected_answer).
    """

    def __init__(
        self,
        question: str,
        generated_answer: str,
        retrieved_contexts: list[str],
        ground_truth: str,
    ) -> None:
        self.question = question
        self.generated_answer = generated_answer
        self.retrieved_contexts = retrieved_contexts
        self.ground_truth = ground_truth


# ── Evaluator ──────────────────────────────────────────────────────────


class RagasEvaluator:
    """Computes RAGAS metrics for a set of evaluation samples.

    Uses the project's existing Ollama configuration (model, base URL, API key)
    to power RAGAS's LLM-based metrics (faithfulness, answer_relevancy) and
    embedding-based metrics (answer_relevancy needs embeddings).

    Metrics computed:
        - **Faithfulness**: Does the answer stay grounded in the retrieved
          context? (LLM-based, 0–1)
        - **Answer Relevancy**: How relevant is the answer to the question?
          (embedding-based, 0–1) — requires Ollama embeddings
        - **Context Precision**: Fraction of retrieved chunks that were
          relevant.  (LLM-based, 0–1)
        - **Context Recall**: Fraction of relevant chunks that were retrieved.
          (LLM-based, 0–1)

    Usage::

        evaluator = RagasEvaluator()
        samples = [EvalSample(...), ...]
        scores = await evaluator.evaluate(samples)
        print(scores["faithfulness"])
    """

    def __init__(self) -> None:
        if not _RAGAS_AVAILABLE:
            raise ImportError(
                "ragas is not installed. Install it with: uv add ragas"
            )

        settings = get_settings()

        # Configure LLM for ragas (uses Ollama via OpenAI-compatible API)
        from openai import AsyncOpenAI

        llm_client = AsyncOpenAI(
            base_url=settings.openai_base_url,
            api_key=settings.openai_api_key or "sk-placeholder",
        )
        self._llm = llm_factory(settings.openai_chat_model, client=llm_client)

        # Configure embeddings for ragas (uses Ollama via openai/ prefix)
        self._embeddings = RagasOpenAIEmbeddings(
            model=f"openai/{settings.openai_embedding_model}",
            openai_api_base=settings.openai_base_url,
            openai_api_key=settings.openai_api_key or "sk-placeholder",
        )

        self._metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

    async def evaluate(
        self,
        samples: list[EvalSample],
        metrics: list[str] | None = None,
    ) -> dict[str, float]:
        """Run RAGAS evaluation on the provided samples.

        Args:
            samples: List of ``EvalSample`` objects with pipeline results.
            metrics: Subset of metrics to compute.  ``None`` means all four.
                Accepted values: "faithfulness", "answer_relevancy",
                "context_precision", "context_recall".

        Returns:
            Dict mapping metric names to scores (0–1).  Missing or failed
            metrics return ``float("nan")``.
        """
        if not samples:
            logger.warning("No samples provided — returning empty scores")
            return {m.name: float("nan") for m in self._metrics}

        active_metrics = self._metrics
        if metrics is not None:
            active_metrics = [m for m in self._metrics if m.name in metrics]

        # Build the HuggingFace Dataset that RAGAS expects
        data: dict[str, list[Any]] = {
            "user_input": [],
            "response": [],
            "retrieved_contexts": [],
            "reference": [],
        }

        for sample in samples:
            data["user_input"].append(sample.question)
            data["response"].append(sample.generated_answer)
            data["retrieved_contexts"].append(sample.retrieved_contexts)
            data["reference"].append(sample.ground_truth)

        dataset = Dataset.from_dict(data)

        # Run evaluation
        import asyncio

        try:
            result = await asyncio.to_thread(
                ragas_evaluate,
                dataset,
                metrics=active_metrics,
                llm=self._llm,
                embeddings=self._embeddings,
            )
        except Exception as exc:
            logger.error("RAGAS evaluation failed: %s", exc)
            return {m.name: float("nan") for m in active_metrics}

        return {m.name: float(result[m.name]) for m in active_metrics}


# ── Convenience runner ────────────────────────────────────────────────


async def run_ragas_evaluation(
    samples: list[EvalSample],
    metrics: list[str] | None = None,
) -> dict[str, float]:
    """One-shot convenience wrapper — create an evaluator and run it."""
    evaluator = RagasEvaluator()
    return await evaluator.evaluate(samples, metrics=metrics)
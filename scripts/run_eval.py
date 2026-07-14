#!/usr/bin/env python3
"""CLI: run full evaluation suite.

Loads test queries, runs them through the RAG pipeline, then scores
results using LLM-as-a-judge and optionally RAGAS metrics.

Usage:
    uv run python scripts/run_eval.py
    uv run python scripts/run_eval.py --sample 3        # first 3 queries only
    uv run python scripts/run_eval.py --category general  # one category
    uv run python scripts/run_eval.py --ragas             # include RAGAS metrics
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time

from app.core.config import get_settings
from app.core.dependencies import get_embedder, get_indexer
from app.evaluation.test_queries import TestQuery, TestQuerySet
from app.generation.llm_client import LLMClient
from app.generation.prompt_builder import PromptBuilder
from app.retrieval.reranker import Reranker
from app.retrieval.retriever import Retriever, SearchResult

logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(message)s")


# ── Pipeline runner ────────────────────────────────────────────────────


class PipelineResult:
    """Results from running the RAG pipeline on a single query."""

    def __init__(
        self,
        test_query: TestQuery,
        chunks: list[SearchResult],
        answer: str,
        latency_ms: int,
    ) -> None:
        self.test_query = test_query
        self.chunks = chunks
        self.answer = answer
        self.latency_ms = latency_ms

    @property
    def retrieved_contexts(self) -> list[str]:
        return [c.content for c in self.chunks]

    @property
    def sources(self) -> list[dict]:
        return [
            {"chunk_id": c.chunk_id, "filename": c.filename, "score": c.score}
            for c in self.chunks
        ]


async def run_one_query(
    query: TestQuery,
    embedder,
    retriever: Retriever,
    reranker: Reranker,
    llm: LLMClient,
    top_k: int = 5,
) -> PipelineResult:
    """Run the full RAG pipeline for a single test query."""
    start = time.monotonic()

    # 1. Embed
    [query_vec] = await embedder.embed([query.question])

    # 2. Retrieve
    chunks = await retriever.search(query_vector=query_vec, top_k=top_k)

    # 3. Rerank
    chunks = await reranker.rerank(query=query.question, chunks=chunks, top_k=top_k)

    # 4. Build prompt and generate
    prompt = PromptBuilder().build(question=query.question, chunks=chunks)
    result = await llm.generate(
        user_message=prompt.user, system_prompt=prompt.system
    )

    latency = int((time.monotonic() - start) * 1000)
    return PipelineResult(
        test_query=query,
        chunks=chunks,
        answer=result.text,
        latency_ms=latency,
    )


# ── Reporting ──────────────────────────────────────────────────────────


def print_summary(results: list[PipelineResult], judgments: list | None = None,
                  ragas_scores: dict | None = None) -> None:
    """Print a human-readable evaluation summary."""
    print("\n" + "=" * 60)
    print("RAG EVALUATION RESULTS")
    print("=" * 60)

    # ── RAGAS scores ──────────────────────────────────────────────
    if ragas_scores:
        print("\n📊 RAGAS Metrics:")
        print("-" * 40)
        for metric, score in ragas_scores.items():
            label = score if score == "N/A" else f"{score:.3f}"
            print(f"  {metric:25s}  {label}")
        print()

    # ── LLM Judge scores ──────────────────────────────────────────
    if judgments:
        print("\n⚖️  LLM-as-a-Judge Scores:")
        print("-" * 40)
        overalls = []
        for j in judgments:
            s = j.scores
            print(f"  Accuracy:        {s.accuracy}/5")
            print(f"  Completeness:    {s.completeness}/5")
            print(f"  Conciseness:     {s.conciseness}/5")
            print(f"  Citation:        {s.citation_quality}/5")
            print(f"  Overall:         {s.overall}/5")
            print(f"  Hallucination:   {'⚠️ YES' if j.is_hallucination else '✅ No'}")
            print(f"  Reasoning:       {j.reasoning[:200]}...")
            print()
            overalls.append(s.overall)
        if overalls:
            avg = sum(overalls) / len(overalls)
            print(f"  Average Overall: {avg:.2f}/5")

    # ── Per-query details ─────────────────────────────────────────
    print("\n📋 Per-Query Breakdown:")
    print("-" * 40)
    latencies = []
    for r in results:
        symbol = "⚠️" if not r.chunks else "✅"
        print(
            f"  {symbol} [{r.test_query.category:12s}] "
            f"{r.test_query.question[:50]:50s} "
            f"{r.latency_ms:4d}ms  ({len(r.chunks)} chunks)"
        )
        latencies.append(r.latency_ms)

    # ── Aggregate stats ───────────────────────────────────────────
    if latencies:
        avg_lat = sum(latencies) / len(latencies)
        p95 = sorted(latencies)[int(len(latencies) * 0.95)]
        print(f"\n  ⏱ Mean latency: {avg_lat:.0f}ms  p95: {p95}ms")

    no_results = sum(1 for r in results if not r.chunks)
    if no_results:
        print(f"  ⚠️  {no_results}/{len(results)} queries returned no chunks")

    if judgments:
        hallu_count = sum(1 for j in judgments if j.is_hallucination)
        print(f"  🌀 Hallucination rate: {hallu_count}/{len(judgments)}")

    print("=" * 60 + "\n")


# ── Main ───────────────────────────────────────────────────────────────


async def main(args: argparse.Namespace) -> None:
    settings = get_settings()

    # 1. Load queries
    all_queries = TestQuerySet.load_all().queries
    if args.category:
        all_queries = [q for q in all_queries if q.category == args.category]
    if args.sample:
        all_queries = all_queries[: args.sample]

    if not all_queries:
        print("No test queries found.")
        sys.exit(1)

    print(f"Running {len(all_queries)} test queries...")

    # 2. Initialize components
    embedder = get_embedder()
    indexer = get_indexer()
    retriever = Retriever(indexer)
    reranker = Reranker(settings.openai_embedding_model)
    llm = LLMClient()

    # 3. Run pipeline for each query
    results: list[PipelineResult] = []
    for i, query in enumerate(all_queries, start=1):
        label = f"[{i}/{len(all_queries)}]"
        print(f"  {label} Querying: {query.question[:60]}...")
        try:
            result = await run_one_query(
                query, embedder, retriever, reranker, llm,
                top_k=args.top_k,
            )
            results.append(result)
        except Exception as exc:
            print(f"  {label} FAILED: {exc}")

    # 4. LLM-as-a-judge
    from app.evaluation.llm_judge import LLMJudge

    judge = LLMJudge(llm)
    judgments = []
    for r in results:
        ctx = r.retrieved_contexts if args.include_context else None
        judgment = await judge.evaluate(
            question=r.test_query.question,
            generated_answer=r.answer,
            expected_answer=r.test_query.expected_answer,
            retrieved_contexts=ctx,
        )
        judgments.append(judgment)

    # 5. RAGAS (optional)
    ragas_scores = None
    if args.ragas:
        try:
            from app.evaluation.ragas_eval import EvalSample, run_ragas_evaluation

            samples = [
                EvalSample(
                    question=r.test_query.question,
                    generated_answer=r.answer,
                    retrieved_contexts=r.retrieved_contexts,
                    ground_truth=r.test_query.expected_answer,
                )
                for r in results
            ]
            ragas_scores = await run_ragas_evaluation(samples)
        except Exception as exc:
            print(f"RAGAS evaluation skipped: {exc}")

    # 6. Print summary
    print_summary(results, judgments, ragas_scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG evaluation suite")
    parser.add_argument("--sample", type=int, default=0,
                        help="Run only first N queries")
    parser.add_argument("--category", type=str, default="",
                        help="Filter by category (general, edge_case, synthesis, evaluation)")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Number of chunks to retrieve per query")
    parser.add_argument("--include-context", action="store_true",
                        help="Include retrieved context in LLM judge prompts")
    parser.add_argument("--ragas", action="store_true",
                        help="Also compute RAGAS metrics (requires ragas package)")
    args = parser.parse_args()

    asyncio.run(main(args))
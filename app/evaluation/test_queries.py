from __future__ import annotations

from pydantic import BaseModel, Field

class TestQuery(BaseModel):
    question: str
    expected_answer: str
    relevant_chunk_ids: list[str] = Field(default_factory=list)
    document_ids: list[str] = Field(default_factory=list)
    category: str = "general"


class TestQuerySet(BaseModel):
    queries: list[TestQuery] = Field(default_factory=list)

    def by_category(self, category: str) -> list[TestQuery]:
        return [q for q in self.queries if q.category == category]
    
    def sample(self, n: int) -> list[TestQuery]:
        return self.queries[:n]
    
    @classmethod
    def load_all(cls) -> TestQuerySet:
        return cls(queries=_CURATED_QUERIES)
    


# ── Curated test queries ─────────────────────────────────────────────────

# These are designed to be document-agnostic — they test general RAG
# capabilities.  Once you have indexed specific documents, add queries
# tied to those documents with the correct relevant_chunk_ids.

_CURATED_QUERIES: list[TestQuery] = [
    # ── General knowledge (no specific doc required) ────────────────
    TestQuery(
        question="What is retrieval-augmented generation?",
        expected_answer=(
            "Retrieval-Augmented Generation (RAG) is an AI framework that "
            "combines information retrieval with text generation. It retrieves "
            "relevant documents from a knowledge base and feeds them as context "
            "to a large language model, which then generates a grounded answer "
            "citing those sources."
        ),
        category="general",
    ),
    TestQuery(
        question="How does RAG reduce hallucinations?",
        expected_answer=(
            "RAG reduces hallucinations by grounding the LLM's response in "
            "retrieved documents rather than relying solely on its parametric "
            "knowledge. The model is instructed to answer only from the provided "
            "context, which constrains its output to evidence-backed claims."
        ),
        category="general",
    ),
    TestQuery(
        question="What is the difference between dense and sparse retrieval?",
        expected_answer=(
            "Dense retrieval uses neural embeddings (vectors) to capture semantic "
            "similarity between query and documents. Sparse retrieval (e.g., BM25) "
            "uses keyword matching based on term frequency and inverse document "
            "frequency. Dense is better at capturing meaning, while sparse excels "
            "at exact keyword matching."
        ),
        category="general",
    ),
    # ── Edge cases ──────────────────────────────────────────────────
    TestQuery(
        question="What is chunking?",
        expected_answer=(
            "Chunking is the process of splitting a document into smaller, "
            "meaningful segments (chunks) for indexing and retrieval."
        ),
        category="edge_case",
    ),
    TestQuery(
        question="Tell me about something that doesn't exist in the documents",
        expected_answer=(
            "I don't have sufficient information to answer this question."
        ),
        category="edge_case",
    ),
    TestQuery(
        question="",
        expected_answer="I don't have sufficient information to answer this question.",
        category="edge_case",
    ),
    # ── Synthesis ───────────────────────────────────────────────────
    TestQuery(
        question="Compare chunking strategies for RAG",
        expected_answer=(
            "Different chunking strategies include fixed-size splitting, "
            "recursive character splitting, semantic boundary detection, and "
            "structure-aware chunking that preserves tables, code blocks, and "
            "lists. The choice depends on document type and retrieval goals."
        ),
        category="synthesis",
    ),
    TestQuery(
        question="What factors affect RAG answer quality?",
        expected_answer=(
            "RAG answer quality depends on chunk quality, retrieval recall, "
            "reranker effectiveness, prompt design, and the LLM's instruction "
            "following. Poor chunking or low retrieval recall leads to incomplete "
            "context, while a weak reranker may surface irrelevant chunks."
        ),
        category="synthesis",
    ),
    # ── Evaluation-specific ─────────────────────────────────────────
    TestQuery(
        question="What metrics are used to evaluate RAG systems?",
        expected_answer=(
            "Common RAG evaluation metrics include Faithfulness (does the answer "
            "stay grounded in context?), Answer Relevancy (does it address the "
            "question?), Context Precision (what % of retrieved chunks were "
            "relevant?), and Context Recall (what % of relevant chunks were "
            "retrieved?)."
        ),
        category="evaluation",
    ),
    TestQuery(
        question="What is LLM-as-a-judge?",
        expected_answer=(
            "LLM-as-a-judge uses a separate language model to score the quality "
            "of generated answers on dimensions like accuracy, completeness, "
            "conciseness, and citation quality. It provides an automated "
            "alternative to human evaluation."
        ),
        category="evaluation",
    ),
]
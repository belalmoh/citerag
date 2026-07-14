from __future__ import annotations

import json
import logging
import re

from pydantic import BaseModel, Field

from app.generation.llm_client import LLMClient

logger = logging.getLogger(__name__)


# ── Scoring model ────────────────────────────────────────────────────────


class JudgeScores(BaseModel):
    """Score dimensions for evaluating a RAG answer."""

    accuracy: int = Field(ge=1, le=5, description="Is the answer factually correct?")
    completeness: int = Field(ge=1, le=5, description="Does it cover all aspects of the question?")
    conciseness: int = Field(ge=1, le=5, description="Is it concise without unnecessary info?")
    citation_quality: int = Field(ge=1, le=5, description="Are sources cited correctly?")

    @property
    def overall(self) -> float:
        """Weighted average across all dimensions."""
        return round(
            self.accuracy * 0.4
            + self.completeness * 0.3
            + self.conciseness * 0.1
            + self.citation_quality * 0.2,
            2,
        )


class Judgment(BaseModel):
    """The judge's verdict on a single generated answer."""

    scores: JudgeScores
    reasoning: str = Field(description="Explanation for the scores")
    is_hallucination: bool = Field(
        description="Does the answer contain claims not supported by the context?"
    )


# ── Judge class ─────────────────────────────────────────────────────────


_EVALUATION_SYSTEM_PROMPT = """You are a strict but fair judge evaluating RAG (Retrieval-Augmented Generation) answers.

Score each dimension 1 (worst) to 5 (best):

**Accuracy** — Is every claim in the answer supported by the provided context?
  5 = All claims directly supported, no errors
  3 = Most claims supported, minor inaccuracies
  1 = Major factual errors or hallucinations

**Completeness** — Does the answer fully address the question?
  5 = Covers all aspects of the question comprehensively
  3 = Addresses the main point but misses nuance
  1 = Misses the key information entirely

**Conciseness** — Is the answer focused and to the point?
  5 = Direct answer with no filler
  3 = Some unnecessary detail but still useful
  1 = Overly verbose or rambling

**Citation Quality** — Are sources cited appropriately?
  5 = Every claim cites a source, citations are relevant
  3 = Some citations but not all claims are cited
  1 = No citations or citations don't match claims

Also determine: **is_hallucination** (true/false) — does the answer contain any claim that contradicts or is not supported by the context?"""


class LLMJudge:
    """Uses an LLM to score the quality of a generated RAG answer.

    Why use a separate LLM as judge instead of relying on metrics?
        An LLM judge can evaluate semantic qualities (accuracy, completeness)
        that automated metrics struggle with. It's the closest automated
        proxy for human evaluation.
    """

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self._llm = llm_client or LLMClient()

    async def evaluate(
        self,
        question: str,
        generated_answer: str,
        expected_answer: str | None = None,
        retrieved_contexts: list[str] | None = None,
    ) -> Judgment:
        """Score a single generated answer.

        Args:
            question: The original user query.
            generated_answer: The answer produced by the RAG pipeline.
            expected_answer: Ground-truth answer (optional, for reference).
            retrieved_contexts: The chunks retrieved and fed to the LLM.

        Returns:
            A ``Judgment`` with scores, reasoning, and hallucination flag.
        """
        context_block = ""
        if retrieved_contexts:
            context_block = "\n\n".join(
                f"[{i+1}] {c}" for i, c in enumerate(retrieved_contexts)
            )

        expected_block = ""
        if expected_answer:
            expected_block = f"\n\nExpected answer (ground truth):\n{expected_answer}"

        prompt = (
            f"Question: {question}\n\n"
            f"Generated answer:\n{generated_answer}"
            f"{expected_block}"
            f"\n\nRetrieved context:\n{context_block}"
            if context_block
            else ""
        )

        response = await self._llm.generate(
            user_message=prompt,
            system_prompt=_EVALUATION_SYSTEM_PROMPT,
            temperature=0.1,  # Low temperature for consistent scoring
            max_tokens=512,
        )

        return self._parse_response(response.text)

    @staticmethod
    def _parse_response(text: str) -> Judgment:
        """Extract structured judgment from the LLM's free-form response.

        Looks for a JSON block in the response first. Falls back to
        regex-based extraction of individual scores.
        """
        # Try JSON extraction first
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                # Validate required fields
                scores = JudgeScores(
                    accuracy=data.get("accuracy", 3),
                    completeness=data.get("completeness", 3),
                    conciseness=data.get("conciseness", 3),
                    citation_quality=data.get("citation_quality", 3),
                )
                return Judgment(
                    scores=scores,
                    reasoning=data.get("reasoning", text[:500]),
                    is_hallucination=data.get("is_hallucination", False),
                )
            except (json.JSONDecodeError, ValueError, KeyError):
                logger.warning("Failed to parse JSON from judge response", exc_info=True)

        # Fallback: extract scores via regex
        accuracy = _extract_score(text, r"accuracy.*?(\d)")
        completeness = _extract_score(text, r"completeness.*?(\d)")
        conciseness = _extract_score(text, r"conciseness.*?(\d)")
        citation_quality = _extract_score(text, r"citation.*?(\d)")
        has_hallucination = bool(re.search(r"(?i)hallucinat.*?true|yes", text))

        return Judgment(
            scores=JudgeScores(
                accuracy=accuracy,
                completeness=completeness,
                conciseness=conciseness,
                citation_quality=citation_quality,
            ),
            reasoning=text[:500],
            is_hallucination=has_hallucination,
        )


def _extract_score(text: str, pattern: str) -> int:
    """Extract a numeric score (1-5) from text using a regex pattern."""
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        try:
            score = int(match.group(1))
            return max(1, min(5, score))
        except ValueError:
            pass
    return 3  # Default to middle score on parse failure
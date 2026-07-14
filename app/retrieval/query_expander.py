from __future__ import annotations

import logging
import re

from app.generation.llm_client import LLMClient

logger = logging.getLogger(__name__)

class QueryExpander:
    def __init__(self, llm_client: LLMClient):
        self._llm_client = llm_client

    async def expand_query(self, query: str, num_expansions: int = 3) -> list[str]:
        """Generate alternative phrasings of *query*.

        Returns ``[original_query, expansion_1, expansion_2, ...]``.
        The original is always first so callers can weight it higher if desired.
        """
        if not query.strip():
            return [query]
        
        prompt = (
            f"Generate {num_expansions} alternative phrasings of the following "
            f"question. Each phrasing should preserve the original meaning but "
            f"use different words and sentence structure.\n\n"
            f"Return ONLY the phrasings, one per line, numbered 1-{num_expansions}.\n"
            f"Do not include the original question or any explanation.\n\n"
            f"Original: {query}\n\n"
            f"Alternative phrasings:"
        )

        response = await self._llm_client.generate(
            user_message=prompt,
            system_prompt="You are a query expansion assistant. Output only the requested phrasings, nothing else.",
            temperature=0.7,
            max_tokens=256
        )

        expansions = self._parse_response(response.text)
        logger.info(
            "Expanded query '%s' → %d variations (requested %d)",
            query[:50], len(expansions), num_expansions,
        )
        return [query] + expansions
    
    @staticmethod
    def _parse_response(text: str) -> list[str]:
        """Extract clean phrasings from the LLM's numbered response."""
        lines = text.strip().split("\n")
        phrasings: list[str] = []
        for line in lines:
            line = line.strip()
            # Remove leading numbering like "1.", "2)", "- "
            line = re.sub(r"^[\d]+[\.\)]\s*", "", line).strip()
            line = re.sub(r"^[-*]\s+", "", line).strip()
            if line and len(line) > 5:  # Ignore empty or garbage lines
                phrasings.append(line)
        return phrasings
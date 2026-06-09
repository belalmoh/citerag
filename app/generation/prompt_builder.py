
from pydantic import BaseModel

from app.retrieval.retriever import SearchResult


class Prompt(BaseModel):
    system: str
    user: str

class PromptBuilder:

    DEFAULT_SYSTEM_PROMPT = (
        "You are a helpful assistant that answers questions "
        "based ONLY on the provided context. If the context "
        "does not contain enough information to answer the "
        "question, say so clearly. Do not make up facts.\n\n"
        "Cite your sources by referring to the chunk numbers "
        "listed in the context (e.g., [1], [2])."
    )

    def build(self, question: str, chunks: list[SearchResult], system_prompt: str | None = None) -> Prompt:
        if system_prompt is None:
            system_prompt = self.DEFAULT_SYSTEM_PROMPT

        context_parts: list[str] = []
        for i, chunk in enumerate(chunks, start=1):
            source = f"[{i}] {chunk.content.strip()}"
            if chunk.filename:
                source += f"\n     (source: {chunk.filename})"
            context_parts.append(source)

        context = "\n\n".join(context_parts) if context_parts else "(no relevant documents found)"
        user_prompt = (
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            f"Answer:"
        )

        return Prompt(system=system_prompt, user=user_prompt)
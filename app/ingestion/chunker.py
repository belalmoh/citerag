from __future__ import annotations

import re

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel


class Chunk(BaseModel):
    content: str
    index: int
    metadata: dict


# ── Protected-region markers ─────────────────────────────────────────────────
# The parser can wrap structural elements (tables, code blocks, etc.) with
# these markers.  The chunker extracts them first so that LangChain's
# RecursiveCharacterTextSplitter never splits inside a protected region.

PROTECTED_MARKERS: dict[str, tuple[str, str]] = {
    "table": ("[TABLE_START]", "[TABLE_END]"),
    "code": ("[CODE_START]", "[CODE_END]"),
    "list": ("[LIST_START]", "[LIST_END]"),
}

_PLACEHOLDER_PREFIX = "\x00PROTECTED_"


class Chunker:
    """Structure-aware text chunker powered by LangChain's
    ``RecursiveCharacterTextSplitter`` with an added layer that prevents
    splitting inside protected regions (tables, code blocks, lists, etc.).

    Strategy
    --------
    1. **Extract protected regions** — replace each marked block with a unique
       placeholder so the splitter never sees (and cannot break) the content.
    2. **Recursive split** — delegate to ``RecursiveCharacterTextSplitter``
       which tries paragraph → line → sentence → word → character separators.
    3. **Restore protected regions** — swap placeholders back with their
       original content.
    4. **Assemble Chunk objects** — attach metadata and sequential indices.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: list[str] | None = None,
        protected_markers: dict[str, tuple[str, str]] | None = None,
    ) -> None:
        if chunk_overlap >= chunk_size:
            msg = (
                f"chunk_overlap ({chunk_overlap}) must be smaller than "
                f"chunk_size ({chunk_size})"
            )
            raise ValueError(msg)

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.protected_markers = protected_markers or dict(PROTECTED_MARKERS)

        splitter_kwargs: dict = {
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "length_function": len,
        }
        if separators is not None:
            splitter_kwargs["separators"] = separators

        self._splitter = RecursiveCharacterTextSplitter(**splitter_kwargs)

    # ── Public API ────────────────────────────────────────────────────────

    def split(self, text: str, metadata: dict) -> list[Chunk]:
        if not text.strip():
            return [
                Chunk(content=text, index=0, metadata={**metadata, "chunk_index": 0})
            ]
        protected_map: dict[str, str] = {}
        prepared_text = self._extract_protected(text, protected_map)
        raw_chunks = self._splitter.split_text(prepared_text)
        restored = self._restore_protected(raw_chunks, protected_map)
        return self._build_chunks(restored, metadata)

    # ── Protected-region handling ─────────────────────────────────────────

    def _extract_protected(self, text: str, placeholder_map: dict[str, str]) -> str:
        """Replace every protected region with a unique placeholder token."""
        result = text
        for kind, (start_marker, end_marker) in self.protected_markers.items():
            pattern = re.compile(
                rf"{re.escape(start_marker)}.*?{re.escape(end_marker)}", re.DOTALL
            )
            for idx, match in enumerate(pattern.finditer(text), start=1):
                placeholder = f"{_PLACEHOLDER_PREFIX}{kind}_{idx}\x00"
                placeholder_map[placeholder] = match.group(0)
                result = result.replace(match.group(0), placeholder, 1)
        return result

    def _restore_protected(
        self, chunks: list[str], placeholder_map: dict[str, str]
    ) -> list[str]:
        """Swap placeholders back with their original protected content."""
        return [
            "".join(
                placeholder_map.get(part, part)
                for part in re.split(r"(\x00PROTECTED_\w+\x00)", c)
            )
            for c in chunks
        ]

    # ── Chunk assembly ────────────────────────────────────────────────────

    @staticmethod
    def _build_chunks(contents: list[str], metadata: dict) -> list[Chunk]:
        chunks: list[Chunk] = []
        for idx, content in enumerate(contents):
            chunk_metadata = metadata.copy()
            chunk_metadata["chunk_index"] = idx
            chunks.append(Chunk(content=content, index=idx, metadata=chunk_metadata))
        return chunks

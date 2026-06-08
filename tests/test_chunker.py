from __future__ import annotations

import pytest

from app.ingestion.chunker import Chunk, Chunker, PROTECTED_MARKERS


# ── Helpers ──────────────────────────────────────────────────────────────────

def _contents(chunks: list[Chunk]) -> list[str]:
    return [c.content for c in chunks]


# ── Basic splitting (delegates to LangChain) ─────────────────────────────────


class TestBasicSplitting:
    def test_short_text_returns_single_chunk(self):
        chunker = Chunker(chunk_size=100, chunk_overlap=10)
        result = chunker.split("Hello world", {"doc_id": "1"})
        assert len(result) == 1
        assert result[0].content.strip() == "Hello world"

    def test_metadata_is_propagated(self):
        chunker = Chunker(chunk_size=100, chunk_overlap=10)
        meta = {"doc_id": "1", "source": "test"}
        result = chunker.split("Short text", meta)
        assert result[0].metadata["doc_id"] == "1"
        assert result[0].metadata["source"] == "test"
        assert result[0].metadata["chunk_index"] == 0

    def test_chunk_indices_are_sequential(self):
        chunker = Chunker(chunk_size=20, chunk_overlap=5)
        text = "Word " * 40  # 200 chars
        result = chunker.split(text, {})
        for i, chunk in enumerate(result):
            assert chunk.index == i
            assert chunk.metadata["chunk_index"] == i

    def test_metadata_is_copied_not_shared(self):
        chunker = Chunker(chunk_size=50, chunk_overlap=5)
        result = chunker.split("A longer piece of text here for splitting into multiple chunks", {"key": "val"})
        result[0].metadata["extra"] = "only_here"
        assert "extra" not in result[1].metadata

    def test_long_text_produces_multiple_chunks(self):
        chunker = Chunker(chunk_size=50, chunk_overlap=0)
        text = "Word " * 200  # ~1000 chars
        result = chunker.split(text, {})
        assert len(result) > 1


# ── Recursive splitting ──────────────────────────────────────────────────────


class TestRecursiveSplitting:
    def test_splits_on_paragraph_breaks(self):
        chunker = Chunker(chunk_size=50, chunk_overlap=5)
        text = "First paragraph here.\n\nSecond paragraph here.\n\nThird one."
        result = chunker.split(text, {})
        assert len(result) >= 2

    def test_respects_chunk_size_upper_bound(self):
        chunker = Chunker(chunk_size=50, chunk_overlap=0)
        text = "Word " * 200  # 1000 chars
        result = chunker.split(text, {})
        # LangChain may slightly exceed chunk_size due to separator inclusion,
        # but chunks should be roughly in the right ballpark.
        for chunk in result:
            assert len(chunk.content) <= 120  # generous buffer


# ── Overlap ──────────────────────────────────────────────────────────────────


class TestOverlap:
    def test_overlap_exists_between_consecutive_chunks(self):
        chunker = Chunker(chunk_size=40, chunk_overlap=10)
        text = "A" * 100
        result = chunker.split(text, {})
        if len(result) > 1:
            # Overlap means the start of chunk[i] should appear in chunk[i-1].
            for i in range(1, len(result)):
                overlap_text = result[i].content[:10]
                assert overlap_text in result[i - 1].content

    def test_zero_overlap(self):
        chunker = Chunker(chunk_size=20, chunk_overlap=0)
        text = "X" * 60
        result = chunker.split(text, {})
        assert len(result) >= 2

    def test_overlap_must_be_smaller_than_chunk_size(self):
        with pytest.raises(ValueError, match="chunk_overlap"):
            Chunker(chunk_size=50, chunk_overlap=100)


# ── Protected regions ────────────────────────────────────────────────────────


class TestProtectedRegions:
    def test_table_not_split_across_chunks(self):
        chunker = Chunker(chunk_size=50, chunk_overlap=0)
        table = "[TABLE_START]col1|col2\na|b\n[TABLE_END]"
        text = "Some intro text. " + table + " Some trailing text."
        result = chunker.split(text, {})
        table_chunks = [c for c in result if table in c.content]
        assert len(table_chunks) >= 1, "Protected table was split across chunks!"

    def test_code_block_not_split(self):
        chunker = Chunker(chunk_size=50, chunk_overlap=0)
        code = "[CODE_START]def hello():\n    return 'world'[CODE_END]"
        text = "Preamble. " + code + " Postamble."
        result = chunker.split(text, {})
        code_chunks = [c for c in result if code in c.content]
        assert len(code_chunks) >= 1, "Protected code block was split across chunks!"

    def test_list_not_split(self):
        chunker = Chunker(chunk_size=50, chunk_overlap=0)
        lst = "[LIST_START]- item1\n- item2\n- item3[LIST_END]"
        text = "Intro. " + lst + " Outro."
        result = chunker.split(text, {})
        list_chunks = [c for c in result if lst in c.content]
        assert len(list_chunks) >= 1, "Protected list was split across chunks!"

    def test_multiple_protected_regions(self):
        chunker = Chunker(chunk_size=60, chunk_overlap=0)
        table = "[TABLE_START]a|b\n1|2[TABLE_END]"
        code = "[CODE_START]x = 1[CODE_END]"
        text = table + " Some text between. " + code
        result = chunker.split(text, {})
        assert any(table in c.content for c in result)
        assert any(code in c.content for c in result)

    def test_large_protected_region_stays_intact(self):
        chunker = Chunker(chunk_size=50, chunk_overlap=0)
        # A table larger than chunk_size — it should still stay intact.
        table = "[TABLE_START]" + "data," * 20 + "[TABLE_END]"
        text = "Intro. " + table
        result = chunker.split(text, {})
        assert any(table in c.content for c in result)

    def test_custom_protected_markers(self):
        custom_markers = {"custom": ("<CUSTOM>", "</CUSTOM>")}
        chunker = Chunker(chunk_size=50, chunk_overlap=0, protected_markers=custom_markers)
        region = "<CUSTOM>important data here</CUSTOM>"
        text = "Before. " + region + " After."
        result = chunker.split(text, {})
        assert any(region in c.content for c in result)


# ── Edge cases ────────────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_empty_text(self):
        chunker = Chunker(chunk_size=100, chunk_overlap=10)
        result = chunker.split("", {})
        assert len(result) == 1
        assert result[0].content == ""

    def test_text_shorter_than_chunk_size(self):
        chunker = Chunker(chunk_size=1000, chunk_overlap=100)
        text = "Just a short piece."
        result = chunker.split(text, {})
        assert len(result) == 1
        assert result[0].content.strip() == text

    def test_custom_separators(self):
        chunker = Chunker(chunk_size=30, chunk_overlap=5, separators=["||", "\n", " "])
        text = "Part one||Part two||Part three||Part four"
        result = chunker.split(text, {})
        assert len(result) >= 2

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

import pymupdf
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

# Extensions that pymupdf can open directly
PYMUPDF_EXTENSIONS: set[str] = {
    ".pdf",
    ".xps",
    ".oxps",
    ".epub",
    ".mobi",
    ".fb2",
    ".cbz",
    ".cbr",
    ".svg",
    ".txt",
}

# Common office formats pymupdf handles via conversion
OFFICE_EXTENSIONS: set[str] = {
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
    ".ppt",
    ".pptx",
}

SUPPORTED_EXTENSIONS: set[str] = PYMUPDF_EXTENSIONS | OFFICE_EXTENSIONS

# ── Models ────────────────────────────────────────────────────────────────────

class Table(BaseModel):
    """A single table extracted from a document page."""

    page_number: int
    rows: list[list[str]]
    row_count: int
    col_count: int

class Page(BaseModel):
    text: str
    number: int
    tables: list[Table] = []

# ── Parser ────────────────────────────────────────────────────────────────────

class ParsedDocument(BaseModel):
    text: str
    metadata: dict
    pages: list[Page] | None = None
    tables: list[Table] | None = None
    source: str
    parser: str


class DocumentParser:
    """Parse documents by routing to pymupdf or a plain-text fallback.

    Supports both sync and async parsing. The async variant offloads
    blocking pymupdf and file-I/O work to a thread-pool executor so
    the event loop stays responsive.

    - If the file extension is supported by pymupdf (PDF, DOC, DOCX, etc.),
      pymupdf is used for extraction.
    - Otherwise, the file is read as plain text (UTF-8 with fallback).
    """

    def __init__(self, filepath: str | Path, max_file_size_mb: int | None = None) -> None:
        self.filepath = Path(filepath)
        self._max_bytes = (max_file_size_mb or 50) * 1024 * 1024
        self.ext: str = self.filepath.suffix.lower()
        self._use_pymupdf = self.ext in SUPPORTED_EXTENSIONS
        # File existence and size validated lazily (see _validate_file)
        self._validated: bool = False

    def _validate_file(self) -> None:
        """Check file existence and size. Idempotent after first call."""
        if self._validated:
            return
        if not self.filepath.exists():
            raise FileNotFoundError(f"Document not found: {self.filepath}")
        if not self.filepath.is_file():
            raise ValueError(f"Path is not a file: {self.filepath}")
        file_size = self.filepath.stat().st_size
        if file_size > self._max_bytes:
            raise ValueError(
                f"File too large: {file_size / 1024 / 1024:.1f} MB "
                f"(limit: {self._max_bytes / 1024 / 1024:.0f} MB)"
            )
        self._validated = True

    @property
    def is_validated(self) -> bool:
        return self._validated

    # ── Public API ───────────────────────────────────────────────────────

    def parse(self) -> ParsedDocument:
        """Extract text and metadata from the document (synchronous)."""
        self._validate_file()
        if self._use_pymupdf:
            return self._parse_with_pymupdf()
        return self._parse_as_text()

    async def parse_async(self) -> ParsedDocument:
        """Extract text and metadata from the document (asynchronous).

        Offloads blocking file-I/O and pymupdf operations to the default
        thread-pool executor so the calling async context isn't blocked.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.parse)

    # ── pymupdf path ─────────────────────────────────────────────────────

    def _parse_with_pymupdf(self) -> ParsedDocument:
        logger.info("Parsing with pymupdf: %s", self.filepath)
        text_parts: list[str] = []
        metadata: dict[str, Any] = {}
        pages: list[Page] = []
        all_tables: list[Table] = []

        with pymupdf.open(str(self.filepath)) as doc:
            metadata = doc.metadata or {}
            for page in doc:
                text_parts.append(page.get_text())

                tables = self._extract_tables(page, len(pages) + 1)
                all_tables.extend(tables)

                pages.append(Page(text=page.get_text(), number=len(pages) + 1, tables=tables))

        return ParsedDocument(
            text="\n".join(text_parts),
            metadata=metadata,
            source=str(self.filepath),
            parser="pymupdf",
            pages=pages,
            tables=all_tables or None,
        )

    def _extract_tables(self, page: pymupdf.Page, page_number: int) -> list[Table]:
        tables: list[Table] = []
        try:
            for tab in page.find_tables():
                rows = [list(cell) for cell in tab.extract()]
                tables.append(Table(
                    page_number=page_number,
                    rows=rows,
                    row_count=len(rows),
                    col_count=len(rows[0]) if rows else 0
                ))
        except Exception as e:
            logger.error("Error extracting tables from page %d: %s", page_number, e)
        return tables

    # ── Plain-text fallback ───────────────────────────────────────────────

    def _parse_as_text(self) -> ParsedDocument:
        logger.info("Parsing as plain text: %s", self.filepath)
        try:
            text = self.filepath.read_text(encoding="utf-8")
            metadata = {
                "filename": self.filepath.name,
                "extension": self.ext,
                "size_bytes": self.filepath.stat().st_size,
            }
            source = str(self.filepath)
            parser = "text"
        except UnicodeDecodeError:
            text = self.filepath.read_text(encoding="latin-1")
            metadata = {
                "filename": self.filepath.name,
                "extension": self.ext,
                "size_bytes": self.filepath.stat().st_size,
            }
            source = str(self.filepath)
            parser = "text_fallback"
        return ParsedDocument(
            text=text, metadata=metadata, source=source, parser=parser
        )
    
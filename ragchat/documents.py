"""PDF extraction, cleanup, and provenance-preserving chunking."""

from __future__ import annotations

import re
from io import BytesIO

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


class PdfExtractionError(ValueError):
    """Raised when an uploaded file cannot yield readable PDF text."""


def clean_text(text: str) -> str:
    """Normalize common PDF extraction artefacts without joining paragraphs."""
    text = text.replace("\x00", "")
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pdf_pages(pdf_bytes: bytes, filename: str) -> list[Document]:
    """Extract non-empty pages and attach stable filename/page metadata."""
    try:
        reader = PdfReader(BytesIO(pdf_bytes))
    except Exception as error:  # pypdf exposes several parser-specific exceptions.
        raise PdfExtractionError(f"Could not read '{filename}' as a PDF.") from error

    pages: list[Document] = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = clean_text(page.extract_text() or "")
        if text:
            pages.append(
                Document(page_content=text, metadata={"filename": filename, "page": page_number})
            )
    if not pages:
        raise PdfExtractionError(f"'{filename}' has no extractable text (it may be a scanned PDF).")
    return pages


def chunk_pages(
    pages: list[Document], *, chunk_size: int, chunk_overlap: int
) -> list[Document]:
    """Split pages into retrieval chunks while retaining page-level provenance."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks: list[Document] = []
    for page in pages:
        for chunk_number, text in enumerate(splitter.split_text(page.page_content), start=1):
            metadata = {**page.metadata, "chunk": chunk_number}
            chunks.append(Document(page_content=text, metadata=metadata))
    return chunks


def documents_from_uploads(
    uploads: list[tuple[str, bytes]], *, chunk_size: int, chunk_overlap: int
) -> list[Document]:
    """Extract and chunk many uploads, failing clearly if none provide text."""
    chunks = [
        chunk
        for filename, pdf_bytes in uploads
        for chunk in chunk_pages(
            extract_pdf_pages(pdf_bytes, filename),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    ]
    if not chunks:
        raise PdfExtractionError("No extractable text was found in the uploaded PDFs.")
    return chunks

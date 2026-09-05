"""
Pydantic schemas for text chunking.

These describe the API response shape for the chunking endpoint. The
service layer (backend/services/chunking_service.py) works with plain
dataclasses; these schemas mirror that structure for the API boundary.
"""

from pydantic import BaseModel, Field


class TextChunkSchema(BaseModel):
    """A single chunk of text, traceable back to its document and page."""

    document_id: str = Field(..., description="Identifier of the document this chunk belongs to.")
    page_number: int = Field(..., ge=1, description="1-indexed page number this chunk was extracted from.")
    chunk_index: int = Field(..., ge=0, description="0-indexed position of this chunk across the whole document.")
    chunk_index_in_page: int = Field(..., ge=0, description="0-indexed position of this chunk within its page only.")
    text: str = Field(..., description="The chunk's text content.")
    char_count: int = Field(..., ge=0, description="Number of characters in this chunk's text.")


class ChunkingResponse(BaseModel):
    """Response returned after successfully chunking a stored document's extracted text."""

    document_id: str = Field(..., description="Identifier of the document that was chunked.")
    total_pages: int = Field(..., ge=0, description="Total number of pages in the source document.")
    total_chunks: int = Field(..., ge=0, description="Total number of chunks produced across all pages.")
    chunks: list[TextChunkSchema] = Field(..., description="Chunks in document order.")
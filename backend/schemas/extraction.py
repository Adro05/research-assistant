"""
Pydantic schemas for PDF text extraction.

These describe the API response shape for the extraction endpoint. The
service layer (backend/services/extraction_service.py) works with plain
dataclasses; these schemas mirror that structure for the API boundary.
"""

from pydantic import BaseModel, Field


class ExtractedPageSchema(BaseModel):
    """A single extracted page, with page boundaries preserved."""

    page_number: int = Field(..., ge=1, description="1-indexed page number within the document.")
    text: str = Field(..., description="Extracted text for this page. Empty string if the page has no extractable text.")


class PDFExtractionResponse(BaseModel):
    """Response returned after successfully extracting text from a stored PDF."""

    document_id: str = Field(..., description="Identifier of the document that was extracted.")
    stored_filename: str = Field(..., description="Filename under which the source PDF is stored in data/uploads/.")
    total_pages: int = Field(..., ge=0, description="Total number of pages in the document.")
    pages: list[ExtractedPageSchema] = Field(..., description="Extracted text for each page, in page order.")
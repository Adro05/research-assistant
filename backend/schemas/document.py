"""
Pydantic schemas for document upload.

Only the upload response is defined here. Schemas for later processing
stages (text extraction, chunking, etc.) will be added in their own
milestones.
"""

from pydantic import BaseModel, Field


class PDFUploadResponse(BaseModel):
    """Response returned after a PDF has been validated and stored."""

    document_id: str = Field(
        ...,
        description="Server-generated unique identifier for this uploaded document.",
    )
    original_filename: str = Field(
        ...,
        description="Sanitized original filename as provided by the client (path components stripped).",
    )
    stored_filename: str = Field(
        ...,
        description="Filename under which the PDF was stored in data/uploads/.",
    )
    size_bytes: int = Field(
        ...,
        ge=0,
        description="Size of the uploaded file in bytes.",
    )
    upload_status: str = Field(
        ...,
        description="Status of the upload operation, e.g. 'success'.",
    )
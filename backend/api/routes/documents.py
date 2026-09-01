"""
Document upload endpoint.

Accepts a PDF upload, validates it, and stores it under data/uploads/.
This module does not perform text extraction — see the upload service for
the validation/storage logic itself.
"""

from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from backend.core.config import get_settings
from backend.schemas.document import PDFUploadResponse
from backend.services.upload_service import (
    UploadValidationError,
    sanitize_filename,
    save_pdf,
    validate_extension,
    validate_not_empty,
    validate_pdf_signature,
)


router = APIRouter(prefix="/documents", tags=["documents"])


def get_uploads_dir() -> Path:
    """
    FastAPI dependency resolving the upload storage directory.

    Exposed as a dependency (rather than read directly inside the route)
    so tests can override it to point at temporary storage instead of the
    real data/uploads/ directory.
    """
    return Path(get_settings().uploads_dir)


@router.post(
    "/upload",
    response_model=PDFUploadResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_pdf(
    file: UploadFile = File(...),
    uploads_dir: Path = Depends(get_uploads_dir),
) -> PDFUploadResponse:
    """Validate and store an uploaded PDF file."""
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No file was provided.",
        )

    try:
        safe_filename = sanitize_filename(file.filename)
        validate_extension(safe_filename)
    except UploadValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    content = await file.read()

    try:
        validate_not_empty(content)
        validate_pdf_signature(content)
    except UploadValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    try:
        stored = save_pdf(content, safe_filename, uploads_dir)
    except UploadValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to store the uploaded file.",
        ) from exc

    return PDFUploadResponse(
        document_id=stored.document_id,
        original_filename=stored.original_filename,
        stored_filename=stored.stored_filename,
        size_bytes=stored.size_bytes,
        upload_status="success",
    )
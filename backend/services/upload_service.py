"""
PDF upload validation and storage.

This module only handles accepting and safely persisting a raw PDF file.
It intentionally does not read, parse, or extract anything from the PDF's
contents beyond the leading signature bytes needed for validation — text
extraction is a separate, later milestone.
"""

import uuid
from dataclasses import dataclass
from pathlib import Path

PDF_MAGIC_BYTES = b"%PDF-"
ALLOWED_EXTENSION = ".pdf"


class UploadValidationError(Exception):
    """Raised when an uploaded file fails validation or cannot be stored."""


@dataclass
class StoredPDF:
    """Metadata describing a successfully stored PDF."""

    document_id: str
    original_filename: str
    stored_filename: str
    stored_path: Path
    size_bytes: int


def sanitize_filename(filename: str) -> str:
    """
    Reduce a client-supplied filename to a safe base filename.

    Strips any directory components (whether using '/' or '\\' separators)
    so that '../', absolute paths, and other path-traversal attempts can
    never influence where a file is read from or written to. The result is
    only used for display in API responses — it is never used to build a
    filesystem path for storage.
    """
    name = filename.replace("\\", "/").split("/")[-1].strip()
    if not name or name in {".", ".."}:
        raise UploadValidationError("Invalid or missing filename.")
    return name


def validate_extension(filename: str) -> None:
    """Reject filenames that do not end in .pdf (case-insensitive)."""
    if not filename.lower().endswith(ALLOWED_EXTENSION):
        raise UploadValidationError("Only .pdf files are accepted.")


def validate_not_empty(content: bytes) -> None:
    """Reject zero-byte uploads."""
    if len(content) == 0:
        raise UploadValidationError("Uploaded file is empty.")


def validate_pdf_signature(content: bytes) -> None:
    """Reject content that does not start with the PDF magic bytes."""
    if not content.startswith(PDF_MAGIC_BYTES):
        raise UploadValidationError("File is not a valid PDF (signature mismatch).")


def save_pdf(content: bytes, original_filename: str, uploads_dir: Path) -> StoredPDF:
    """
    Persist validated PDF content under `uploads_dir` using a generated,
    collision-free filename.

    Writes to a temporary file first and renames it into place only after
    the write succeeds, so a failed or interrupted write never leaves a
    partial file under its final name. On failure, any temporary file is
    removed and an UploadValidationError is raised.
    """
    uploads_dir.mkdir(parents=True, exist_ok=True)

    document_id = str(uuid.uuid4())
    stored_filename = f"{document_id}.pdf"
    stored_path = uploads_dir / stored_filename

    # Practically unreachable given uuid4, but never silently overwrite.
    while stored_path.exists():
        document_id = str(uuid.uuid4())
        stored_filename = f"{document_id}.pdf"
        stored_path = uploads_dir / stored_filename

    tmp_path = stored_path.with_suffix(".part")
    try:
        tmp_path.write_bytes(content)
        tmp_path.rename(stored_path)
    except OSError as exc:
        tmp_path.unlink(missing_ok=True)
        raise UploadValidationError("Failed to store the uploaded file.") from exc

    return StoredPDF(
        document_id=document_id,
        original_filename=original_filename,
        stored_filename=stored_filename,
        stored_path=stored_path,
        size_bytes=len(content),
    )
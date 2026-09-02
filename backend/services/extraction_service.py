"""
PDF text extraction.

Extracts text page-by-page from a PDF that was already uploaded and
stored by the upload stage (backend/services/upload_service.py). This
module is intentionally independent from upload/validation and from any
later chunking/RAG stage — it only turns a stored PDF into structured,
page-bounded text.

No OCR is performed; only the PDF's existing text layer is read via
PyMuPDF's `page.get_text()`.
"""

import re
from dataclasses import dataclass
from pathlib import Path

import pymupdf

# Canonical lowercase-hyphenated UUID form, e.g. "b2a1c9d4-1234-4abc-8def-0123456789ab".
# Deliberately strict: anything that doesn't match exactly this shape (including
# valid-but-differently-formatted UUIDs, path separators, or extra characters)
# is rejected before it is ever used to build a filesystem path.
_UUID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)


class ExtractionError(Exception):
    """Base class for extraction failures."""


class InvalidDocumentIdError(ExtractionError):
    """Raised when a document identifier is not a well-formed UUID."""


class DocumentNotFoundError(ExtractionError):
    """Raised when no stored PDF exists for the given document identifier."""


class PDFExtractionError(ExtractionError):
    """Raised when a stored file exists but cannot be read as a valid PDF."""


@dataclass
class ExtractedPage:
    """Extracted text for a single page, with its page number preserved."""

    page_number: int  # 1-indexed
    text: str


@dataclass
class PDFExtractionResult:
    """Structured, page-bounded extraction result for one document."""

    document_id: str
    stored_filename: str
    total_pages: int
    pages: list[ExtractedPage]


def validate_document_id(document_id: str) -> None:
    """
    Validate that `document_id` is a well-formed UUID in canonical form.

    This must be called before `document_id` is used to build any
    filesystem path. Rejects anything containing path separators, '..',
    or other characters outside the fixed hex/hyphen UUID shape, so a
    malicious or malformed identifier can never cause a lookup outside
    the uploads directory.
    """
    if not _UUID_PATTERN.match(document_id):
        raise InvalidDocumentIdError("Invalid document identifier format.")


def extract_pdf_text(document_id: str, uploads_dir: Path) -> PDFExtractionResult:
    """
    Extract page-by-page text from the stored PDF identified by `document_id`.

    Raises:
        InvalidDocumentIdError: `document_id` is not a well-formed UUID.
        DocumentNotFoundError: no stored PDF exists for this identifier.
        PDFExtractionError: the stored file exists but cannot be opened
            or read as a valid PDF.
    """
    validate_document_id(document_id)

    stored_filename = f"{document_id}.pdf"
    stored_path = uploads_dir / stored_filename

    if not stored_path.is_file():
        raise DocumentNotFoundError("No stored document was found for the given identifier.")

    try:
        content = stored_path.read_bytes()
    except OSError as exc:
        raise PDFExtractionError("Failed to read the stored document.") from exc

    try:
        doc = pymupdf.open(stream=content, filetype="pdf")
    except Exception as exc:  # PyMuPDF raises various error types for malformed input.
        raise PDFExtractionError("The stored document could not be read as a valid PDF.") from exc

    try:
        pages: list[ExtractedPage] = []
        for index, page in enumerate(doc, start=1):
            text = page.get_text()
            pages.append(ExtractedPage(page_number=index, text=text))
        total_pages = doc.page_count
    except Exception as exc:  # Guard against malformed PDFs that open but fail while reading pages.
        raise PDFExtractionError("Failed to extract text from the stored document.") from exc
    finally:
        doc.close()

    return PDFExtractionResult(
        document_id=document_id,
        stored_filename=stored_filename,
        total_pages=total_pages,
        pages=pages,
    )
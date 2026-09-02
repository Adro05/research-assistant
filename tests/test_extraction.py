"""
Tests for PDF text extraction.

These tests exercise extraction only — not upload/validation (covered in
test_upload.py) and not any later chunking/RAG stage, since none of that
exists yet.

Test PDFs are generated programmatically with PyMuPDF itself (no external
fixture files, no extra test-only dependency), written under pytest's
tmp_path, and never touch the real data/uploads/ directory.
"""

from pathlib import Path
from typing import Iterator

import pymupdf
import pytest
from fastapi.testclient import TestClient

from backend.api.routes.documents import get_uploads_dir
from backend.main import app
from backend.services.extraction_service import (
    DocumentNotFoundError,
    InvalidDocumentIdError,
    PDFExtractionError,
    extract_pdf_text,
    validate_document_id,
)

VALID_DOCUMENT_ID = "b2a1c9d4-1234-4abc-8def-0123456789ab"


def _make_pdf_bytes(page_texts: list[str]) -> bytes:
    """Build a small in-memory PDF with one page per entry in `page_texts`.

    An empty string produces a page with no inserted text (i.e. a
    genuinely blank page), rather than being skipped.
    """
    doc = pymupdf.open()
    for text in page_texts:
        page = doc.new_page()
        if text:
            page.insert_text((72, 72), text)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def _write_pdf(uploads_dir: Path, document_id: str, page_texts: list[str]) -> Path:
    """Write a generated PDF directly into uploads_dir under its stored filename."""
    stored_path = uploads_dir / f"{document_id}.pdf"
    stored_path.write_bytes(_make_pdf_bytes(page_texts))
    return stored_path


@pytest.fixture
def uploads_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "uploads"
    directory.mkdir()
    return directory


@pytest.fixture
def client(uploads_dir: Path) -> Iterator[TestClient]:
    app.dependency_overrides[get_uploads_dir] = lambda: uploads_dir
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.pop(get_uploads_dir, None)


# ---------------------------------------------------------------------------
# Service-level tests
# ---------------------------------------------------------------------------


def test_single_page_extraction(uploads_dir: Path) -> None:
    _write_pdf(uploads_dir, VALID_DOCUMENT_ID, ["Hello, research world."])

    result = extract_pdf_text(VALID_DOCUMENT_ID, uploads_dir)

    assert result.document_id == VALID_DOCUMENT_ID
    assert result.stored_filename == f"{VALID_DOCUMENT_ID}.pdf"
    assert result.total_pages == 1
    assert len(result.pages) == 1
    assert result.pages[0].page_number == 1
    assert "Hello, research world." in result.pages[0].text


def test_multi_page_extraction_preserves_page_boundaries(uploads_dir: Path) -> None:
    _write_pdf(uploads_dir, VALID_DOCUMENT_ID, ["Page one content.", "Page two content.", "Page three content."])

    result = extract_pdf_text(VALID_DOCUMENT_ID, uploads_dir)

    assert result.total_pages == 3
    assert [p.page_number for p in result.pages] == [1, 2, 3]
    assert "Page one content." in result.pages[0].text
    assert "Page two content." in result.pages[1].text
    assert "Page three content." in result.pages[2].text
    # Each page's text must be distinct and not bleed into another page.
    assert "Page two" not in result.pages[0].text
    assert "Page three" not in result.pages[1].text


def test_empty_page_is_represented_with_empty_text(uploads_dir: Path) -> None:
    _write_pdf(uploads_dir, VALID_DOCUMENT_ID, ["Some text here.", ""])

    result = extract_pdf_text(VALID_DOCUMENT_ID, uploads_dir)

    assert result.total_pages == 2
    assert len(result.pages) == 2
    assert result.pages[1].page_number == 2
    assert result.pages[1].text.strip() == ""


def test_missing_document_raises_not_found(uploads_dir: Path) -> None:
    with pytest.raises(DocumentNotFoundError):
        extract_pdf_text(VALID_DOCUMENT_ID, uploads_dir)


def test_malformed_pdf_raises_extraction_error(uploads_dir: Path) -> None:
    stored_path = uploads_dir / f"{VALID_DOCUMENT_ID}.pdf"
    stored_path.write_bytes(b"%PDF-1.4\nthis is not actually a valid pdf body")

    with pytest.raises(PDFExtractionError):
        extract_pdf_text(VALID_DOCUMENT_ID, uploads_dir)


@pytest.mark.parametrize(
    "malicious_id",
    [
        "../../etc/passwd",
        "../secret",
        "..",
        "abc/def",
        "abc\\def",
        "b2a1c9d4-1234-4abc-8def-0123456789ab/../../etc/passwd",
        "not-a-uuid-at-all",
        "",
        "B2A1C9D4-1234-4ABC-8DEF-0123456789AB",  # uppercase: rejected, must match canonical lowercase form
    ],
)
def test_invalid_document_id_is_rejected_before_touching_filesystem(
    malicious_id: str, uploads_dir: Path
) -> None:
    with pytest.raises(InvalidDocumentIdError):
        validate_document_id(malicious_id)

    # extract_pdf_text must also reject it up front, without ever
    # constructing or reading a path outside uploads_dir.
    with pytest.raises(InvalidDocumentIdError):
        extract_pdf_text(malicious_id, uploads_dir)

    # Confirm nothing was created inside uploads_dir either.
    assert list(uploads_dir.iterdir()) == []


# ---------------------------------------------------------------------------
# API-level tests
# ---------------------------------------------------------------------------


def test_api_successful_extraction_returns_expected_structure(client: TestClient, uploads_dir: Path) -> None:
    _write_pdf(uploads_dir, VALID_DOCUMENT_ID, ["First page.", "Second page."])

    response = client.post(f"/documents/{VALID_DOCUMENT_ID}/extract")

    assert response.status_code == 200
    body = response.json()
    assert body["document_id"] == VALID_DOCUMENT_ID
    assert body["stored_filename"] == f"{VALID_DOCUMENT_ID}.pdf"
    assert body["total_pages"] == 2
    assert len(body["pages"]) == 2
    assert body["pages"][0]["page_number"] == 1
    assert "First page." in body["pages"][0]["text"]
    assert body["pages"][1]["page_number"] == 2
    assert "Second page." in body["pages"][1]["text"]


def test_api_missing_document_returns_404(client: TestClient) -> None:
    response = client.post(f"/documents/{VALID_DOCUMENT_ID}/extract")

    assert response.status_code == 404
    # No internal filesystem paths should leak into the error response.
    assert "uploads" not in response.text
    assert "/home" not in response.text
    assert "\\" not in response.text


def test_api_malformed_pdf_returns_422(client: TestClient, uploads_dir: Path) -> None:
    stored_path = uploads_dir / f"{VALID_DOCUMENT_ID}.pdf"
    stored_path.write_bytes(b"%PDF-1.4\nnot a valid pdf body")

    response = client.post(f"/documents/{VALID_DOCUMENT_ID}/extract")

    assert response.status_code == 422


def test_api_invalid_document_id_returns_400_and_stays_inside_uploads_dir(
    client: TestClient, uploads_dir: Path
) -> None:
    response = client.post("/documents/not-a-uuid/extract")

    assert response.status_code == 400
    assert list(uploads_dir.iterdir()) == []


def test_upload_endpoint_still_behaves_as_before(client: TestClient, uploads_dir: Path) -> None:
    """Regression guard: Milestone 2.1's upload endpoint must be unaffected."""
    valid_pdf_bytes = _make_pdf_bytes(["Unrelated upload check."])

    response = client.post(
        "/documents/upload",
        files={"file": ("paper.pdf", valid_pdf_bytes, "application/pdf")},
    )

    assert response.status_code == 201
    body = response.json()
    assert body["upload_status"] == "success"
    assert (uploads_dir / body["stored_filename"]).exists()
"""
Tests for the PDF upload/validation endpoint.

These tests exercise only upload + validation + storage behavior. They do
not test text extraction, chunking, or any later pipeline stage, since
none of that exists yet.

Uploads are written to a pytest tmp_path directory rather than the real
data/uploads/, via a FastAPI dependency override.
"""

from pathlib import Path
from typing import Iterator

import pytest
from fastapi.testclient import TestClient

from backend.api.routes.documents import get_uploads_dir
from backend.main import app

VALID_PDF_BYTES = b"%PDF-1.4\n1 0 obj\n<< >>\nendobj\ntrailer\n<< >>\n%%EOF"


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


def test_valid_pdf_upload_is_stored(client: TestClient, uploads_dir: Path) -> None:
    response = client.post(
        "/documents/upload",
        files={"file": ("paper.pdf", VALID_PDF_BYTES, "application/pdf")},
    )

    assert response.status_code == 201
    body = response.json()
    assert body["original_filename"] == "paper.pdf"
    assert body["upload_status"] == "success"
    assert body["size_bytes"] == len(VALID_PDF_BYTES)
    assert body["document_id"]
    assert body["stored_filename"].endswith(".pdf")

    stored_file = uploads_dir / body["stored_filename"]
    assert stored_file.exists()
    assert stored_file.read_bytes() == VALID_PDF_BYTES


def test_empty_file_is_rejected_and_nothing_is_stored(client: TestClient, uploads_dir: Path) -> None:
    response = client.post(
        "/documents/upload",
        files={"file": ("empty.pdf", b"", "application/pdf")},
    )

    assert response.status_code == 400
    assert list(uploads_dir.iterdir()) == []


def test_wrong_extension_is_rejected_and_nothing_is_stored(client: TestClient, uploads_dir: Path) -> None:
    response = client.post(
        "/documents/upload",
        files={"file": ("notes.txt", VALID_PDF_BYTES, "text/plain")},
    )

    assert response.status_code == 400
    assert list(uploads_dir.iterdir()) == []


def test_fake_pdf_with_bad_signature_is_rejected_and_nothing_is_stored(
    client: TestClient, uploads_dir: Path
) -> None:
    response = client.post(
        "/documents/upload",
        files={"file": ("fake.pdf", b"this is definitely not a pdf", "application/pdf")},
    )

    assert response.status_code == 400
    assert list(uploads_dir.iterdir()) == []


def test_path_traversal_filename_is_sanitized(client: TestClient, uploads_dir: Path) -> None:
    response = client.post(
        "/documents/upload",
        files={"file": ("../../etc/passwd.pdf", VALID_PDF_BYTES, "application/pdf")},
    )

    assert response.status_code == 201
    body = response.json()

    # The returned filename must be reduced to a safe base name.
    assert body["original_filename"] == "passwd.pdf"
    assert ".." not in body["stored_filename"]
    assert "/" not in body["stored_filename"]

    # Only one file should exist, and it must be inside uploads_dir.
    stored_files = list(uploads_dir.iterdir())
    assert len(stored_files) == 1
    assert stored_files[0].parent == uploads_dir


def test_repeated_uploads_get_unique_stored_filenames(client: TestClient, uploads_dir: Path) -> None:
    files = {"file": ("paper.pdf", VALID_PDF_BYTES, "application/pdf")}

    first = client.post("/documents/upload", files=files)
    second = client.post("/documents/upload", files=files)

    assert first.status_code == 201
    assert second.status_code == 201
    assert first.json()["document_id"] != second.json()["document_id"]
    assert first.json()["stored_filename"] != second.json()["stored_filename"]
    assert len(list(uploads_dir.iterdir())) == 2


def test_missing_file_field_is_rejected(client: TestClient, uploads_dir: Path) -> None:
    response = client.post("/documents/upload")

    assert response.status_code == 422
    assert list(uploads_dir.iterdir()) == []
"""
Tests for text embedding (Milestone 2.4).

Most tests use a deterministic, hash-based FakeEmbeddingModel rather than
a real Hugging Face model: this keeps the suite fast, fully offline, and
independent of any model download, while still exercising all the real
logic in embedding_service.py (batching, ordering, provenance, error
handling). This mirrors the existing test conventions in this project
(tmp_path storage, dependency overrides, no real network/filesystem
dependencies beyond what's explicitly under test).

One additional smoke test attempts to load and use the real configured
Hugging Face model; it is skipped (not failed) if the model can't be
downloaded, so it remains a genuine integration check wherever network
access to huggingface.co is available, without making the suite flaky
where it isn't.
"""

from pathlib import Path
from typing import Iterator

import pymupdf
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from backend.api.routes.documents import get_embedding_model_instance, get_uploads_dir
from backend.core.config import get_settings
from backend.main import app
from backend.services.chunking_service import ChunkingResult, TextChunk, chunk_document
from backend.services.embedding_service import (
    EmbeddingGenerationError,
    embed_chunking_result,
    embed_texts,
    get_embedding_model,
)
from backend.services.extraction_service import ExtractedPage, PDFExtractionResult

VALID_DOCUMENT_ID = "b2a1c9d4-1234-4abc-8def-0123456789ab"


class FakeEmbeddingModel:
    """
    Deterministic, network-free stand-in for a real embedding model.

    Produces a fixed-dimension vector derived from a hash of each input
    text, so the same text always yields the same vector (mirroring how a
    real model is deterministic at inference), without needing any actual
    model weights. Also records the size of every `encode()` call so
    tests can verify batching behavior.
    """

    def __init__(self, dimension: int = 8) -> None:
        self.dimension = dimension
        self.calls: list[int] = []  # records len(texts) for each encode() call

    def encode(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(len(texts))
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        import hashlib

        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [b / 255.0 for b in digest[: self.dimension]]


class FailingEmbeddingModel:
    """A fake model that always raises, to test failure handling."""

    def encode(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("simulated model failure")


class InconsistentDimensionModel:
    """A fake model that returns vectors of varying length, to test the dimensionality guard."""

    def encode(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * (i + 1) for i in range(len(texts))]


def _build_chunking_result(chunks_data: list[tuple[str, int, int, int, str]]) -> ChunkingResult:
    """Build a ChunkingResult from (document_id, page_number, chunk_index, chunk_index_in_page, text) tuples."""
    chunks = [
        TextChunk(
            document_id=doc_id,
            page_number=page,
            chunk_index=idx,
            chunk_index_in_page=idx_in_page,
            text=text,
            char_count=len(text),
        )
        for doc_id, page, idx, idx_in_page, text in chunks_data
    ]
    return ChunkingResult(
        document_id=chunks_data[0][0] if chunks_data else VALID_DOCUMENT_ID,
        total_pages=max((c[1] for c in chunks_data), default=0),
        total_chunks=len(chunks),
        chunks=chunks,
    )


# ---------------------------------------------------------------------------
# embed_texts: core embedding generation
# ---------------------------------------------------------------------------


def test_embedding_generation_produces_one_vector_per_text() -> None:
    model = FakeEmbeddingModel(dimension=8)
    texts = ["first chunk", "second chunk", "third chunk"]

    vectors = embed_texts(texts, model, batch_size=32)

    assert len(vectors) == 3
    assert all(isinstance(v, list) for v in vectors)


def test_vector_dimensionality_matches_model() -> None:
    model = FakeEmbeddingModel(dimension=16)
    vectors = embed_texts(["some text"], model, batch_size=32)

    assert len(vectors[0]) == 16


def test_embedding_is_deterministic_for_same_text() -> None:
    model = FakeEmbeddingModel(dimension=8)

    first = embed_texts(["repeated text"], model, batch_size=32)
    second = embed_texts(["repeated text"], model, batch_size=32)

    assert first == second


def test_different_texts_produce_different_vectors() -> None:
    model = FakeEmbeddingModel(dimension=8)

    vectors = embed_texts(["alpha text", "beta text"], model, batch_size=32)

    assert vectors[0] != vectors[1]


def test_empty_input_produces_no_vectors_and_does_not_call_model() -> None:
    model = FakeEmbeddingModel()

    vectors = embed_texts([], model, batch_size=32)

    assert vectors == []
    assert model.calls == []  # model should never be invoked for empty input


def test_invalid_batch_size_raises_value_error() -> None:
    model = FakeEmbeddingModel()

    with pytest.raises(ValueError):
        embed_texts(["text"], model, batch_size=0)

    with pytest.raises(ValueError):
        embed_texts(["text"], model, batch_size=-5)


def test_model_failure_raises_embedding_generation_error() -> None:
    model = FailingEmbeddingModel()

    with pytest.raises(EmbeddingGenerationError):
        embed_texts(["text"], model, batch_size=32)


def test_inconsistent_vector_dimensions_raise_embedding_generation_error() -> None:
    model = InconsistentDimensionModel()

    with pytest.raises(EmbeddingGenerationError):
        embed_texts(["a", "bb", "ccc"], model, batch_size=32)


# ---------------------------------------------------------------------------
# embed_texts: batching behavior
# ---------------------------------------------------------------------------


def test_batching_splits_calls_according_to_batch_size() -> None:
    model = FakeEmbeddingModel()
    texts = [f"chunk {i}" for i in range(10)]

    embed_texts(texts, model, batch_size=3)

    # 10 texts at batch_size 3 -> batches of [3, 3, 3, 1]
    assert model.calls == [3, 3, 3, 1]


def test_batching_does_not_affect_output_or_ordering() -> None:
    texts = [f"unique chunk number {i}" for i in range(9)]

    single_batch_model = FakeEmbeddingModel()
    small_batch_model = FakeEmbeddingModel()

    single_batch_result = embed_texts(texts, single_batch_model, batch_size=100)
    small_batch_result = embed_texts(texts, small_batch_model, batch_size=2)

    assert single_batch_result == small_batch_result
    assert single_batch_model.calls == [9]
    assert small_batch_model.calls == [2, 2, 2, 2, 1]


def test_batch_size_exactly_matching_input_size_is_a_single_call() -> None:
    model = FakeEmbeddingModel()
    texts = ["a", "b", "c", "d"]

    embed_texts(texts, model, batch_size=4)

    assert model.calls == [4]


# ---------------------------------------------------------------------------
# embed_chunking_result: provenance, ordering, empty handling
# ---------------------------------------------------------------------------


def test_embed_chunking_result_preserves_order_and_provenance() -> None:
    chunking_result = _build_chunking_result(
        [
            (VALID_DOCUMENT_ID, 1, 0, 0, "first chunk text"),
            (VALID_DOCUMENT_ID, 1, 1, 1, "second chunk text"),
            (VALID_DOCUMENT_ID, 2, 2, 0, "third chunk text"),
        ]
    )
    model = FakeEmbeddingModel(dimension=8)

    result = embed_chunking_result(chunking_result, model=model, model_name="fake-model", batch_size=32)

    assert result.total_chunks == 3
    assert result.embedding_dimension == 8
    assert result.model_name == "fake-model"

    for original_chunk, embedded in zip(chunking_result.chunks, result.embeddings):
        assert embedded.document_id == original_chunk.document_id
        assert embedded.page_number == original_chunk.page_number
        assert embedded.chunk_index == original_chunk.chunk_index
        assert embedded.chunk_index_in_page == original_chunk.chunk_index_in_page
        assert embedded.text == original_chunk.text
        assert len(embedded.embedding) == 8


def test_embed_chunking_result_with_zero_chunks_is_not_an_error() -> None:
    empty_result = ChunkingResult(document_id=VALID_DOCUMENT_ID, total_pages=1, total_chunks=0, chunks=[])
    model = FakeEmbeddingModel()

    result = embed_chunking_result(empty_result, model=model, model_name="fake-model", batch_size=32)

    assert result.total_chunks == 0
    assert result.embeddings == []
    assert result.embedding_dimension == 0
    assert model.calls == []  # model never invoked for zero chunks


def test_embed_chunking_result_uses_configured_defaults_when_not_overridden() -> None:
    """model_name and batch_size fall back to application settings when not provided."""
    chunking_result = _build_chunking_result([(VALID_DOCUMENT_ID, 1, 0, 0, "some text")])
    model = FakeEmbeddingModel(dimension=8)
    settings = get_settings()

    result = embed_chunking_result(chunking_result, model=model)

    assert result.model_name == settings.embedding_model_name


def test_embed_chunking_result_end_to_end_through_real_chunking() -> None:
    """Exercises the actual chunk_document -> embed_chunking_result pipeline."""
    extraction = PDFExtractionResult(
        document_id=VALID_DOCUMENT_ID,
        stored_filename=f"{VALID_DOCUMENT_ID}.pdf",
        total_pages=1,
        pages=[ExtractedPage(page_number=1, text=("Research content for embedding. " * 40).strip())],
    )
    chunking_result = chunk_document(extraction)
    model = FakeEmbeddingModel(dimension=8)

    result = embed_chunking_result(chunking_result, model=model, model_name="fake-model", batch_size=2)

    assert result.total_chunks == chunking_result.total_chunks
    assert len(result.embeddings) == len(chunking_result.chunks)
    for chunk, embedded in zip(chunking_result.chunks, result.embeddings):
        assert embedded.text == chunk.text
        assert embedded.chunk_index == chunk.chunk_index


# ---------------------------------------------------------------------------
# API-level tests
# ---------------------------------------------------------------------------


def _make_pdf_bytes(page_texts: list[str]) -> bytes:
    doc = pymupdf.open()
    for text in page_texts:
        page = doc.new_page()
        if text:
            page.insert_text((72, 72), text)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def _write_pdf(uploads_dir: Path, document_id: str, page_texts: list[str]) -> Path:
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
    app.dependency_overrides[get_embedding_model_instance] = lambda: FakeEmbeddingModel(dimension=8)
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.pop(get_uploads_dir, None)
    app.dependency_overrides.pop(get_embedding_model_instance, None)


def test_api_embeddings_endpoint_success(client: TestClient, uploads_dir: Path) -> None:
    page_one = ("First page content for embedding endpoint test. " * 30).strip()
    page_two = "Second page, quite short."
    _write_pdf(uploads_dir, VALID_DOCUMENT_ID, [page_one, page_two])

    response = client.post(f"/documents/{VALID_DOCUMENT_ID}/embeddings")

    assert response.status_code == 200
    body = response.json()
    assert body["document_id"] == VALID_DOCUMENT_ID
    assert body["total_chunks"] == len(body["embeddings"])
    assert body["total_chunks"] >= 2
    assert body["embedding_dimension"] == 8

    for embedding in body["embeddings"]:
        assert embedding["document_id"] == VALID_DOCUMENT_ID
        assert embedding["page_number"] in (1, 2)
        assert len(embedding["embedding"]) == 8

    indices = [e["chunk_index"] for e in body["embeddings"]]
    assert indices == list(range(len(indices)))


def test_api_embeddings_missing_document_returns_404(client: TestClient) -> None:
    response = client.post(f"/documents/{VALID_DOCUMENT_ID}/embeddings")

    assert response.status_code == 404
    assert "uploads" not in response.text
    assert "/home" not in response.text


def test_api_embeddings_invalid_document_id_returns_400(client: TestClient, uploads_dir: Path) -> None:
    response = client.post("/documents/not-a-uuid/embeddings")

    assert response.status_code == 400
    assert list(uploads_dir.iterdir()) == []


def test_api_embeddings_malformed_pdf_returns_422(client: TestClient, uploads_dir: Path) -> None:
    stored_path = uploads_dir / f"{VALID_DOCUMENT_ID}.pdf"
    stored_path.write_bytes(b"%PDF-1.4\nnot a valid pdf body")

    response = client.post(f"/documents/{VALID_DOCUMENT_ID}/embeddings")

    assert response.status_code == 422


def test_api_embeddings_model_failure_returns_500(client: TestClient, uploads_dir: Path) -> None:
    _write_pdf(uploads_dir, VALID_DOCUMENT_ID, ["Some content to embed."])
    app.dependency_overrides[get_embedding_model_instance] = lambda: FailingEmbeddingModel()

    response = client.post(f"/documents/{VALID_DOCUMENT_ID}/embeddings")

    assert response.status_code == 500


def test_upload_extract_and_chunks_endpoints_unaffected_by_embeddings_addition(
    client: TestClient, uploads_dir: Path
) -> None:
    """Regression guard: Milestones 2.1, 2.2, and 2.3 must still behave identically."""
    valid_pdf_bytes = _make_pdf_bytes(["Regression check content."])

    upload_response = client.post(
        "/documents/upload",
        files={"file": ("paper.pdf", valid_pdf_bytes, "application/pdf")},
    )
    assert upload_response.status_code == 201
    document_id = upload_response.json()["document_id"]

    extract_response = client.post(f"/documents/{document_id}/extract")
    assert extract_response.status_code == 200

    chunks_response = client.post(f"/documents/{document_id}/chunks")
    assert chunks_response.status_code == 200

    embeddings_response = client.post(f"/documents/{document_id}/embeddings")
    assert embeddings_response.status_code == 200
    assert embeddings_response.json()["total_chunks"] >= 1


# ---------------------------------------------------------------------------
# Dependency-resolution-time model failure -> HTTP 503
#
# Regression tests for a bug where a model-load failure inside
# get_embedding_model_instance() (a FastAPI dependency) occurred before
# the route body's try/except could run, so it surfaced as an unhandled
# 500 instead of the intended 503. Fixed by catching
# EmbeddingModelUnavailableError inside the dependency function itself.
# ---------------------------------------------------------------------------


def test_get_embedding_model_instance_converts_unavailable_error_to_http_503(monkeypatch) -> None:
    """
    Unit-level check of the dependency function in isolation: simulates a
    real model-load failure (as would happen if get_embedding_model()
    itself raised) and confirms it is caught and converted to
    HTTPException(503) rather than propagating as a raw exception.
    """
    import backend.api.routes.documents as documents_module
    from backend.services.embedding_service import EmbeddingModelUnavailableError

    def _raise_unavailable(model_name: str):
        raise EmbeddingModelUnavailableError("simulated model load failure")

    monkeypatch.setattr(documents_module, "get_embedding_model", _raise_unavailable)

    with pytest.raises(HTTPException) as exc_info:
        documents_module.get_embedding_model_instance()

    assert exc_info.value.status_code == 503


def test_api_embeddings_returns_503_when_model_unavailable_at_dependency_resolution(
    uploads_dir: Path, monkeypatch
) -> None:
    """
    Full end-to-end check through the actual FastAPI request cycle,
    without overriding get_embedding_model_instance itself, so the real
    dependency-resolution code path (including its own try/except) runs.
    Only the underlying model loader is monkeypatched to fail, exactly
    reproducing the originally reported bug scenario.
    """
    import backend.api.routes.documents as documents_module
    from backend.services.embedding_service import EmbeddingModelUnavailableError

    def _raise_unavailable(model_name: str):
        raise EmbeddingModelUnavailableError("simulated model load failure")

    monkeypatch.setattr(documents_module, "get_embedding_model", _raise_unavailable)

    _write_pdf(uploads_dir, VALID_DOCUMENT_ID, ["Some content to embed."])

    app.dependency_overrides[get_uploads_dir] = lambda: uploads_dir
    # Deliberately NOT overriding get_embedding_model_instance here, so the
    # real dependency function (with its own try/except) actually runs.
    app.dependency_overrides.pop(get_embedding_model_instance, None)

    try:
        with TestClient(app) as test_client:
            response = test_client.post(f"/documents/{VALID_DOCUMENT_ID}/embeddings")
    finally:
        app.dependency_overrides.pop(get_uploads_dir, None)

    assert response.status_code == 503


# ---------------------------------------------------------------------------
# Real model smoke test (skipped gracefully if unavailable, e.g. offline)
# ---------------------------------------------------------------------------


def test_real_huggingface_model_smoke_test() -> None:
    """
    Attempts to load and use the actual configured Hugging Face model.

    Skipped (not failed) if the model cannot be downloaded — e.g. no
    network access to huggingface.co — so this remains a genuine
    integration check in environments with network access, without
    making the suite fail in restricted/offline environments.
    """
    settings = get_settings()
    try:
        model = get_embedding_model(settings.embedding_model_name)
    except Exception as exc:
        pytest.skip(f"Real embedding model unavailable in this environment: {exc}")

    vectors = embed_texts(["This is a real embedding smoke test."], model, batch_size=1)

    assert len(vectors) == 1
    assert len(vectors[0]) > 0
    assert all(isinstance(v, float) for v in vectors[0])
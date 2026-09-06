"""
Tests for FAISS vector indexing (Milestone 2.5).

Service-level tests construct EmbeddingResult objects directly with known
vector values (no embedding model involved at all), giving precise
control for exact cosine-similarity assertions. API-level tests use a
small deterministic FakeEmbeddingModel, exactly like test_embedding.py,
so the full extract -> chunk -> embed -> index pipeline can be exercised
without any network access or real Hugging Face model.

All generated FAISS/metadata artifacts are written under pytest's
tmp_path fixtures only — the real data/vector_stores/ directory is never
touched by these tests.
"""

import json
from pathlib import Path
from typing import Iterator

import numpy as np
import pymupdf
import pytest
from fastapi.testclient import TestClient

from backend.api.routes.documents import (
    get_embedding_model_instance,
    get_uploads_dir,
    get_vector_stores_dir,
)
from backend.main import app
from backend.services.embedding_service import ChunkEmbedding, EmbeddingResult
from backend.services.vector_store_service import (
    EmptyEmbeddingsError,
    InconsistentDimensionError,
    InvalidIndexIdentifierError,
    VectorIndexNotFoundError,
    VectorStoreCorruptionError,
    VectorStoreError,
    build_vector_index,
    load_vector_index,
    save_vector_index,
)

VALID_DOCUMENT_ID = "b2a1c9d4-1234-4abc-8def-0123456789ab"
OTHER_DOCUMENT_ID = "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"


def _make_embedding_result(
    vectors: list[list[float]],
    document_id: str = VALID_DOCUMENT_ID,
    embedding_dimension: int | None = None,
) -> EmbeddingResult:
    """Build an EmbeddingResult directly from explicit vector values, with sequential provenance."""
    embeddings = [
        ChunkEmbedding(
            document_id=document_id,
            page_number=(i // 2) + 1,  # two chunks per page, arbitrary but deterministic
            chunk_index=i,
            chunk_index_in_page=i % 2,
            text=f"chunk text {i}",
            embedding=vector,
            model_name="test-model",
        )
        for i, vector in enumerate(vectors)
    ]
    dimension = embedding_dimension if embedding_dimension is not None else (len(vectors[0]) if vectors else 0)
    return EmbeddingResult(
        document_id=document_id,
        model_name="test-model",
        embedding_dimension=dimension,
        total_chunks=len(embeddings),
        embeddings=embeddings,
    )


# ---------------------------------------------------------------------------
# build_vector_index: core behavior
# ---------------------------------------------------------------------------


def test_empty_embeddings_raises_clear_error() -> None:
    empty_result = _make_embedding_result([])

    with pytest.raises(EmptyEmbeddingsError):
        build_vector_index(empty_result)


def test_single_vector_can_be_indexed() -> None:
    result = _make_embedding_result([[1.0, 0.0, 0.0, 0.0]])

    index, index_result = build_vector_index(result)

    assert index.ntotal == 1
    assert index_result.total_vectors == 1


def test_multiple_vectors_can_be_indexed() -> None:
    vectors = [[float(i), float(i + 1), 0.0, 0.0] for i in range(5)]
    result = _make_embedding_result(vectors)

    index, index_result = build_vector_index(result)

    assert index.ntotal == 5
    assert index_result.total_vectors == 5
    assert len(index_result.records) == 5


def test_vector_dimension_is_preserved() -> None:
    vectors = [[1.0, 2.0, 3.0, 4.0, 5.0], [5.0, 4.0, 3.0, 2.0, 1.0]]
    result = _make_embedding_result(vectors)

    index, index_result = build_vector_index(result)

    assert index.d == 5
    assert index_result.embedding_dimension == 5


def test_number_of_indexed_vectors_is_correct() -> None:
    vectors = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.0]]
    result = _make_embedding_result(vectors)

    _, index_result = build_vector_index(result)

    assert index_result.total_vectors == len(vectors)


def test_deterministic_faiss_position_to_chunk_mapping() -> None:
    vectors = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    result = _make_embedding_result(vectors)

    _, index_result = build_vector_index(result)

    for position, record in enumerate(index_result.records):
        assert record.faiss_position == position
        assert record.chunk_index == position  # matches insertion order from _make_embedding_result
        assert record.document_id == VALID_DOCUMENT_ID
        assert record.text == f"chunk text {position}"


def test_inconsistent_vector_dimensions_are_rejected() -> None:
    result = _make_embedding_result([[1.0, 0.0], [1.0, 0.0, 0.0]], embedding_dimension=2)

    with pytest.raises(InconsistentDimensionError):
        build_vector_index(result)


def test_declared_dimension_mismatch_is_rejected() -> None:
    """embedding_dimension field disagrees with the actual vector length."""
    result = _make_embedding_result([[1.0, 0.0, 0.0]], embedding_dimension=99)

    with pytest.raises(InconsistentDimensionError):
        build_vector_index(result)


def test_zero_dimensional_embeddings_are_rejected() -> None:
    """A vector of length 0 cannot be used to construct a FAISS index."""
    result = _make_embedding_result([[], []], embedding_dimension=0)

    with pytest.raises(InconsistentDimensionError):
        build_vector_index(result)


def test_total_chunks_mismatch_with_embeddings_list_is_rejected() -> None:
    """embedding_result.total_chunks must match len(embedding_result.embeddings)."""
    consistent_result = _make_embedding_result([[1.0, 0.0], [0.0, 1.0]])
    # Deliberately corrupt total_chunks to disagree with the actual embeddings list.
    inconsistent_result = EmbeddingResult(
        document_id=consistent_result.document_id,
        model_name=consistent_result.model_name,
        embedding_dimension=consistent_result.embedding_dimension,
        total_chunks=5,  # actual list below has only 2 entries
        embeddings=consistent_result.embeddings,
    )

    with pytest.raises(InconsistentDimensionError):
        build_vector_index(inconsistent_result)


# ---------------------------------------------------------------------------
# Cosine similarity correctness
# ---------------------------------------------------------------------------


def test_cosine_similarity_behavior_is_correct() -> None:
    """
    Vectors pointing in the same direction (regardless of magnitude) must
    have similarity ~1.0; orthogonal vectors ~0.0; opposite vectors ~-1.0.
    """
    same_direction_large_magnitude = [10.0, 0.0]  # same direction as [1, 0], different magnitude
    orthogonal = [0.0, 1.0]
    opposite = [-1.0, 0.0]
    vectors = [[1.0, 0.0], same_direction_large_magnitude, orthogonal, opposite]
    result = _make_embedding_result(vectors)

    index, _ = build_vector_index(result)

    import faiss

    query = np.array([[1.0, 0.0]], dtype="float32")
    faiss.normalize_L2(query)
    scores, indices = index.search(query, k=4)

    scores_by_position = {int(idx): float(score) for idx, score in zip(indices[0], scores[0])}

    assert scores_by_position[0] == pytest.approx(1.0, abs=1e-5)  # identical direction
    assert scores_by_position[1] == pytest.approx(1.0, abs=1e-5)  # same direction, different magnitude
    assert scores_by_position[2] == pytest.approx(0.0, abs=1e-5)  # orthogonal
    assert scores_by_position[3] == pytest.approx(-1.0, abs=1e-5)  # opposite


def test_magnitude_does_not_affect_similarity_ranking() -> None:
    """A vector scaled up should not be ranked differently than its unit-length equivalent."""
    small = [1.0, 1.0]
    large = [50.0, 50.0]  # same direction, much larger magnitude
    different_direction = [1.0, -1.0]
    result = _make_embedding_result([small, large, different_direction])

    index, _ = build_vector_index(result)

    import faiss

    query = np.array([[1.0, 1.0]], dtype="float32")
    faiss.normalize_L2(query)
    scores, indices = index.search(query, k=3)

    # Both `small` and `large` (same direction) should score ~1.0, well above `different_direction`.
    scores_by_position = {int(idx): float(score) for idx, score in zip(indices[0], scores[0])}
    assert scores_by_position[0] == pytest.approx(1.0, abs=1e-5)
    assert scores_by_position[1] == pytest.approx(1.0, abs=1e-5)
    assert scores_by_position[2] < 0.5


# ---------------------------------------------------------------------------
# Save / load round trip
# ---------------------------------------------------------------------------


@pytest.fixture
def vector_stores_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "vector_stores"
    directory.mkdir()
    return directory


def test_save_and_load_round_trip(vector_stores_dir: Path) -> None:
    vectors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    result = _make_embedding_result(vectors)
    index, index_result = build_vector_index(result)

    paths = save_vector_index(index, index_result, vector_stores_dir)

    assert (vector_stores_dir / paths.index_filename).is_file()
    assert (vector_stores_dir / paths.metadata_filename).is_file()

    loaded_index, loaded_result = load_vector_index(VALID_DOCUMENT_ID, vector_stores_dir)

    assert loaded_index.ntotal == index.ntotal
    assert loaded_index.d == index.d
    assert loaded_result.document_id == index_result.document_id
    assert loaded_result.index_type == index_result.index_type
    assert loaded_result.similarity_metric == index_result.similarity_metric
    assert loaded_result.embedding_dimension == index_result.embedding_dimension
    assert loaded_result.total_vectors == index_result.total_vectors
    assert loaded_result.records == index_result.records


def test_loaded_index_produces_same_search_results_as_original(vector_stores_dir: Path) -> None:
    vectors = [[1.0, 0.2], [0.3, 1.0], [-1.0, 0.1], [0.5, 0.5]]
    result = _make_embedding_result(vectors)
    index, index_result = build_vector_index(result)
    save_vector_index(index, index_result, vector_stores_dir)

    loaded_index, _ = load_vector_index(VALID_DOCUMENT_ID, vector_stores_dir)

    import faiss

    query = np.array([[0.9, 0.1]], dtype="float32")
    faiss.normalize_L2(query)

    original_scores, original_indices = index.search(query, k=4)
    loaded_scores, loaded_indices = loaded_index.search(query, k=4)

    assert np.array_equal(original_indices, loaded_indices)
    assert np.allclose(original_scores, loaded_scores, atol=1e-6)


def test_save_writes_only_expected_files(vector_stores_dir: Path) -> None:
    result = _make_embedding_result([[1.0, 0.0]])
    index, index_result = build_vector_index(result)

    save_vector_index(index, index_result, vector_stores_dir)

    written_files = sorted(p.name for p in vector_stores_dir.iterdir())
    assert written_files == sorted(
        [f"{VALID_DOCUMENT_ID}.faiss", f"{VALID_DOCUMENT_ID}.metadata.json"]
    )


def test_metadata_file_does_not_embed_raw_vectors_as_source_of_truth(vector_stores_dir: Path) -> None:
    """The JSON sidecar should carry provenance, not duplicate the FAISS vectors themselves."""
    result = _make_embedding_result([[1.0, 0.0], [0.0, 1.0]])
    index, index_result = build_vector_index(result)

    paths = save_vector_index(index, index_result, vector_stores_dir)
    metadata = json.loads((vector_stores_dir / paths.metadata_filename).read_text())

    for record in metadata["records"]:
        assert "embedding" not in record
        assert "vector" not in record


def test_repeated_save_replaces_existing_index_safely(vector_stores_dir: Path) -> None:
    """
    Saving again for the same document_id (e.g. re-indexing after an
    embedding-config change) must safely replace the existing pair, work
    identically on the second call, and leave no leftover temp/backup
    files. This specifically exercises Path.replace()'s cross-platform
    overwrite behavior (including Windows, where Path.rename() would
    raise FileExistsError instead).
    """
    first_result = _make_embedding_result([[1.0, 0.0]])
    first_index, first_index_result = build_vector_index(first_result)
    save_vector_index(first_index, first_index_result, vector_stores_dir)

    # Re-index the same document_id with a different vector count/dimension.
    second_result = _make_embedding_result(
        [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], embedding_dimension=3
    )
    second_index, second_index_result = build_vector_index(second_result)
    paths = save_vector_index(second_index, second_index_result, vector_stores_dir)

    loaded_index, loaded_result = load_vector_index(VALID_DOCUMENT_ID, vector_stores_dir)
    assert loaded_index.ntotal == 3
    assert loaded_index.d == 3
    assert loaded_result.total_vectors == 3
    assert loaded_result.embedding_dimension == 3

    # Only the two real artifact files should exist — no .part or .bak leftovers.
    remaining_files = sorted(p.name for p in vector_stores_dir.iterdir())
    assert remaining_files == sorted([paths.index_filename, paths.metadata_filename])


def test_failed_replacement_does_not_leave_mismatched_pair(vector_stores_dir: Path, monkeypatch) -> None:
    """
    If the metadata replace step fails partway through re-indexing (after
    the index replace already succeeded), the original index must be
    restored from its backup, so the stored pair is never left as a new
    index paired with stale/old metadata (or missing metadata).
    """
    first_result = _make_embedding_result([[1.0, 0.0]])
    first_index, first_index_result = build_vector_index(first_result)
    save_vector_index(first_index, first_index_result, vector_stores_dir)

    original_faiss_bytes = (vector_stores_dir / f"{VALID_DOCUMENT_ID}.faiss").read_bytes()
    original_metadata_text = (vector_stores_dir / f"{VALID_DOCUMENT_ID}.metadata.json").read_text()

    second_result = _make_embedding_result([[0.0, 1.0], [1.0, 1.0], [1.0, 0.0]])
    second_index, second_index_result = build_vector_index(second_result)

    original_replace = Path.replace

    def failing_replace(self: Path, target):
        if self.name.endswith(".metadata.json.part"):
            raise OSError("simulated metadata replace failure (e.g. Windows file lock)")
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", failing_replace)

    with pytest.raises(VectorStoreError):
        save_vector_index(second_index, second_index_result, vector_stores_dir)

    monkeypatch.undo()

    # The OLD pair must remain byte-for-byte intact.
    assert (vector_stores_dir / f"{VALID_DOCUMENT_ID}.faiss").read_bytes() == original_faiss_bytes
    assert (vector_stores_dir / f"{VALID_DOCUMENT_ID}.metadata.json").read_text() == original_metadata_text

    # And it must still load successfully, reflecting the ORIGINAL (first) index.
    loaded_index, loaded_result = load_vector_index(VALID_DOCUMENT_ID, vector_stores_dir)
    assert loaded_index.ntotal == 1
    assert loaded_result.total_vectors == 1

    # No leftover temporary or backup files.
    leftover_names = [p.name for p in vector_stores_dir.iterdir()]
    assert all(".part" not in name and ".bak" not in name for name in leftover_names)
    assert sorted(leftover_names) == sorted(
        [f"{VALID_DOCUMENT_ID}.faiss", f"{VALID_DOCUMENT_ID}.metadata.json"]
    )


# ---------------------------------------------------------------------------
# Missing / corrupt artifact handling
# ---------------------------------------------------------------------------


def test_load_missing_index_raises_not_found(vector_stores_dir: Path) -> None:
    with pytest.raises(VectorIndexNotFoundError):
        load_vector_index(OTHER_DOCUMENT_ID, vector_stores_dir)


def test_load_corrupt_faiss_file_raises_corruption_error(vector_stores_dir: Path) -> None:
    result = _make_embedding_result([[1.0, 0.0]])
    index, index_result = build_vector_index(result)
    save_vector_index(index, index_result, vector_stores_dir)

    # Corrupt the FAISS file itself after a valid save.
    (vector_stores_dir / f"{VALID_DOCUMENT_ID}.faiss").write_bytes(b"not a real faiss index")

    with pytest.raises(VectorStoreCorruptionError):
        load_vector_index(VALID_DOCUMENT_ID, vector_stores_dir)


def test_load_corrupt_metadata_raises_corruption_error(vector_stores_dir: Path) -> None:
    result = _make_embedding_result([[1.0, 0.0]])
    index, index_result = build_vector_index(result)
    save_vector_index(index, index_result, vector_stores_dir)

    # Corrupt the metadata file itself after a valid save.
    (vector_stores_dir / f"{VALID_DOCUMENT_ID}.metadata.json").write_text("{ this is not valid json")

    with pytest.raises(VectorStoreCorruptionError):
        load_vector_index(VALID_DOCUMENT_ID, vector_stores_dir)


def test_load_metadata_missing_required_keys_raises_corruption_error(vector_stores_dir: Path) -> None:
    result = _make_embedding_result([[1.0, 0.0]])
    index, index_result = build_vector_index(result)
    save_vector_index(index, index_result, vector_stores_dir)

    (vector_stores_dir / f"{VALID_DOCUMENT_ID}.metadata.json").write_text(json.dumps({"document_id": "x"}))

    with pytest.raises(VectorStoreCorruptionError):
        load_vector_index(VALID_DOCUMENT_ID, vector_stores_dir)


def _save_valid_index_and_load_metadata_dict(vector_stores_dir: Path, vectors: list[list[float]]) -> dict:
    """Helper: build+save a valid index, then return its metadata as a mutable dict for corruption tests."""
    result = _make_embedding_result(vectors)
    index, index_result = build_vector_index(result)
    save_vector_index(index, index_result, vector_stores_dir)
    metadata_path = vector_stores_dir / f"{VALID_DOCUMENT_ID}.metadata.json"
    return json.loads(metadata_path.read_text())


def _write_metadata(vector_stores_dir: Path, metadata: dict) -> None:
    (vector_stores_dir / f"{VALID_DOCUMENT_ID}.metadata.json").write_text(json.dumps(metadata))


def test_load_rejects_total_vectors_mismatch(vector_stores_dir: Path) -> None:
    metadata = _save_valid_index_and_load_metadata_dict(vector_stores_dir, [[1.0, 0.0], [0.0, 1.0]])
    metadata["total_vectors"] = 999  # actual FAISS index still has 2 vectors
    _write_metadata(vector_stores_dir, metadata)

    with pytest.raises(VectorStoreCorruptionError):
        load_vector_index(VALID_DOCUMENT_ID, vector_stores_dir)


def test_load_rejects_embedding_dimension_mismatch(vector_stores_dir: Path) -> None:
    metadata = _save_valid_index_and_load_metadata_dict(vector_stores_dir, [[1.0, 0.0], [0.0, 1.0]])
    metadata["embedding_dimension"] = 999  # actual FAISS index dimension is 2
    _write_metadata(vector_stores_dir, metadata)

    with pytest.raises(VectorStoreCorruptionError):
        load_vector_index(VALID_DOCUMENT_ID, vector_stores_dir)


def test_load_rejects_records_length_mismatch(vector_stores_dir: Path) -> None:
    metadata = _save_valid_index_and_load_metadata_dict(vector_stores_dir, [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    # Drop one record while total_vectors still claims 3 -> len(records) != index.ntotal.
    metadata["records"] = metadata["records"][:-1]
    _write_metadata(vector_stores_dir, metadata)

    with pytest.raises(VectorStoreCorruptionError):
        load_vector_index(VALID_DOCUMENT_ID, vector_stores_dir)


def test_load_rejects_faiss_position_mismatch(vector_stores_dir: Path) -> None:
    metadata = _save_valid_index_and_load_metadata_dict(vector_stores_dir, [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    # Swap faiss_position values between two records so they no longer match their list index.
    metadata["records"][0]["faiss_position"], metadata["records"][1]["faiss_position"] = (
        metadata["records"][1]["faiss_position"],
        metadata["records"][0]["faiss_position"],
    )
    _write_metadata(vector_stores_dir, metadata)

    with pytest.raises(VectorStoreCorruptionError):
        load_vector_index(VALID_DOCUMENT_ID, vector_stores_dir)


def test_load_rejects_document_id_mismatch(vector_stores_dir: Path) -> None:
    metadata = _save_valid_index_and_load_metadata_dict(vector_stores_dir, [[1.0, 0.0]])
    metadata["document_id"] = OTHER_DOCUMENT_ID  # disagrees with the requested/filename document_id
    _write_metadata(vector_stores_dir, metadata)

    with pytest.raises(VectorStoreCorruptionError):
        load_vector_index(VALID_DOCUMENT_ID, vector_stores_dir)


def test_load_rejects_unexpected_index_type(vector_stores_dir: Path) -> None:
    metadata = _save_valid_index_and_load_metadata_dict(vector_stores_dir, [[1.0, 0.0]])
    metadata["index_type"] = "IndexIVFFlat"  # not the index type this module actually builds
    _write_metadata(vector_stores_dir, metadata)

    with pytest.raises(VectorStoreCorruptionError):
        load_vector_index(VALID_DOCUMENT_ID, vector_stores_dir)


def test_load_rejects_unexpected_similarity_metric(vector_stores_dir: Path) -> None:
    metadata = _save_valid_index_and_load_metadata_dict(vector_stores_dir, [[1.0, 0.0]])
    metadata["similarity_metric"] = "euclidean"  # not the convention this module actually uses
    _write_metadata(vector_stores_dir, metadata)

    with pytest.raises(VectorStoreCorruptionError):
        load_vector_index(VALID_DOCUMENT_ID, vector_stores_dir)


# ---------------------------------------------------------------------------
# Path safety
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "malicious_id",
    ["../../etc/passwd", "../secret", "abc/def", "not-a-uuid", ""],
)
def test_invalid_document_id_rejected_on_save(malicious_id: str, vector_stores_dir: Path) -> None:
    result = _make_embedding_result([[1.0, 0.0]], document_id=malicious_id)
    index, index_result = build_vector_index(result)

    with pytest.raises(InvalidIndexIdentifierError):
        save_vector_index(index, index_result, vector_stores_dir)

    assert list(vector_stores_dir.iterdir()) == []


@pytest.mark.parametrize(
    "malicious_id",
    ["../../etc/passwd", "../secret", "abc/def", "not-a-uuid", ""],
)
def test_invalid_document_id_rejected_on_load(malicious_id: str, vector_stores_dir: Path) -> None:
    with pytest.raises(InvalidIndexIdentifierError):
        load_vector_index(malicious_id, vector_stores_dir)


# ---------------------------------------------------------------------------
# API-level tests
# ---------------------------------------------------------------------------


class FakeEmbeddingModel:
    """Deterministic, network-free stand-in for a real embedding model (see test_embedding.py)."""

    def __init__(self, dimension: int = 8) -> None:
        self.dimension = dimension

    def encode(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        import hashlib

        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [b / 255.0 for b in digest[: self.dimension]]


class FailingEmbeddingModel:
    def encode(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("simulated model failure")


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
def client(uploads_dir: Path, vector_stores_dir: Path) -> Iterator[TestClient]:
    app.dependency_overrides[get_uploads_dir] = lambda: uploads_dir
    app.dependency_overrides[get_vector_stores_dir] = lambda: vector_stores_dir
    app.dependency_overrides[get_embedding_model_instance] = lambda: FakeEmbeddingModel(dimension=8)
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.pop(get_uploads_dir, None)
    app.dependency_overrides.pop(get_vector_stores_dir, None)
    app.dependency_overrides.pop(get_embedding_model_instance, None)


def test_api_index_endpoint_success(client: TestClient, uploads_dir: Path, vector_stores_dir: Path) -> None:
    page_one = ("Research content for the indexing endpoint test. " * 30).strip()
    _write_pdf(uploads_dir, VALID_DOCUMENT_ID, [page_one])

    response = client.post(f"/documents/{VALID_DOCUMENT_ID}/index")

    assert response.status_code == 200
    body = response.json()
    assert body["document_id"] == VALID_DOCUMENT_ID
    assert body["index_type"] == "IndexFlatIP"
    assert body["similarity_metric"] == "cosine"
    assert body["embedding_dimension"] == 8
    assert body["total_vectors"] >= 1
    assert body["index_filename"] == f"{VALID_DOCUMENT_ID}.faiss"
    assert body["metadata_filename"] == f"{VALID_DOCUMENT_ID}.metadata.json"

    assert (vector_stores_dir / body["index_filename"]).is_file()
    assert (vector_stores_dir / body["metadata_filename"]).is_file()


def test_api_index_missing_document_returns_404(client: TestClient) -> None:
    response = client.post(f"/documents/{VALID_DOCUMENT_ID}/index")

    assert response.status_code == 404


def test_api_index_invalid_document_id_returns_400(client: TestClient, vector_stores_dir: Path) -> None:
    response = client.post("/documents/not-a-uuid/index")

    assert response.status_code == 400
    assert list(vector_stores_dir.iterdir()) == []


def test_api_index_malformed_pdf_returns_422(client: TestClient, uploads_dir: Path) -> None:
    stored_path = uploads_dir / f"{VALID_DOCUMENT_ID}.pdf"
    stored_path.write_bytes(b"%PDF-1.4\nnot a valid pdf body")

    response = client.post(f"/documents/{VALID_DOCUMENT_ID}/index")

    assert response.status_code == 422


def test_api_index_zero_chunk_document_returns_422(
    client: TestClient, uploads_dir: Path, vector_stores_dir: Path
) -> None:
    """A PDF with a single, entirely blank page produces zero chunks -> nothing to index."""
    _write_pdf(uploads_dir, VALID_DOCUMENT_ID, [""])

    response = client.post(f"/documents/{VALID_DOCUMENT_ID}/index")

    assert response.status_code == 422
    assert list(vector_stores_dir.iterdir()) == []


def test_api_index_model_failure_returns_500(client: TestClient, uploads_dir: Path) -> None:
    _write_pdf(uploads_dir, VALID_DOCUMENT_ID, ["Some content to index."])
    app.dependency_overrides[get_embedding_model_instance] = lambda: FailingEmbeddingModel()

    response = client.post(f"/documents/{VALID_DOCUMENT_ID}/index")

    assert response.status_code == 500


def test_full_pipeline_regression_upload_through_index(
    client: TestClient, uploads_dir: Path, vector_stores_dir: Path
) -> None:
    """Regression guard: Milestones 2.1-2.4 must still behave identically alongside indexing."""
    valid_pdf_bytes = _make_pdf_bytes(["Regression pipeline content."])

    upload_response = client.post(
        "/documents/upload",
        files={"file": ("paper.pdf", valid_pdf_bytes, "application/pdf")},
    )
    assert upload_response.status_code == 201
    document_id = upload_response.json()["document_id"]

    assert client.post(f"/documents/{document_id}/extract").status_code == 200
    assert client.post(f"/documents/{document_id}/chunks").status_code == 200
    assert client.post(f"/documents/{document_id}/embeddings").status_code == 200

    index_response = client.post(f"/documents/{document_id}/index")
    assert index_response.status_code == 200
    assert index_response.json()["total_vectors"] >= 1
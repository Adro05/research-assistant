"""
Tests for retrieval (Milestone 2.6).

Service-level tests build FAISS indexes directly via the real
build_vector_index/save_vector_index from Milestone 2.5, and use a small
deterministic FakeEmbeddingModel for query embedding — no PDF, no
network, no real Hugging Face model required for the vast majority of
tests. One smoke test attempts the real configured model and skips
gracefully if it can't be downloaded, matching the convention already
established in test_embedding.py.

API-level tests exercise the full upload -> index -> retrieve pipeline
through TestClient, with dependency overrides for storage directories
and the embedding model, exactly like test_vector_store.py.
"""

import json
from pathlib import Path
from typing import Iterator

import faiss
import pymupdf
import pytest
from fastapi.testclient import TestClient

from backend.api.routes.documents import (
    get_embedding_model_instance,
    get_uploads_dir,
    get_vector_stores_dir,
)
from backend.core.config import get_settings
from backend.main import app
from backend.services.embedding_service import ChunkEmbedding, EmbeddingResult, embed_texts
from backend.services.retrieval_service import (
    EmbeddingDimensionMismatchError,
    EmbeddingModelMismatchError,
    EmptyQueryError,
    InvalidTopKError,
    RetrievalResult,
    retrieve,
)
from backend.services.vector_store_service import (
    VectorIndexResult,
    VectorStoreCorruptionError,
    build_vector_index,
    save_vector_index,
)

VALID_DOCUMENT_ID = "b2a1c9d4-1234-4abc-8def-0123456789ab"
OTHER_DOCUMENT_ID = "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"
FAKE_MODEL_NAME = "fake-model"


class FakeEmbeddingModel:
    """
    Deterministic, network-free stand-in for a real embedding model.

    Maps known query strings to known vectors (for precise ranking
    assertions) and falls back to a neutral vector for anything else.
    """

    def __init__(self, mapping: dict[str, list[float]] | None = None, dimension: int = 2) -> None:
        self.mapping = mapping or {}
        self.dimension = dimension
        self.calls: list[list[str]] = []

    def encode(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [self.mapping.get(text, [0.0] * self.dimension) for text in texts]


class FailingEmbeddingModel:
    def encode(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("simulated model failure")


def _make_embedding_result(
    vectors: list[list[float]],
    texts: list[str] | None = None,
    document_id: str = VALID_DOCUMENT_ID,
    model_name: str = FAKE_MODEL_NAME,
) -> EmbeddingResult:
    texts = texts or [f"chunk text {i}" for i in range(len(vectors))]
    embeddings = [
        ChunkEmbedding(
            document_id=document_id,
            page_number=(i // 2) + 1,
            chunk_index=i,
            chunk_index_in_page=i % 2,
            text=text,
            embedding=vector,
            model_name=model_name,
        )
        for i, (vector, text) in enumerate(zip(vectors, texts))
    ]
    return EmbeddingResult(
        document_id=document_id,
        model_name=model_name,
        embedding_dimension=len(vectors[0]) if vectors else 0,
        total_chunks=len(embeddings),
        embeddings=embeddings,
    )


@pytest.fixture
def vector_stores_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "vector_stores"
    directory.mkdir()
    return directory


def _build_and_save(vector_stores_dir: Path, vectors: list[list[float]], texts=None, model_name=FAKE_MODEL_NAME):
    result = _make_embedding_result(vectors, texts=texts, model_name=model_name)
    index, index_result = build_vector_index(result)
    save_vector_index(index, index_result, vector_stores_dir)
    return index_result


# ---------------------------------------------------------------------------
# Query embedding, normalization, ranking
# ---------------------------------------------------------------------------


def test_query_is_embedded_via_the_provided_model(vector_stores_dir: Path) -> None:
    _build_and_save(vector_stores_dir, [[1.0, 0.0], [0.0, 1.0]])
    model = FakeEmbeddingModel(mapping={"my query": [1.0, 0.0]})

    retrieve(VALID_DOCUMENT_ID, "my query", top_k=1, vector_stores_dir=vector_stores_dir, model=model, model_name=FAKE_MODEL_NAME)

    assert model.calls == [["my query"]]


def test_query_vector_is_normalized_before_search() -> None:
    """A query vector's magnitude must not affect ranking, only its direction."""
    query_vectors = embed_texts(["scaled query"], FakeEmbeddingModel(mapping={"scaled query": [50.0, 0.0]}), batch_size=1)
    assert query_vectors == [[50.0, 0.0]]  # embedding itself is un-normalized; normalization happens in retrieve()


def test_ranking_orders_results_by_descending_similarity(vector_stores_dir: Path) -> None:
    _build_and_save(
        vector_stores_dir,
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]],
        texts=["aligned", "orthogonal", "opposite"],
    )
    model = FakeEmbeddingModel(mapping={"query": [1.0, 0.0]})

    result = retrieve(VALID_DOCUMENT_ID, "query", top_k=3, vector_stores_dir=vector_stores_dir, model=model, model_name=FAKE_MODEL_NAME)

    ordered_texts = [r.text for r in result.results]
    assert ordered_texts == ["aligned", "orthogonal", "opposite"]
    scores = [r.score for r in result.results]
    assert scores == sorted(scores, reverse=True)
    assert result.results[0].score == pytest.approx(1.0, abs=1e-5)
    assert result.results[-1].score == pytest.approx(-1.0, abs=1e-5)


def test_query_magnitude_does_not_affect_ranking(vector_stores_dir: Path) -> None:
    _build_and_save(vector_stores_dir, [[1.0, 0.0], [0.0, 1.0]], texts=["aligned", "orthogonal"])
    large_magnitude_model = FakeEmbeddingModel(mapping={"query": [100.0, 0.0]})  # same direction, huge magnitude

    result = retrieve(VALID_DOCUMENT_ID, "query", top_k=2, vector_stores_dir=vector_stores_dir, model=large_magnitude_model, model_name=FAKE_MODEL_NAME)

    assert result.results[0].text == "aligned"
    assert result.results[0].score == pytest.approx(1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# top_k behavior
# ---------------------------------------------------------------------------


def test_top_k_limits_number_of_results(vector_stores_dir: Path) -> None:
    _build_and_save(vector_stores_dir, [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [-1.0, 0.0], [0.1, 0.9]])
    model = FakeEmbeddingModel(mapping={"query": [1.0, 0.0]})

    result = retrieve(VALID_DOCUMENT_ID, "query", top_k=2, vector_stores_dir=vector_stores_dir, model=model, model_name=FAKE_MODEL_NAME)

    assert result.total_results == 2
    assert len(result.results) == 2


def test_top_k_greater_than_total_vectors_is_clamped_not_an_error(vector_stores_dir: Path) -> None:
    _build_and_save(vector_stores_dir, [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    model = FakeEmbeddingModel(mapping={"query": [1.0, 0.0]})

    result = retrieve(VALID_DOCUMENT_ID, "query", top_k=100, vector_stores_dir=vector_stores_dir, model=model, model_name=FAKE_MODEL_NAME)

    assert result.total_results == 3
    assert len(result.results) == 3


def test_ranks_are_sequential_starting_at_one(vector_stores_dir: Path) -> None:
    _build_and_save(vector_stores_dir, [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    model = FakeEmbeddingModel(mapping={"query": [1.0, 0.0]})

    result = retrieve(VALID_DOCUMENT_ID, "query", top_k=3, vector_stores_dir=vector_stores_dir, model=model, model_name=FAKE_MODEL_NAME)

    assert [r.rank for r in result.results] == [1, 2, 3]


# ---------------------------------------------------------------------------
# Determinism and provenance mapping
# ---------------------------------------------------------------------------


def test_retrieval_is_deterministic(vector_stores_dir: Path) -> None:
    _build_and_save(vector_stores_dir, [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    model = FakeEmbeddingModel(mapping={"query": [0.9, 0.1]})

    first = retrieve(VALID_DOCUMENT_ID, "query", top_k=3, vector_stores_dir=vector_stores_dir, model=model, model_name=FAKE_MODEL_NAME)
    second = retrieve(VALID_DOCUMENT_ID, "query", top_k=3, vector_stores_dir=vector_stores_dir, model=model, model_name=FAKE_MODEL_NAME)

    assert first == second


def test_metadata_provenance_is_mapped_correctly(vector_stores_dir: Path) -> None:
    """Out-of-sequence chunk_index/page_number values rule out coincidental alignment."""
    index_result = _build_and_save(
        vector_stores_dir,
        [[1.0, 0.0], [0.0, 1.0]],
        texts=["first chunk", "second chunk"],
    )
    # Manually verify the underlying records carry distinguishable provenance.
    assert index_result.records[0].chunk_index == 0
    assert index_result.records[1].chunk_index == 1

    model = FakeEmbeddingModel(mapping={"query": [0.0, 1.0]})
    result = retrieve(VALID_DOCUMENT_ID, "query", top_k=1, vector_stores_dir=vector_stores_dir, model=model, model_name=FAKE_MODEL_NAME)

    top = result.results[0]
    assert top.text == "second chunk"
    assert top.chunk_index == 1
    assert top.document_id == VALID_DOCUMENT_ID
    assert top.model_name == FAKE_MODEL_NAME


# ---------------------------------------------------------------------------
# Empty query / invalid top_k
# ---------------------------------------------------------------------------


def test_empty_string_query_is_rejected(vector_stores_dir: Path) -> None:
    _build_and_save(vector_stores_dir, [[1.0, 0.0]])
    with pytest.raises(EmptyQueryError):
        retrieve(VALID_DOCUMENT_ID, "", top_k=1, vector_stores_dir=vector_stores_dir, model=FakeEmbeddingModel(), model_name=FAKE_MODEL_NAME)


def test_whitespace_only_query_is_rejected(vector_stores_dir: Path) -> None:
    _build_and_save(vector_stores_dir, [[1.0, 0.0]])
    with pytest.raises(EmptyQueryError):
        retrieve(VALID_DOCUMENT_ID, "   \n\t  ", top_k=1, vector_stores_dir=vector_stores_dir, model=FakeEmbeddingModel(), model_name=FAKE_MODEL_NAME)


@pytest.mark.parametrize("bad_k", [0, -1, -100])
def test_invalid_top_k_is_rejected(bad_k: int, vector_stores_dir: Path) -> None:
    _build_and_save(vector_stores_dir, [[1.0, 0.0]])
    with pytest.raises(InvalidTopKError):
        retrieve(VALID_DOCUMENT_ID, "query", top_k=bad_k, vector_stores_dir=vector_stores_dir, model=FakeEmbeddingModel(), model_name=FAKE_MODEL_NAME)


# ---------------------------------------------------------------------------
# Empty index (zero vectors) — explicit regression test
# ---------------------------------------------------------------------------


def test_empty_index_returns_successful_empty_result_without_calling_search(vector_stores_dir: Path) -> None:
    """
    An index with zero vectors is not an error: retrieve() must return an
    empty, successful result without ever calling FAISS search with k=0,
    and without even embedding the query (nothing to compare it to).
    """
    empty_faiss_index = faiss.IndexFlatIP(4)  # zero vectors added
    empty_index_result = VectorIndexResult(
        document_id=VALID_DOCUMENT_ID,
        index_type="IndexFlatIP",
        similarity_metric="cosine",
        embedding_dimension=4,
        total_vectors=0,
        records=[],
    )
    save_vector_index(empty_faiss_index, empty_index_result, vector_stores_dir)

    model = FakeEmbeddingModel(dimension=4)
    result = retrieve(VALID_DOCUMENT_ID, "anything at all", top_k=5, vector_stores_dir=vector_stores_dir, model=model, model_name=FAKE_MODEL_NAME)

    assert isinstance(result, RetrievalResult)
    assert result.total_results == 0
    assert result.results == []
    assert model.calls == []  # query was never embedded, since there was nothing to search


# ---------------------------------------------------------------------------
# Missing / corrupt index
# ---------------------------------------------------------------------------


def test_missing_index_raises_vector_index_not_found(vector_stores_dir: Path) -> None:
    from backend.services.vector_store_service import VectorIndexNotFoundError

    with pytest.raises(VectorIndexNotFoundError):
        retrieve(OTHER_DOCUMENT_ID, "query", top_k=1, vector_stores_dir=vector_stores_dir, model=FakeEmbeddingModel(), model_name=FAKE_MODEL_NAME)


def test_corrupted_index_metadata_raises_corruption_error(vector_stores_dir: Path) -> None:
    _build_and_save(vector_stores_dir, [[1.0, 0.0], [0.0, 1.0]])
    metadata_path = vector_stores_dir / f"{VALID_DOCUMENT_ID}.metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["total_vectors"] = 999  # inconsistent with actual FAISS index
    metadata_path.write_text(json.dumps(metadata))

    with pytest.raises(VectorStoreCorruptionError):
        retrieve(VALID_DOCUMENT_ID, "query", top_k=1, vector_stores_dir=vector_stores_dir, model=FakeEmbeddingModel(), model_name=FAKE_MODEL_NAME)


# ---------------------------------------------------------------------------
# Embedding model mismatch / dimension mismatch
# ---------------------------------------------------------------------------


def test_embedding_model_name_mismatch_is_rejected(vector_stores_dir: Path) -> None:
    _build_and_save(vector_stores_dir, [[1.0, 0.0]], model_name="model-a")

    with pytest.raises(EmbeddingModelMismatchError):
        retrieve(
            VALID_DOCUMENT_ID,
            "query",
            top_k=1,
            vector_stores_dir=vector_stores_dir,
            model=FakeEmbeddingModel(mapping={"query": [1.0, 0.0]}),
            model_name="model-b",  # different from the model the index was built with
        )


def test_embedding_dimension_mismatch_with_same_model_name_is_rejected(vector_stores_dir: Path) -> None:
    """
    Two different models could theoretically share the same declared
    name yet produce vectors of different length — the dimension check
    must be independent of (and not skipped just because) the name check
    passing.
    """
    _build_and_save(vector_stores_dir, [[1.0, 0.0]], model_name=FAKE_MODEL_NAME)  # 2-dim index
    wrong_dim_model = FakeEmbeddingModel(mapping={"query": [1.0, 0.0, 0.0]}, dimension=3)  # 3-dim output

    with pytest.raises(EmbeddingDimensionMismatchError):
        retrieve(
            VALID_DOCUMENT_ID,
            "query",
            top_k=1,
            vector_stores_dir=vector_stores_dir,
            model=wrong_dim_model,
            model_name=FAKE_MODEL_NAME,  # same name as the index, but dimension still differs
        )


def test_model_mismatch_takes_precedence_and_avoids_calling_model(vector_stores_dir: Path) -> None:
    """A model-name mismatch is detected before the query is ever embedded."""
    _build_and_save(vector_stores_dir, [[1.0, 0.0]], model_name="model-a")
    model = FakeEmbeddingModel(mapping={"query": [1.0, 0.0]})

    with pytest.raises(EmbeddingModelMismatchError):
        retrieve(VALID_DOCUMENT_ID, "query", top_k=1, vector_stores_dir=vector_stores_dir, model=model, model_name="model-b")

    assert model.calls == []  # never reached the embedding step


# ---------------------------------------------------------------------------
# Embedding generation failure
# ---------------------------------------------------------------------------


def test_embedding_generation_failure_propagates(vector_stores_dir: Path) -> None:
    _build_and_save(vector_stores_dir, [[1.0, 0.0]])
    from backend.services.embedding_service import EmbeddingGenerationError

    with pytest.raises(EmbeddingGenerationError):
        retrieve(VALID_DOCUMENT_ID, "query", top_k=1, vector_stores_dir=vector_stores_dir, model=FailingEmbeddingModel(), model_name=FAKE_MODEL_NAME)


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


class ApiFakeEmbeddingModel:
    """Deterministic model for API tests: hash-based, consistent with other test files."""

    def __init__(self, dimension: int = 8) -> None:
        self.dimension = dimension

    def encode(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> list[float]:
        import hashlib

        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [b / 255.0 for b in digest[: self.dimension]]


@pytest.fixture
def client(uploads_dir: Path, vector_stores_dir: Path) -> Iterator[TestClient]:
    app.dependency_overrides[get_uploads_dir] = lambda: uploads_dir
    app.dependency_overrides[get_vector_stores_dir] = lambda: vector_stores_dir
    app.dependency_overrides[get_embedding_model_instance] = lambda: ApiFakeEmbeddingModel(dimension=8)
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.pop(get_uploads_dir, None)
    app.dependency_overrides.pop(get_vector_stores_dir, None)
    app.dependency_overrides.pop(get_embedding_model_instance, None)


def test_api_retrieve_success(client: TestClient, uploads_dir: Path, vector_stores_dir: Path) -> None:
    page = ("Research content for the retrieval endpoint test. " * 30).strip()
    _write_pdf(uploads_dir, VALID_DOCUMENT_ID, [page])
    index_response = client.post(f"/documents/{VALID_DOCUMENT_ID}/index")
    assert index_response.status_code == 200

    response = client.post(f"/documents/{VALID_DOCUMENT_ID}/retrieve", json={"query": "research content", "top_k": 2})

    assert response.status_code == 200
    body = response.json()
    assert body["document_id"] == VALID_DOCUMENT_ID
    assert body["query"] == "research content"
    assert body["total_results"] == len(body["results"])
    assert body["total_results"] >= 1
    for result in body["results"]:
        assert result["document_id"] == VALID_DOCUMENT_ID
        assert "score" in result
        assert result["rank"] >= 1


def test_api_retrieve_invalid_document_id_returns_400(client: TestClient) -> None:
    response = client.post("/documents/not-a-uuid/retrieve", json={"query": "hello", "top_k": 3})

    assert response.status_code == 400


def test_api_retrieve_missing_index_returns_404(client: TestClient) -> None:
    response = client.post(f"/documents/{VALID_DOCUMENT_ID}/retrieve", json={"query": "hello", "top_k": 3})

    assert response.status_code == 404


def test_api_retrieve_corrupted_index_returns_422(client: TestClient, uploads_dir: Path, vector_stores_dir: Path) -> None:
    _write_pdf(uploads_dir, VALID_DOCUMENT_ID, ["Some content to index for corruption test."])
    assert client.post(f"/documents/{VALID_DOCUMENT_ID}/index").status_code == 200

    metadata_path = vector_stores_dir / f"{VALID_DOCUMENT_ID}.metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["embedding_dimension"] = 999
    metadata_path.write_text(json.dumps(metadata))

    response = client.post(f"/documents/{VALID_DOCUMENT_ID}/retrieve", json={"query": "hello", "top_k": 3})

    assert response.status_code == 422


def test_api_retrieve_fully_empty_query_returns_422_via_schema_validation(client: TestClient) -> None:
    """An empty string fails Pydantic's min_length=1 before the route body even runs."""
    response = client.post(f"/documents/{VALID_DOCUMENT_ID}/retrieve", json={"query": "", "top_k": 3})

    assert response.status_code == 422


def test_api_retrieve_whitespace_only_query_returns_400_via_service_check(
    client: TestClient, uploads_dir: Path
) -> None:
    """Whitespace-only text passes schema validation but is caught by the service's own check."""
    _write_pdf(uploads_dir, VALID_DOCUMENT_ID, ["Some content."])
    assert client.post(f"/documents/{VALID_DOCUMENT_ID}/index").status_code == 200

    response = client.post(f"/documents/{VALID_DOCUMENT_ID}/retrieve", json={"query": "   ", "top_k": 3})

    assert response.status_code == 400


def test_api_retrieve_invalid_top_k_returns_422_via_schema_validation(client: TestClient) -> None:
    response = client.post(f"/documents/{VALID_DOCUMENT_ID}/retrieve", json={"query": "hello", "top_k": 0})

    assert response.status_code == 422


def test_api_retrieve_model_unavailable_returns_503(uploads_dir: Path, vector_stores_dir: Path, monkeypatch) -> None:
    import backend.api.routes.documents as documents_module
    from backend.services.embedding_service import EmbeddingModelUnavailableError

    def _raise_unavailable(model_name: str):
        raise EmbeddingModelUnavailableError("simulated model load failure")

    monkeypatch.setattr(documents_module, "get_embedding_model", _raise_unavailable)

    app.dependency_overrides[get_uploads_dir] = lambda: uploads_dir
    app.dependency_overrides[get_vector_stores_dir] = lambda: vector_stores_dir
    app.dependency_overrides.pop(get_embedding_model_instance, None)  # let the real dependency run

    try:
        with TestClient(app) as test_client:
            response = test_client.post(f"/documents/{VALID_DOCUMENT_ID}/retrieve", json={"query": "hello", "top_k": 3})
    finally:
        app.dependency_overrides.pop(get_uploads_dir, None)
        app.dependency_overrides.pop(get_vector_stores_dir, None)

    assert response.status_code == 503


def test_api_retrieve_embedding_generation_failure_returns_500(
    client: TestClient, uploads_dir: Path
) -> None:
    _write_pdf(uploads_dir, VALID_DOCUMENT_ID, ["Some content."])
    assert client.post(f"/documents/{VALID_DOCUMENT_ID}/index").status_code == 200

    app.dependency_overrides[get_embedding_model_instance] = lambda: FailingEmbeddingModel()
    response = client.post(f"/documents/{VALID_DOCUMENT_ID}/retrieve", json={"query": "hello", "top_k": 3})

    assert response.status_code == 500


def test_api_retrieve_top_k_greater_than_indexed_vectors_returns_all_available(
    client: TestClient, uploads_dir: Path
) -> None:
    _write_pdf(uploads_dir, VALID_DOCUMENT_ID, ["A short single page of content."])
    assert client.post(f"/documents/{VALID_DOCUMENT_ID}/index").status_code == 200

    response = client.post(f"/documents/{VALID_DOCUMENT_ID}/retrieve", json={"query": "short content", "top_k": 1000})

    assert response.status_code == 200
    body = response.json()
    assert body["total_results"] < 1000
    assert body["total_results"] == len(body["results"])


# ---------------------------------------------------------------------------
# Regression: full pipeline through all six endpoints
# ---------------------------------------------------------------------------


def test_full_pipeline_regression_upload_through_retrieve(
    client: TestClient, uploads_dir: Path, vector_stores_dir: Path
) -> None:
    valid_pdf_bytes = _make_pdf_bytes(["Regression pipeline content for retrieval test."])

    upload_response = client.post(
        "/documents/upload",
        files={"file": ("paper.pdf", valid_pdf_bytes, "application/pdf")},
    )
    assert upload_response.status_code == 201
    document_id = upload_response.json()["document_id"]

    assert client.post(f"/documents/{document_id}/extract").status_code == 200
    assert client.post(f"/documents/{document_id}/chunks").status_code == 200
    assert client.post(f"/documents/{document_id}/embeddings").status_code == 200
    assert client.post(f"/documents/{document_id}/index").status_code == 200

    retrieve_response = client.post(f"/documents/{document_id}/retrieve", json={"query": "regression pipeline", "top_k": 3})
    assert retrieve_response.status_code == 200
    assert retrieve_response.json()["total_results"] >= 1


# ---------------------------------------------------------------------------
# Real model smoke test (skipped gracefully if unavailable, e.g. offline)
# ---------------------------------------------------------------------------


def test_real_huggingface_model_retrieval_smoke_test(vector_stores_dir: Path) -> None:
    """
    Attempts a real embedding model + retrieval end-to-end. Skipped (not
    failed) if the model cannot be downloaded, matching the convention in
    test_embedding.py.
    """
    from backend.services.embedding_service import get_embedding_model

    settings = get_settings()
    try:
        model = get_embedding_model(settings.embedding_model_name)
    except Exception as exc:
        pytest.skip(f"Real embedding model unavailable in this environment: {exc}")

    result = _make_embedding_result(
        embed_texts(["A sentence about cats.", "A sentence about finance."], model, batch_size=2),
        texts=["A sentence about cats.", "A sentence about finance."],
        model_name=settings.embedding_model_name,
    )
    index, index_result = build_vector_index(result)
    save_vector_index(index, index_result, vector_stores_dir)

    retrieval_result = retrieve(
        VALID_DOCUMENT_ID,
        "Tell me about kittens.",
        top_k=1,
        vector_stores_dir=vector_stores_dir,
        model=model,
        model_name=settings.embedding_model_name,
    )

    assert retrieval_result.total_results == 1
    assert "cats" in retrieval_result.results[0].text
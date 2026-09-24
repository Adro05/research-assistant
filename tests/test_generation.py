"""
Tests for generation (Milestone 2.7).

Service-level tests use a deterministic FakeGenerationModel implementing
the GenerationModel protocol (generate/count_tokens/max_input_tokens)
with a simple word-count tokenizer stand-in — no network, no real model
download required. API-level tests build a FAISS index directly (as in
test_retrieval.py) and use dependency overrides for storage dirs, the
embedding model, and the generation model.

One smoke test attempts the real configured Hugging Face generation
model (google/flan-t5-small) and skips gracefully if it can't be
downloaded, matching the convention in test_embedding.py/test_retrieval.py.
"""

import json
from pathlib import Path
from typing import Iterator

import pymupdf
import pytest
from fastapi.testclient import TestClient

from backend.api.routes.documents import (
    get_embedding_model_instance,
    get_generation_model_instance,
    get_uploads_dir,
    get_vector_stores_dir,
)
from backend.core.config import get_settings
from backend.main import app
from backend.services.generation_service import (
    EvidenceContextTooLargeError,
    GeneratedAnswer,
    GenerationFailedError,
    NoEvidenceError,
    _build_prompt,
    generate_answer,
)
from backend.services.retrieval_service import RetrievedChunk

VALID_DOCUMENT_ID = "b2a1c9d4-1234-4abc-8def-0123456789ab"
FAKE_EMBED_MODEL_NAME = "fake-embed-model"


class FakeGenerationModel:
    """
    Deterministic, network-free stand-in for a real generation model,
    implementing the GenerationModel protocol used by generation_service.

    Uses whitespace word count as a simple, fully deterministic stand-in
    for real subword tokenization, so prompt-budget tests can precisely
    control which evidence chunks fit by crafting texts with known word
    counts.
    """

    def __init__(self, max_input_tokens: int = 1000, response: str = "This is the generated answer.") -> None:
        self._max_input_tokens = max_input_tokens
        self.response = response
        self.prompts_received: list[str] = []

    @property
    def max_input_tokens(self) -> int:
        return self._max_input_tokens

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def generate(self, prompt: str) -> str:
        self.prompts_received.append(prompt)
        return self.response


class FailingGenerationModel:
    """A model that fails only at generation time (budgeting still works normally)."""

    max_input_tokens = 1000

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def generate(self, prompt: str) -> str:
        raise RuntimeError("simulated generation failure")


class EmptyAnswerGenerationModel:
    """A model whose generate() call succeeds but returns nothing usable."""

    max_input_tokens = 1000

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def generate(self, prompt: str) -> str:
        return "   "


def _make_evidence(texts: list[str], document_id: str = VALID_DOCUMENT_ID) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            rank=i + 1,
            score=1.0 - (i * 0.1),
            document_id=document_id,
            page_number=(i // 2) + 1,
            chunk_index=i,
            chunk_index_in_page=i % 2,
            text=text,
            model_name=FAKE_EMBED_MODEL_NAME,
        )
        for i, text in enumerate(texts)
    ]


# ---------------------------------------------------------------------------
# Service-level: core generation behavior
# ---------------------------------------------------------------------------


def test_generation_produces_an_answer_with_valid_evidence() -> None:
    evidence = _make_evidence(["The sky is blue.", "Water is wet."])
    model = FakeGenerationModel(response="The sky is blue because of Rayleigh scattering.")

    result = generate_answer(query="Why is the sky blue?", evidence=evidence, model=model)

    assert isinstance(result, GeneratedAnswer)
    assert result.answer == "The sky is blue because of Rayleigh scattering."
    assert result.query == "Why is the sky blue?"
    assert result.document_id == VALID_DOCUMENT_ID


def test_query_is_present_in_the_generated_prompt() -> None:
    evidence = _make_evidence(["Some evidence text."])
    model = FakeGenerationModel()

    generate_answer(query="What is the meaning of life?", evidence=evidence, model=model)

    assert len(model.prompts_received) == 1
    assert "What is the meaning of life?" in model.prompts_received[0]


def test_evidence_text_is_present_in_the_generated_prompt() -> None:
    evidence = _make_evidence(["Photosynthesis converts light into chemical energy."])
    model = FakeGenerationModel()

    generate_answer(query="What is photosynthesis?", evidence=evidence, model=model)

    assert "Photosynthesis converts light into chemical energy." in model.prompts_received[0]


def test_evidence_ordering_is_preserved_in_the_prompt() -> None:
    evidence = _make_evidence(["Alpha evidence first.", "Beta evidence second.", "Gamma evidence third."])
    model = FakeGenerationModel()

    generate_answer(query="query", evidence=evidence, model=model)

    prompt = model.prompts_received[0]
    alpha_pos = prompt.index("Alpha evidence first.")
    beta_pos = prompt.index("Beta evidence second.")
    gamma_pos = prompt.index("Gamma evidence third.")
    assert alpha_pos < beta_pos < gamma_pos


def test_prompt_construction_is_deterministic() -> None:
    evidence = _make_evidence(["Evidence one.", "Evidence two."])

    model_a = FakeGenerationModel()
    model_b = FakeGenerationModel()
    generate_answer(query="same query", evidence=evidence, model=model_a)
    generate_answer(query="same query", evidence=evidence, model=model_b)

    assert model_a.prompts_received[0] == model_b.prompts_received[0]


def test_evidence_provenance_is_preserved_exactly_in_generated_answer() -> None:
    evidence = _make_evidence(["First chunk.", "Second chunk."])
    model = FakeGenerationModel()

    result = generate_answer(query="query", evidence=evidence, model=model)

    assert result.evidence == evidence
    assert result.evidence[0].chunk_index == 0
    assert result.evidence[1].chunk_index == 1


# ---------------------------------------------------------------------------
# Context/prompt size budget behavior
# ---------------------------------------------------------------------------


def test_context_size_limit_excludes_chunks_that_do_not_fit_deterministically() -> None:
    """
    With a small token budget (word-count based in the fake model), only
    the evidence that fits should be included, in ranked order, and a
    chunk that doesn't fit must be excluded entirely — never truncated
    mid-text, and never causing chunks to be reordered or partially cut.

    Inclusion is decided by counting tokens on the actual assembled
    candidate prompt (not by summing independently-tokenized chunk
    counts), so this also proves the final returned prompt is itself
    within the model's real token budget.
    """
    # Each chunk is exactly 5 words; craft a budget that fits some but not all.
    evidence = _make_evidence(
        [
            "one two three four five",
            "six seven eight nine ten",
            "eleven twelve thirteen fourteen fifteen",
        ]
    )
    # Verified directly: the assembled prompt with exactly the first chunk
    # is 49 tokens (fake word-count tokenizer); with the first two chunks
    # it is 54 tokens. A budget of 50 therefore fits chunk 1 alone but not
    # chunks 1+2 together.
    small_model = FakeGenerationModel(max_input_tokens=50)

    generate_answer(query="q", evidence=evidence, model=small_model)
    prompt = small_model.prompts_received[0]

    assert "one two three four five" in prompt
    # With such a tight budget, later chunks must not appear at all.
    assert "six seven eight nine ten" not in prompt
    assert "eleven twelve thirteen fourteen fifteen" not in prompt

    # The actual assembled prompt itself must be within the model's budget,
    # verified using the model's own tokenizer/count_tokens — this is the
    # exact guarantee _build_prompt is responsible for providing.
    assert small_model.count_tokens(prompt) <= small_model.max_input_tokens


def test_a_later_chunk_is_excluded_when_it_would_push_the_actual_prompt_over_budget() -> None:
    """
    Explicitly proves the budget check is performed against the real
    assembled candidate prompt (instruction + accepted chunks so far +
    the next candidate chunk + question/answer scaffolding), not against
    a sum of independently-tokenized chunk lengths.
    """
    evidence = _make_evidence(["alpha beta gamma delta epsilon", "zeta eta theta iota kappa"])

    # Budget fits the first chunk's full assembled prompt but not a second.
    first_only_model = FakeGenerationModel(max_input_tokens=1000)
    full_prompt_with_one = _build_prompt("q", evidence[:1], first_only_model)
    budget_for_one_chunk_only = first_only_model.count_tokens(full_prompt_with_one)

    limited_model = FakeGenerationModel(max_input_tokens=budget_for_one_chunk_only)
    generate_answer(query="q", evidence=evidence, model=limited_model)
    prompt = limited_model.prompts_received[0]

    assert "alpha beta gamma delta epsilon" in prompt
    assert "zeta eta theta iota kappa" not in prompt
    assert limited_model.count_tokens(prompt) <= limited_model.max_input_tokens


def test_context_size_limit_behavior_is_deterministic_across_calls() -> None:
    evidence = _make_evidence([f"chunk number {i} with some filler words here" for i in range(10)])
    model_a = FakeGenerationModel(max_input_tokens=60)
    model_b = FakeGenerationModel(max_input_tokens=60)

    generate_answer(query="q", evidence=evidence, model=model_a)
    generate_answer(query="q", evidence=evidence, model=model_b)

    assert model_a.prompts_received[0] == model_b.prompts_received[0]
    # Ranked order preserved: whichever chunks were included, they must
    # appear in the same relative order as in `evidence`.
    prompt = model_a.prompts_received[0]
    positions = [prompt.index(f"chunk number {i} ") for i in range(10) if f"chunk number {i} " in prompt]
    assert positions == sorted(positions)


def test_generous_budget_includes_all_evidence() -> None:
    evidence = _make_evidence(["short one.", "short two.", "short three."])
    model = FakeGenerationModel(max_input_tokens=10_000)

    generate_answer(query="q", evidence=evidence, model=model)
    prompt = model.prompts_received[0]

    assert "short one." in prompt
    assert "short two." in prompt
    assert "short three." in prompt


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_zero_evidence_raises_no_evidence_error_and_never_calls_model() -> None:
    model = FakeGenerationModel()

    with pytest.raises(NoEvidenceError):
        generate_answer(query="q", evidence=[], model=model)

    assert model.prompts_received == []


def test_evidence_that_cannot_fit_at_all_raises_evidence_context_too_large_error() -> None:
    """
    Regression test for a fix: if evidence was supplied but not even the
    single highest-ranked chunk fits within the model's actual token
    budget, _build_prompt() must raise EvidenceContextTooLargeError
    rather than silently assembling and returning a no-evidence prompt.
    The generation model's generate() must never be called in this case.
    """
    evidence = _make_evidence(["Some evidence text that will not fit within a tiny budget."])
    tiny_model = FakeGenerationModel(max_input_tokens=1)  # smaller than even the fixed scaffolding alone

    with pytest.raises(EvidenceContextTooLargeError):
        generate_answer(query="q", evidence=evidence, model=tiny_model)

    assert tiny_model.prompts_received == []  # generate() must never be called


def test_evidence_context_too_large_is_raised_directly_by_build_prompt() -> None:
    """Unit-level check of _build_prompt() in isolation, independent of generate_answer()."""
    evidence = _make_evidence(["This single chunk alone is still too large for the budget."])
    tiny_model = FakeGenerationModel(max_input_tokens=1)

    with pytest.raises(EvidenceContextTooLargeError):
        _build_prompt("q", evidence, tiny_model)


def test_normal_evidence_within_budget_still_succeeds_after_the_fix() -> None:
    """Ensures the fix didn't regress the ordinary, fits-fine case."""
    evidence = _make_evidence(["Perfectly reasonable evidence text."])
    model = FakeGenerationModel(max_input_tokens=1000)

    result = generate_answer(query="q", evidence=evidence, model=model)

    assert result.answer == model.response
    assert "Perfectly reasonable evidence text." in model.prompts_received[0]
    assert model.count_tokens(model.prompts_received[0]) <= model.max_input_tokens


def test_generation_model_failure_raises_generation_failed_error() -> None:
    evidence = _make_evidence(["Some evidence."])

    with pytest.raises(GenerationFailedError):
        generate_answer(query="q", evidence=evidence, model=FailingGenerationModel())


def test_empty_model_output_raises_generation_failed_error() -> None:
    evidence = _make_evidence(["Some evidence."])

    with pytest.raises(GenerationFailedError):
        generate_answer(query="q", evidence=evidence, model=EmptyAnswerGenerationModel())


# ---------------------------------------------------------------------------
# API-level tests
# ---------------------------------------------------------------------------


class ApiFakeEmbeddingModel:
    """Deterministic embedding model for API tests (same pattern as other test files)."""

    def __init__(self, dimension: int = 8) -> None:
        self.dimension = dimension

    def encode(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> list[float]:
        import hashlib

        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [b / 255.0 for b in digest[: self.dimension]]


class FailingEmbeddingModel:
    def encode(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("simulated embedding failure")


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
def vector_stores_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "vector_stores"
    directory.mkdir()
    return directory


@pytest.fixture
def client(uploads_dir: Path, vector_stores_dir: Path) -> Iterator[TestClient]:
    app.dependency_overrides[get_uploads_dir] = lambda: uploads_dir
    app.dependency_overrides[get_vector_stores_dir] = lambda: vector_stores_dir
    app.dependency_overrides[get_embedding_model_instance] = lambda: ApiFakeEmbeddingModel(dimension=8)
    app.dependency_overrides[get_generation_model_instance] = lambda: FakeGenerationModel(
        response="Generated evidence-grounded answer."
    )
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.pop(get_uploads_dir, None)
    app.dependency_overrides.pop(get_vector_stores_dir, None)
    app.dependency_overrides.pop(get_embedding_model_instance, None)
    app.dependency_overrides.pop(get_generation_model_instance, None)


def test_api_generate_success(client: TestClient, uploads_dir: Path) -> None:
    page = ("Research content for the generation endpoint test. " * 30).strip()
    _write_pdf(uploads_dir, VALID_DOCUMENT_ID, [page])
    assert client.post(f"/documents/{VALID_DOCUMENT_ID}/index").status_code == 200

    response = client.post(f"/documents/{VALID_DOCUMENT_ID}/generate", json={"query": "research content", "top_k": 2})

    assert response.status_code == 200
    body = response.json()
    assert body["document_id"] == VALID_DOCUMENT_ID
    assert body["query"] == "research content"
    assert body["answer"] == "Generated evidence-grounded answer."
    assert len(body["evidence"]) >= 1
    for item in body["evidence"]:
        assert item["document_id"] == VALID_DOCUMENT_ID
        assert "score" in item
        assert "rank" in item


def test_api_generate_invalid_document_id_returns_400(client: TestClient) -> None:
    response = client.post("/documents/not-a-uuid/generate", json={"query": "hello", "top_k": 3})

    assert response.status_code == 400


def test_api_generate_missing_index_returns_404(client: TestClient) -> None:
    response = client.post(f"/documents/{VALID_DOCUMENT_ID}/generate", json={"query": "hello", "top_k": 3})

    assert response.status_code == 404


def test_api_generate_corrupted_index_returns_422(client: TestClient, uploads_dir: Path, vector_stores_dir: Path) -> None:
    _write_pdf(uploads_dir, VALID_DOCUMENT_ID, ["Some content to index for corruption test."])
    assert client.post(f"/documents/{VALID_DOCUMENT_ID}/index").status_code == 200

    metadata_path = vector_stores_dir / f"{VALID_DOCUMENT_ID}.metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["embedding_dimension"] = 999
    metadata_path.write_text(json.dumps(metadata))

    response = client.post(f"/documents/{VALID_DOCUMENT_ID}/generate", json={"query": "hello", "top_k": 3})

    assert response.status_code == 422


def test_api_generate_empty_query_returns_422_via_schema_validation(client: TestClient) -> None:
    response = client.post(f"/documents/{VALID_DOCUMENT_ID}/generate", json={"query": "", "top_k": 3})

    assert response.status_code == 422


def test_api_generate_whitespace_only_query_returns_400_via_service_check(
    client: TestClient, uploads_dir: Path
) -> None:
    _write_pdf(uploads_dir, VALID_DOCUMENT_ID, ["Some content."])
    assert client.post(f"/documents/{VALID_DOCUMENT_ID}/index").status_code == 200

    response = client.post(f"/documents/{VALID_DOCUMENT_ID}/generate", json={"query": "   ", "top_k": 3})

    assert response.status_code == 400


def test_api_generate_invalid_top_k_returns_422_via_schema_validation(client: TestClient) -> None:
    response = client.post(f"/documents/{VALID_DOCUMENT_ID}/generate", json={"query": "hello", "top_k": 0})

    assert response.status_code == 422


def test_api_generate_top_k_larger_than_available_vectors_still_succeeds(
    client: TestClient, uploads_dir: Path
) -> None:
    _write_pdf(uploads_dir, VALID_DOCUMENT_ID, ["A short single page of content."])
    assert client.post(f"/documents/{VALID_DOCUMENT_ID}/index").status_code == 200

    response = client.post(f"/documents/{VALID_DOCUMENT_ID}/generate", json={"query": "short content", "top_k": 1000})

    assert response.status_code == 200
    body = response.json()
    assert body["evidence"]
    assert len(body["evidence"]) < 1000


def test_api_generate_model_unavailable_returns_503(uploads_dir: Path, vector_stores_dir: Path, monkeypatch) -> None:
    import backend.api.routes.documents as documents_module
    from backend.services.generation_service import GenerationModelUnavailableError

    def _raise_unavailable(model_name: str):
        raise GenerationModelUnavailableError("simulated generation model load failure")

    monkeypatch.setattr(documents_module, "get_generation_model", _raise_unavailable)

    app.dependency_overrides[get_uploads_dir] = lambda: uploads_dir
    app.dependency_overrides[get_vector_stores_dir] = lambda: vector_stores_dir
    app.dependency_overrides[get_embedding_model_instance] = lambda: ApiFakeEmbeddingModel(dimension=8)
    app.dependency_overrides.pop(get_generation_model_instance, None)  # let the real dependency run

    try:
        with TestClient(app) as test_client:
            _write_pdf(uploads_dir, VALID_DOCUMENT_ID, ["Some content."])
            assert test_client.post(f"/documents/{VALID_DOCUMENT_ID}/index").status_code == 200
            response = test_client.post(
                f"/documents/{VALID_DOCUMENT_ID}/generate", json={"query": "hello", "top_k": 3}
            )
    finally:
        app.dependency_overrides.pop(get_uploads_dir, None)
        app.dependency_overrides.pop(get_vector_stores_dir, None)
        app.dependency_overrides.pop(get_embedding_model_instance, None)

    assert response.status_code == 503


def test_api_generate_embedding_failure_returns_500(client: TestClient, uploads_dir: Path) -> None:
    _write_pdf(uploads_dir, VALID_DOCUMENT_ID, ["Some content."])
    assert client.post(f"/documents/{VALID_DOCUMENT_ID}/index").status_code == 200

    app.dependency_overrides[get_embedding_model_instance] = lambda: FailingEmbeddingModel()
    response = client.post(f"/documents/{VALID_DOCUMENT_ID}/generate", json={"query": "hello", "top_k": 3})

    assert response.status_code == 500


def test_api_generate_returns_404_when_no_index_exists_due_to_empty_document(
    client: TestClient, uploads_dir: Path, vector_stores_dir: Path
) -> None:
    """
    A document whose only page is entirely blank produces zero chunks and
    therefore zero vectors -> indexing itself fails with 422 (Milestone
    2.5 behavior, unchanged). Since no index was ever successfully built,
    /generate has nothing to retrieve from and returns 404 (missing
    index) — this does NOT exercise the /generate route's own
    NoEvidenceError -> 422 mapping; see
    test_api_generate_no_evidence_from_retrieval_returns_422 for that.
    """
    _write_pdf(uploads_dir, VALID_DOCUMENT_ID, [""])

    index_response = client.post(f"/documents/{VALID_DOCUMENT_ID}/index")
    assert index_response.status_code == 422

    generate_response = client.post(f"/documents/{VALID_DOCUMENT_ID}/generate", json={"query": "hello", "top_k": 3})
    assert generate_response.status_code == 404  # no index was ever successfully built to retrieve from


def test_api_generate_no_evidence_from_retrieval_returns_422(
    client: TestClient, uploads_dir: Path, monkeypatch
) -> None:
    """
    Directly exercises the /generate route's own NoEvidenceError -> 422
    mapping: a real, successfully-built index exists, but
    retrieval_service.retrieve() (as imported into documents.py) is
    monkeypatched to return a valid RetrievalResult with zero results —
    exactly the condition generate_answer() treats as NoEvidenceError.

    This does not touch Milestone 2.5's indexing behavior at all; the
    index really is built successfully, and only the retrieval step's
    *result* is controlled for this test.
    """
    import backend.api.routes.documents as documents_module
    from backend.services.retrieval_service import RetrievalResult

    _write_pdf(uploads_dir, VALID_DOCUMENT_ID, ["Some content to index."])
    assert client.post(f"/documents/{VALID_DOCUMENT_ID}/index").status_code == 200

    def _fake_retrieve(**kwargs):
        return RetrievalResult(document_id=VALID_DOCUMENT_ID, query=kwargs["query"], total_results=0, results=[])

    monkeypatch.setattr(documents_module, "retrieve", _fake_retrieve)

    response = client.post(f"/documents/{VALID_DOCUMENT_ID}/generate", json={"query": "hello", "top_k": 3})

    assert response.status_code == 422


def test_api_generate_evidence_too_large_for_context_returns_422(
    client: TestClient, uploads_dir: Path
) -> None:
    """
    Regression test for a fix: when not even the single highest-ranked
    evidence chunk fits within the configured generation model's token
    budget, the route must map EvidenceContextTooLargeError to 422
    rather than silently generating from no evidence.
    """
    _write_pdf(uploads_dir, VALID_DOCUMENT_ID, ["Some content to index for the oversized-evidence test."])
    assert client.post(f"/documents/{VALID_DOCUMENT_ID}/index").status_code == 200

    app.dependency_overrides[get_generation_model_instance] = lambda: FakeGenerationModel(max_input_tokens=1)
    response = client.post(f"/documents/{VALID_DOCUMENT_ID}/generate", json={"query": "hello", "top_k": 3})

    assert response.status_code == 422


def test_api_generate_failure_returns_500(client: TestClient, uploads_dir: Path) -> None:
    _write_pdf(uploads_dir, VALID_DOCUMENT_ID, ["Some content."])
    assert client.post(f"/documents/{VALID_DOCUMENT_ID}/index").status_code == 200

    app.dependency_overrides[get_generation_model_instance] = lambda: FailingGenerationModel()
    response = client.post(f"/documents/{VALID_DOCUMENT_ID}/generate", json={"query": "hello", "top_k": 3})

    assert response.status_code == 500


# ---------------------------------------------------------------------------
# End-to-end regression: full pipeline through all seven endpoints
# ---------------------------------------------------------------------------


def test_full_pipeline_regression_upload_through_generate(
    client: TestClient, uploads_dir: Path, vector_stores_dir: Path
) -> None:
    valid_pdf_bytes = _make_pdf_bytes(["Regression pipeline content for generation test."])

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

    generate_response = client.post(f"/documents/{document_id}/generate", json={"query": "regression pipeline", "top_k": 3})
    assert generate_response.status_code == 200
    body = generate_response.json()
    assert body["answer"]
    assert len(body["evidence"]) >= 1


# ---------------------------------------------------------------------------
# Real model smoke test (skipped gracefully if unavailable, e.g. offline)
# ---------------------------------------------------------------------------


def test_real_huggingface_generation_smoke_test(vector_stores_dir: Path) -> None:
    """
    Attempts a real generation model (google/flan-t5-small) end-to-end.
    Skipped (not failed) if the model cannot be downloaded, matching the
    convention in test_embedding.py/test_retrieval.py.
    """
    from backend.services.generation_service import get_generation_model

    settings = get_settings()
    try:
        model = get_generation_model(settings.generation_model_name)
    except Exception as exc:
        pytest.skip(f"Real generation model unavailable in this environment: {exc}")

    evidence = _make_evidence(
        [
            "The Eiffel Tower is located in Paris, France.",
            "It was completed in 1889 for the World's Fair.",
        ]
    )

    result = generate_answer(query="Where is the Eiffel Tower located?", evidence=evidence, model=model)

    assert result.answer.strip() != ""
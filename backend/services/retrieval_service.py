"""
Retrieval.

Embeds a user query with the same Hugging Face embedding model used to
build a document's FAISS index, searches the persisted index (produced by
backend/services/vector_store_service.py), and maps results back to
chunk provenance.

This module does NOT rebuild the extraction -> chunking -> embedding ->
indexing pipeline. It only reads the already-persisted FAISS index and
metadata sidecar for a document_id and searches it. It does not implement
hybrid retrieval, reranking, generation, or citation mapping — those are
later milestones.

This module has no dependency on FastAPI.

Reused, not duplicated:
- Query embedding reuses embedding_service.embed_texts() verbatim (a
  single-item list containing the query is a valid call — no new
  embedding logic was needed).
- Index loading reuses vector_store_service.load_vector_index() verbatim,
  including all of its consistency validation.
- Query-vector normalization reuses vector_store_service.normalize_vectors()
  (the same function used when indexed vectors were normalized), so a
  query is compared against the index using an identical convention.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from backend.core.config import get_settings
from backend.services.embedding_service import EmbeddingModel, embed_texts
from backend.services.vector_store_service import load_vector_index, normalize_vectors


class RetrievalError(Exception):
    """Base class for retrieval failures."""


class EmptyQueryError(RetrievalError):
    """Raised when the query is empty or whitespace-only."""


class InvalidTopKError(RetrievalError):
    """Raised when top_k is not a positive integer."""


class EmbeddingModelMismatchError(RetrievalError):
    """
    Raised when the embedding model configured for the query does not
    match the model that was used to build the document's stored index.

    Silently searching with a mismatched model would produce meaningless
    similarity scores (different models generally do not share a common
    vector space), so this is treated as a distinct, explicit failure
    rather than allowed to proceed.
    """


class EmbeddingDimensionMismatchError(RetrievalError):
    """
    Raised when the query embedding's dimensionality does not match the
    stored index's dimensionality.

    Checked independently of EmbeddingModelMismatchError because two
    different models could theoretically share the same declared name
    yet produce different-dimension vectors (e.g. a model was upgraded in
    place), or, conversely, two different model names could coincidentally
    share a dimension — matching names does not guarantee matching
    dimensions, so both checks are kept.
    """


@dataclass
class RetrievedChunk:
    """A single ranked retrieval result, with full chunk provenance."""

    rank: int  # 1-indexed, best match first
    score: float  # cosine similarity score (higher = more similar)
    document_id: str
    page_number: int
    chunk_index: int
    chunk_index_in_page: int
    text: str
    model_name: str


@dataclass
class RetrievalResult:
    """Structured retrieval result for one query against one document's index."""

    document_id: str
    query: str
    total_results: int
    results: list[RetrievedChunk]


def retrieve(
    document_id: str,
    query: str,
    top_k: int,
    vector_stores_dir: Path,
    model: "EmbeddingModel",
    model_name: str | None = None,
    batch_size: int | None = None,
) -> RetrievalResult:
    """
    Retrieve the top-k chunks most similar to `query` from the persisted
    vector index for `document_id`.

    Raises:
        InvalidTopKError: `top_k` is not a positive integer. (Defense in
            depth — the API schema already enforces `top_k >= 1` via
            Pydantic, so this path is only reachable when calling this
            function directly, outside the API.)
        EmptyQueryError: `query` is empty or whitespace-only. (Defense in
            depth for the same reason: the API schema's `min_length=1`
            rejects a fully empty string before this function is ever
            called, but cannot detect a whitespace-only string, so this
            check still runs unconditionally here.)
        InvalidIndexIdentifierError, VectorIndexNotFoundError,
            VectorStoreCorruptionError: propagated unmodified from
            vector_store_service.load_vector_index().
        EmbeddingModelMismatchError: the query's configured embedding
            model name does not match the model that built the index.
        EmbeddingDimensionMismatchError: the query embedding's dimension
            does not match the index's dimension.
        EmbeddingGenerationError: propagated unmodified from
            embedding_service.embed_texts().

    An index with zero vectors (`index.ntotal == 0`) is not an error: this
    returns a successful, empty RetrievalResult (`total_results=0,
    results=[]`) without ever calling FAISS search with k=0, and without
    embedding the query at all, since there is nothing to compare it
    against regardless of its content.

    If `top_k` exceeds the number of vectors actually stored in the
    index, it is silently clamped to `index.ntotal` rather than treated
    as an error — the caller gets every available result rather than a
    failure.
    """
    if top_k < 1:
        raise InvalidTopKError("top_k must be at least 1.")

    if not query.strip():
        raise EmptyQueryError("Query must not be empty.")

    settings = get_settings()
    resolved_model_name = model_name or settings.embedding_model_name
    resolved_batch_size = batch_size if batch_size is not None else settings.embedding_batch_size

    index, index_result = load_vector_index(document_id, vector_stores_dir)

    if index.ntotal == 0:
        return RetrievalResult(document_id=document_id, query=query, total_results=0, results=[])

    stored_model_name = index_result.records[0].model_name
    if stored_model_name != resolved_model_name:
        raise EmbeddingModelMismatchError(
            f"The index for this document was built with embedding model "
            f"'{stored_model_name}', but the query is configured to use "
            f"'{resolved_model_name}'. Retrieval requires the same model "
            f"used to build the index."
        )

    query_vectors = embed_texts([query], model, resolved_batch_size)
    query_matrix = np.array(query_vectors, dtype="float32")

    if query_matrix.shape[1] != index.d:
        raise EmbeddingDimensionMismatchError(
            f"The query embedding has dimension {query_matrix.shape[1]}, but the "
            f"index has dimension {index.d}."
        )

    normalize_vectors(query_matrix)

    # Clamping here guarantees search() is never asked for more neighbors
    # than exist, so FAISS never has to pad results with invalid (-1)
    # indices for a flat index that simply doesn't have enough vectors.
    effective_k = min(top_k, index.ntotal)

    scores, indices = index.search(query_matrix, effective_k)

    results: list[RetrievedChunk] = []
    for rank, (position, score) in enumerate(zip(indices[0], scores[0]), start=1):
        record = index_result.records[int(position)]
        results.append(
            RetrievedChunk(
                rank=rank,
                score=float(score),
                document_id=record.document_id,
                page_number=record.page_number,
                chunk_index=record.chunk_index,
                chunk_index_in_page=record.chunk_index_in_page,
                text=record.text,
                model_name=record.model_name,
            )
        )

    return RetrievalResult(
        document_id=document_id,
        query=query,
        total_results=len(results),
        results=results,
    )
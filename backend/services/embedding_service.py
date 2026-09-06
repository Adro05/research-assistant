"""
Text embedding.

Converts chunked text (produced by backend/services/chunking_service.py)
into dense vector embeddings using a Hugging Face Sentence Transformers
model, preserving chunk-to-embedding ordering and all provenance metadata.

This module does not implement FAISS, retrieval, reranking, generation,
or any database persistence — it only turns chunks into vectors and
returns them in memory, exactly once per call, mirroring how extraction
and chunking are computed on demand rather than stored.

Design:
- The actual embedding model is never imported or loaded at module import
  time. `get_embedding_model()` lazily imports `sentence_transformers` and
  loads the configured model, caching one instance per model name so it
  is not reloaded on every call.
- `embed_chunking_result()` accepts an optional `model` argument so
  callers (and tests) can inject any object exposing an `encode(texts) ->
  Sequence[Sequence[float]]` method — the real Hugging Face model in
  production, or a lightweight deterministic fake in tests that don't
  need network access or a model download.
- Batching is performed explicitly by this module (not left implicit
  inside the model call), so batch size is a visible, testable, and
  independently configurable behavior.
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Protocol

from backend.core.config import get_settings
from backend.services.chunking_service import ChunkingResult


class EmbeddingError(Exception):
    """Base class for embedding failures."""


class EmbeddingModelUnavailableError(EmbeddingError):
    """Raised when the embedding model cannot be loaded (missing dependency, download failure, etc.)."""


class EmbeddingGenerationError(EmbeddingError):
    """Raised when the model fails to produce embeddings, or produces malformed output."""


class EmbeddingModel(Protocol):
    """Structural interface any embedding model (real or fake) must satisfy."""

    def encode(self, texts: list[str]) -> Any: ...  # returns something iterable of vectors


@dataclass
class ChunkEmbedding:
    """A single chunk's embedding, with full provenance preserved."""

    document_id: str
    page_number: int
    chunk_index: int
    chunk_index_in_page: int
    text: str
    embedding: list[float]
    model_name: str


@dataclass
class EmbeddingResult:
    """Structured embedding result for one document."""

    document_id: str
    model_name: str
    embedding_dimension: int
    total_chunks: int
    embeddings: list[ChunkEmbedding]


@lru_cache(maxsize=4)
def get_embedding_model(model_name: str) -> "EmbeddingModel":
    """
    Lazily load and cache a Sentence Transformers model by name.

    Cached so the (potentially expensive) model load happens at most once
    per distinct `model_name` per process, rather than once per call.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise EmbeddingModelUnavailableError(
            "sentence-transformers is not installed."
        ) from exc

    try:
        return SentenceTransformer(model_name)
    except Exception as exc:
        raise EmbeddingModelUnavailableError(
            f"Failed to load embedding model '{model_name}'."
        ) from exc


def _vector_to_list(vector: Any) -> list[float]:
    """Normalize a single embedding vector (numpy array, list, or tuple) to a plain list of floats."""
    if hasattr(vector, "tolist"):
        return [float(v) for v in vector.tolist()]
    return [float(v) for v in vector]


def embed_texts(
    texts: list[str],
    model: "EmbeddingModel",
    batch_size: int,
) -> list[list[float]]:
    """
    Generate embeddings for a list of texts using `model`, in batches of
    `batch_size`.

    Returns an empty list for empty input, without calling the model.
    Raises EmbeddingGenerationError if the model raises during encoding,
    or if it returns a vector count that doesn't match the input, or
    vectors of inconsistent length.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    if not texts:
        return []

    all_vectors: list[list[float]] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        try:
            raw_output = model.encode(batch)
        except Exception as exc:
            raise EmbeddingGenerationError("Failed to generate embeddings for a batch of chunks.") from exc

        batch_vectors = [_vector_to_list(vector) for vector in raw_output]

        if len(batch_vectors) != len(batch):
            raise EmbeddingGenerationError(
                "The embedding model returned a different number of vectors than input texts."
            )

        all_vectors.extend(batch_vectors)

    dimensions = {len(vector) for vector in all_vectors}
    if len(dimensions) > 1:
        raise EmbeddingGenerationError("The embedding model returned vectors of inconsistent dimensionality.")

    return all_vectors


def embed_chunking_result(
    chunking_result: ChunkingResult,
    model: "EmbeddingModel | None" = None,
    model_name: str | None = None,
    batch_size: int | None = None,
) -> EmbeddingResult:
    """
    Embed every chunk in a ChunkingResult, preserving chunk order and all
    provenance metadata (document_id, page_number, chunk_index,
    chunk_index_in_page).

    If `model` is not provided, the model configured in application
    settings is lazily loaded (and cached) via `get_embedding_model()`.
    `model_name` and `batch_size` default to application settings when
    not explicitly provided, so the model is not hardcoded into this
    function's callers.

    A ChunkingResult with zero chunks produces an EmbeddingResult with
    zero embeddings and embedding_dimension 0 — this is not an error,
    consistent with how an empty/whitespace-only page produces zero
    chunks rather than an error in chunking_service.py.
    """
    settings = get_settings()
    resolved_model_name = model_name or settings.embedding_model_name
    resolved_batch_size = batch_size if batch_size is not None else settings.embedding_batch_size
    resolved_model = model if model is not None else get_embedding_model(resolved_model_name)

    texts = [chunk.text for chunk in chunking_result.chunks]
    vectors = embed_texts(texts, resolved_model, resolved_batch_size)

    if len(vectors) != len(chunking_result.chunks):
        raise EmbeddingGenerationError("Embedding count did not match chunk count.")

    embedding_dimension = len(vectors[0]) if vectors else 0

    embeddings = [
        ChunkEmbedding(
            document_id=chunk.document_id,
            page_number=chunk.page_number,
            chunk_index=chunk.chunk_index,
            chunk_index_in_page=chunk.chunk_index_in_page,
            text=chunk.text,
            embedding=vector,
            model_name=resolved_model_name,
        )
        for chunk, vector in zip(chunking_result.chunks, vectors)
    ]

    return EmbeddingResult(
        document_id=chunking_result.document_id,
        model_name=resolved_model_name,
        embedding_dimension=embedding_dimension,
        total_chunks=len(embeddings),
        embeddings=embeddings,
    )
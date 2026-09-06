"""
FAISS vector indexing.

Builds, saves, and loads a FAISS vector index from an EmbeddingResult
(produced by backend/services/embedding_service.py), preserving a
deterministic mapping from each FAISS vector position back to the
originating chunk's provenance metadata.

This module does not implement retrieval/search as a feature (that is
Milestone 2.6), does not implement approximate/ANN indexing, does not
introduce any database, and does not depend on FastAPI.

Index type and similarity convention
-------------------------------------
Uses `faiss.IndexFlatIP` (exact, brute-force inner-product search) over
L2-normalized vectors. For unit-length vectors, inner product is
mathematically equivalent to cosine similarity, so this gives correct
cosine-similarity search behavior without silently using an incompatible
distance metric (e.g. raw L2 distance on un-normalized vectors, which
would not match how Sentence Transformers embeddings are conventionally
compared). `IndexFlatIP` is appropriate for this milestone specifically
because it is exact (no approximation/recall trade-off) and simple to
reason about and test; approximate indexes (IVF, HNSW, etc.) are
explicitly out of scope until they are actually needed at a larger scale.

Vector storage vs. metadata
----------------------------
FAISS itself stores only the raw (normalized) vectors — it is never
treated as a source of truth for provenance. A separate JSON sidecar file
stores the ordered list of chunk metadata, one entry per FAISS vector
position (`records[i]` describes the vector at FAISS position `i`), plus
index-level metadata (index type, similarity metric, dimension, vector
count). Both files are written under a configurable "vector stores"
directory, keyed by document_id.

Safe writes and safe replacement (Windows-compatible)
-------------------------------------------------------
Both files are first written out fully to temporary paths. Only once both
temporary writes succeed are the real files updated, via `Path.replace()`
(not `Path.rename()` — `.rename()` is not guaranteed to overwrite an
existing destination on Windows and can raise `FileExistsError` there,
whereas `.replace()` is explicitly documented to overwrite cross-platform,
backed by `os.replace()`). Re-indexing an already-indexed document is
supported: if an index already exists for a document_id, its current
FAISS file is moved aside as a backup before being replaced; if the
subsequent metadata replace then fails, the original index is restored
from that backup, so a failure partway through re-indexing can never
leave a mismatched (new-index/old-metadata or old-index/new-metadata)
pair on disk.
"""

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import faiss
import numpy as np

from backend.services.embedding_service import EmbeddingResult

INDEX_TYPE = "IndexFlatIP"
SIMILARITY_METRIC = "cosine"

# Same canonical lowercase-hyphenated UUID form used elsewhere in this
# project (see extraction_service.py) to validate any identifier before
# it is used to build a filesystem path. Duplicated here (rather than
# imported) so this service has no dependency on extraction_service.py.
_DOCUMENT_ID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)


class VectorStoreError(Exception):
    """Base class for vector store failures."""


class EmptyEmbeddingsError(VectorStoreError):
    """Raised when attempting to build an index from zero embeddings."""


class InconsistentDimensionError(VectorStoreError):
    """
    Raised when the embedding vectors' dimensionality is invalid or
    internally inconsistent — including: vectors of differing lengths,
    zero-length vectors, a declared `embedding_dimension` that disagrees
    with the actual vector length, or a declared `total_chunks` that
    disagrees with the actual number of embeddings provided.
    """


class InvalidIndexIdentifierError(VectorStoreError):
    """Raised when a document identifier is not a well-formed UUID."""


class VectorIndexNotFoundError(VectorStoreError):
    """Raised when no stored index/metadata exists for a given document identifier."""


class VectorStoreCorruptionError(VectorStoreError):
    """
    Raised when stored index/metadata files exist but cannot be read or
    parsed, or when they can be read but are internally inconsistent with
    each other (e.g. mismatched vector counts, dimensions, or ordering).
    """


@dataclass
class VectorRecord:
    """Metadata for the vector stored at a specific FAISS index position."""

    faiss_position: int  # 0-indexed, matches the vector's position within the FAISS index
    document_id: str
    page_number: int
    chunk_index: int
    chunk_index_in_page: int
    text: str
    model_name: str


@dataclass
class VectorIndexResult:
    """
    Structured result describing a FAISS vector index for one document.

    `records` is ordered by `faiss_position` — `records[i]` describes the
    vector stored at FAISS index position `i`. This is the deterministic
    mapping a future retrieval layer uses to translate FAISS
    nearest-neighbor results back into chunk provenance.
    """

    document_id: str
    index_type: str
    similarity_metric: str
    embedding_dimension: int
    total_vectors: int
    records: list[VectorRecord]


@dataclass
class VectorIndexPaths:
    """Filenames (relative to the vector stores directory) written for a document's index."""

    index_filename: str
    metadata_filename: str


def _validate_document_id(document_id: str) -> None:
    """Reject any identifier not matching the canonical UUID form, before it is used in a path."""
    if not _DOCUMENT_ID_PATTERN.match(document_id):
        raise InvalidIndexIdentifierError("Invalid document identifier format.")


def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize each row of `vectors` in place, leaving zero vectors as zero (no division by zero)."""
    faiss.normalize_L2(vectors)
    return vectors


def build_vector_index(embedding_result: EmbeddingResult) -> tuple[faiss.Index, VectorIndexResult]:
    """
    Build an exact, cosine-similarity FAISS index from an EmbeddingResult.

    Raises:
        InconsistentDimensionError: `embedding_result.total_chunks` does
            not match the actual number of embeddings provided; the
            embedding vectors do not all share the same length; any
            vector has zero length; or the declared `embedding_dimension`
            disagrees with the actual vector length.
        EmptyEmbeddingsError: `embedding_result` has zero embeddings. A
            FAISS index requires a fixed vector dimension at construction
            time, and there is no way to determine that dimension from
            zero vectors, so this is a distinct, explicit failure rather
            than a silently-empty or dimension-0 index.
    """
    if embedding_result.total_chunks != len(embedding_result.embeddings):
        raise InconsistentDimensionError(
            "embedding_result.total_chunks does not match the number of embeddings provided."
        )

    if embedding_result.total_chunks == 0 or not embedding_result.embeddings:
        raise EmptyEmbeddingsError(
            "Cannot build a vector index from zero embeddings: no vector dimension can be determined."
        )

    dimensions = {len(e.embedding) for e in embedding_result.embeddings}
    if len(dimensions) > 1:
        raise InconsistentDimensionError("Embedding vectors do not all share the same dimensionality.")

    dimension = dimensions.pop()
    if dimension == 0:
        raise InconsistentDimensionError(
            "Embedding vectors have zero dimensionality; cannot build a FAISS index."
        )

    if dimension != embedding_result.embedding_dimension:
        raise InconsistentDimensionError(
            "embedding_dimension does not match the actual length of the embedding vectors."
        )

    vectors = np.array([e.embedding for e in embedding_result.embeddings], dtype="float32")
    _normalize_rows(vectors)

    index = faiss.IndexFlatIP(dimension)
    index.add(vectors)

    records = [
        VectorRecord(
            faiss_position=position,
            document_id=e.document_id,
            page_number=e.page_number,
            chunk_index=e.chunk_index,
            chunk_index_in_page=e.chunk_index_in_page,
            text=e.text,
            model_name=e.model_name,
        )
        for position, e in enumerate(embedding_result.embeddings)
    ]

    result = VectorIndexResult(
        document_id=embedding_result.document_id,
        index_type=INDEX_TYPE,
        similarity_metric=SIMILARITY_METRIC,
        embedding_dimension=dimension,
        total_vectors=index.ntotal,
        records=records,
    )

    return index, result


def _index_path(vector_stores_dir: Path, document_id: str) -> Path:
    return vector_stores_dir / f"{document_id}.faiss"


def _metadata_path(vector_stores_dir: Path, document_id: str) -> Path:
    return vector_stores_dir / f"{document_id}.metadata.json"


def save_vector_index(
    index: faiss.Index,
    index_result: VectorIndexResult,
    vector_stores_dir: Path,
) -> VectorIndexPaths:
    """
    Persist a FAISS index and its metadata sidecar under `vector_stores_dir`,
    keyed by `index_result.document_id`.

    Safe to call again for a document_id that already has a stored index
    (re-indexing) — the existing pair is safely replaced. Both new files
    are fully written to temporary paths first; only once both temporary
    writes succeed are the real files updated via `Path.replace()`
    (cross-platform overwrite, including Windows). If a document already
    has a stored index, its current index file is backed up before being
    replaced; if the subsequent metadata replace fails, the original
    index is restored from that backup — so a failure partway through
    re-indexing never leaves a mismatched FAISS/metadata pair on disk.
    """
    _validate_document_id(index_result.document_id)
    vector_stores_dir.mkdir(parents=True, exist_ok=True)

    index_path = _index_path(vector_stores_dir, index_result.document_id)
    metadata_path = _metadata_path(vector_stores_dir, index_result.document_id)
    tmp_index_path = index_path.with_suffix(".faiss.part")
    tmp_metadata_path = metadata_path.with_suffix(".json.part")

    metadata_dict = {
        "document_id": index_result.document_id,
        "index_type": index_result.index_type,
        "similarity_metric": index_result.similarity_metric,
        "embedding_dimension": index_result.embedding_dimension,
        "total_vectors": index_result.total_vectors,
        "records": [asdict(record) for record in index_result.records],
    }

    # Fully write both new files to temporary paths first. If either write
    # fails, nothing about the existing (if any) stored pair is touched.
    try:
        faiss.write_index(index, str(tmp_index_path))
    except Exception as exc:
        tmp_index_path.unlink(missing_ok=True)
        raise VectorStoreError("Failed to write the FAISS index to disk.") from exc

    try:
        tmp_metadata_path.write_text(json.dumps(metadata_dict, indent=2))
    except OSError as exc:
        tmp_index_path.unlink(missing_ok=True)
        tmp_metadata_path.unlink(missing_ok=True)
        raise VectorStoreError("Failed to write index metadata to disk.") from exc

    # Both temp files are ready. Back up an existing index (if any) so we
    # can restore it if the metadata replace below fails after the index
    # replace has already succeeded — this is what prevents a mismatched
    # pair from ever being left on disk.
    old_index_backup = None
    if index_path.exists():
        old_index_backup = index_path.with_suffix(".faiss.bak")
        index_path.replace(old_index_backup)

    try:
        tmp_index_path.replace(index_path)
    except OSError as exc:
        if old_index_backup is not None:
            old_index_backup.replace(index_path)
        tmp_metadata_path.unlink(missing_ok=True)
        raise VectorStoreError("Failed to write the FAISS index to disk.") from exc

    try:
        tmp_metadata_path.replace(metadata_path)
    except OSError as exc:
        # Metadata replace failed after the index was already replaced:
        # restore the OLD index so the stored pair stays fully consistent
        # (old index + old metadata), rather than ending up with a new
        # index paired with stale or missing metadata.
        if old_index_backup is not None:
            old_index_backup.replace(index_path)
        else:
            index_path.unlink(missing_ok=True)
        tmp_metadata_path.unlink(missing_ok=True)
        raise VectorStoreError("Failed to write index metadata to disk.") from exc
    finally:
        if old_index_backup is not None:
            old_index_backup.unlink(missing_ok=True)

    return VectorIndexPaths(
        index_filename=index_path.name,
        metadata_filename=metadata_path.name,
    )


def load_vector_index(document_id: str, vector_stores_dir: Path) -> tuple[faiss.Index, VectorIndexResult]:
    """
    Load a previously saved FAISS index and its metadata sidecar for `document_id`.

    After loading, the FAISS index and its metadata sidecar are
    cross-checked against each other; any inconsistency is treated as
    corruption rather than silently returned as valid.

    Raises:
        InvalidIndexIdentifierError: `document_id` is not a well-formed UUID.
        VectorIndexNotFoundError: no index/metadata files exist for this identifier.
        VectorStoreCorruptionError: the files exist but cannot be read or
            parsed, or fail any of the following consistency checks:
            - metadata document_id matches the requested document_id
            - metadata index_type / similarity_metric match this
              module's expected index convention
            - the FAISS index's vector count matches metadata total_vectors
            - the FAISS index's dimensionality matches metadata embedding_dimension
            - the number of metadata records matches the FAISS vector count
            - every record's faiss_position matches its position in the list
    """
    _validate_document_id(document_id)

    index_path = _index_path(vector_stores_dir, document_id)
    metadata_path = _metadata_path(vector_stores_dir, document_id)

    if not index_path.is_file() or not metadata_path.is_file():
        raise VectorIndexNotFoundError("No stored vector index was found for the given identifier.")

    try:
        index = faiss.read_index(str(index_path))
    except Exception as exc:
        raise VectorStoreCorruptionError("The stored FAISS index could not be read.") from exc

    try:
        metadata_dict = json.loads(metadata_path.read_text())
        records = [VectorRecord(**record) for record in metadata_dict["records"]]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise VectorStoreCorruptionError("The stored index metadata could not be read.") from exc

    if metadata_dict.get("document_id") != document_id:
        raise VectorStoreCorruptionError(
            "The stored index metadata does not match the requested document identifier."
        )

    if metadata_dict.get("index_type") != INDEX_TYPE:
        raise VectorStoreCorruptionError("The stored index metadata has an unexpected index_type.")

    if metadata_dict.get("similarity_metric") != SIMILARITY_METRIC:
        raise VectorStoreCorruptionError("The stored index metadata has an unexpected similarity_metric.")

    if index.ntotal != metadata_dict.get("total_vectors"):
        raise VectorStoreCorruptionError("The FAISS index vector count does not match the stored metadata.")

    if index.d != metadata_dict.get("embedding_dimension"):
        raise VectorStoreCorruptionError("The FAISS index dimensionality does not match the stored metadata.")

    if len(records) != index.ntotal:
        raise VectorStoreCorruptionError(
            "The number of metadata records does not match the number of vectors in the index."
        )

    for position, record in enumerate(records):
        if record.faiss_position != position:
            raise VectorStoreCorruptionError(
                "Metadata record ordering does not match FAISS vector positions."
            )

    result = VectorIndexResult(
        document_id=metadata_dict["document_id"],
        index_type=metadata_dict["index_type"],
        similarity_metric=metadata_dict["similarity_metric"],
        embedding_dimension=metadata_dict["embedding_dimension"],
        total_vectors=metadata_dict["total_vectors"],
        records=records,
    )

    return index, result
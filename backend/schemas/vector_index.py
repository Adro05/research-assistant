"""
Pydantic schema for FAISS vector indexing responses.

Mirrors backend/services/vector_store_service.py's VectorIndexResult, but
deliberately does not include per-vector records in the API response —
only summary indexing metadata plus the storage filenames. Per-chunk
provenance is available via the chunking/embedding endpoints; this
response describes the index artifact itself.
"""

from pydantic import BaseModel, Field


class VectorIndexResponse(BaseModel):
    """Response returned after successfully building and storing a document's vector index."""

    document_id: str = Field(..., description="Identifier of the document that was indexed.")
    index_type: str = Field(..., description="FAISS index type used, e.g. 'IndexFlatIP'.")
    similarity_metric: str = Field(..., description="Similarity convention the index supports, e.g. 'cosine'.")
    embedding_dimension: int = Field(..., ge=1, description="Dimensionality of the indexed vectors.")
    total_vectors: int = Field(..., ge=1, description="Number of vectors stored in the index.")
    index_filename: str = Field(..., description="Filename of the FAISS index artifact under the vector stores directory.")
    metadata_filename: str = Field(..., description="Filename of the metadata sidecar under the vector stores directory.")
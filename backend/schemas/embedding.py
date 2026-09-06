"""
Pydantic schemas for text embeddings.

These describe the API response shape for the embeddings endpoint. The
service layer (backend/services/embedding_service.py) works with plain
dataclasses; these schemas mirror that structure for the API boundary.
"""

from pydantic import BaseModel, Field


class ChunkEmbeddingSchema(BaseModel):
    """A single chunk's embedding, with full provenance preserved."""

    document_id: str = Field(..., description="Identifier of the document this embedding belongs to.")
    page_number: int = Field(..., ge=1, description="1-indexed page number this chunk was extracted from.")
    chunk_index: int = Field(..., ge=0, description="0-indexed position of this chunk across the whole document.")
    chunk_index_in_page: int = Field(..., ge=0, description="0-indexed position of this chunk within its page only.")
    text: str = Field(..., description="The chunk's text content.")
    embedding: list[float] = Field(..., description="The dense vector embedding for this chunk's text.")
    model_name: str = Field(..., description="Name of the embedding model used to produce this vector.")


class EmbeddingResponse(BaseModel):
    """Response returned after successfully embedding a stored document's chunks."""

    document_id: str = Field(..., description="Identifier of the document that was embedded.")
    model_name: str = Field(..., description="Name of the embedding model used.")
    embedding_dimension: int = Field(..., ge=0, description="Dimensionality of each embedding vector.")
    total_chunks: int = Field(..., ge=0, description="Total number of chunks embedded.")
    embeddings: list[ChunkEmbeddingSchema] = Field(..., description="Embeddings in document order.")
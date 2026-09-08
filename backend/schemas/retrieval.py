"""
Pydantic schemas for retrieval.

Mirrors backend/services/retrieval_service.py's plain dataclasses at the
API boundary, following the same pattern as every other schema module in
this project.
"""

from pydantic import BaseModel, Field


class RetrievalRequest(BaseModel):
    """Request body for POST /documents/{document_id}/retrieve."""

    query: str = Field(..., min_length=1, description="The natural-language query to search for.")
    top_k: int = Field(default=5, ge=1, description="Number of top matching chunks to return.")


class RetrievedChunkSchema(BaseModel):
    """A single ranked retrieval result, with full chunk provenance."""

    rank: int = Field(..., ge=1, description="1-indexed rank of this result, best match first.")
    score: float = Field(..., description="Cosine similarity score for this result (higher is more similar).")
    document_id: str = Field(..., description="Identifier of the document this chunk belongs to.")
    page_number: int = Field(..., ge=1, description="1-indexed page number this chunk was extracted from.")
    chunk_index: int = Field(..., ge=0, description="0-indexed position of this chunk across the whole document.")
    chunk_index_in_page: int = Field(..., ge=0, description="0-indexed position of this chunk within its page only.")
    text: str = Field(..., description="The chunk's text content.")
    model_name: str = Field(..., description="Name of the embedding model used to produce the indexed vector.")


class RetrievalResponse(BaseModel):
    """Response returned after successfully retrieving matching chunks for a query."""

    document_id: str = Field(..., description="Identifier of the document that was searched.")
    query: str = Field(..., description="The query that was searched for.")
    total_results: int = Field(..., ge=0, description="Number of results returned.")
    results: list[RetrievedChunkSchema] = Field(..., description="Results ranked best-match first.")
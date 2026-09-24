"""
Pydantic schemas for generation.

Reuses RetrievedChunkSchema from backend/schemas/retrieval.py for the
evidence list, rather than duplicating it — the evidence shape returned
here is identical to what POST /documents/{document_id}/retrieve already
returns.
"""

from pydantic import BaseModel, Field

from backend.schemas.retrieval import RetrievedChunkSchema


class GenerationRequest(BaseModel):
    """Request body for POST /documents/{document_id}/generate."""

    query: str = Field(
        ...,
        min_length=1,
        description="The question to answer using the document's evidence.",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        description="Number of top evidence chunks to retrieve and ground the answer in.",
    )


class GenerationResponse(BaseModel):
    """Response returned after successfully generating an evidence-grounded answer."""

    document_id: str = Field(
        ...,
        description="Identifier of the document the answer was generated from.",
    )
    query: str = Field(
        ...,
        description="The query that was answered.",
    )
    answer: str = Field(
        ...,
        description="The generated, evidence-grounded answer.",
    )
    evidence: list[RetrievedChunkSchema] = Field(
        ...,
        description="The retrieved evidence the answer was grounded in, ranked best-match first.",
    )
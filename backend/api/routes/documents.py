"""
Document upload, extraction, chunking, embedding, indexing, and
retrieval endpoints.

Accepts a PDF upload, validates it, and stores it under data/uploads/.
Also exposes text extraction, chunking, embedding, FAISS vector indexing,
and retrieval for an already-uploaded document. Route handlers stay
thin — validation/storage logic lives in upload_service, extraction
logic lives in extraction_service, chunking logic lives in
chunking_service, embedding logic lives in embedding_service, FAISS
indexing logic lives in vector_store_service, and retrieval logic lives
in retrieval_service.
"""

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from backend.core.config import get_settings
from backend.schemas.chunking import ChunkingResponse, TextChunkSchema
from backend.schemas.document import PDFUploadResponse
from backend.schemas.embedding import ChunkEmbeddingSchema, EmbeddingResponse
from backend.schemas.extraction import ExtractedPageSchema, PDFExtractionResponse
from backend.schemas.retrieval import RetrievalRequest, RetrievalResponse, RetrievedChunkSchema
from backend.schemas.vector_index import VectorIndexResponse
from backend.services.chunking_service import chunk_document
from backend.services.embedding_service import (
    EmbeddingError,
    EmbeddingGenerationError,
    EmbeddingModelUnavailableError,
    embed_chunking_result,
    get_embedding_model,
)
from backend.services.extraction_service import (
    DocumentNotFoundError,
    InvalidDocumentIdError,
    PDFExtractionError,
    extract_pdf_text,
)
from backend.services.retrieval_service import (
    EmbeddingDimensionMismatchError,
    EmbeddingModelMismatchError,
    EmptyQueryError,
    InvalidTopKError,
    retrieve,
)
from backend.services.upload_service import (
    UploadValidationError,
    sanitize_filename,
    save_pdf,
    validate_extension,
    validate_not_empty,
    validate_pdf_signature,
)
from backend.services.vector_store_service import (
    EmptyEmbeddingsError,
    InconsistentDimensionError,
    InvalidIndexIdentifierError,
    VectorIndexNotFoundError,
    VectorStoreCorruptionError,
    VectorStoreError,
    build_vector_index,
    save_vector_index,
)

router = APIRouter(prefix="/documents", tags=["documents"])


def get_uploads_dir() -> Path:
    """
    FastAPI dependency resolving the upload storage directory.

    Exposed as a dependency (rather than read directly inside the route)
    so tests can override it to point at temporary storage instead of the
    real data/uploads/ directory.
    """
    return Path(get_settings().uploads_dir)


@router.post(
    "/upload",
    response_model=PDFUploadResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_pdf(
    file: UploadFile = File(...),
    uploads_dir: Path = Depends(get_uploads_dir),
) -> PDFUploadResponse:
    """Validate and store an uploaded PDF file."""
    if not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No file was provided.")

    try:
        safe_filename = sanitize_filename(file.filename)
        validate_extension(safe_filename)
    except UploadValidationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    content = await file.read()

    try:
        validate_not_empty(content)
        validate_pdf_signature(content)
    except UploadValidationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    try:
        stored = save_pdf(content, safe_filename, uploads_dir)
    except UploadValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to store the uploaded file.",
        ) from exc

    return PDFUploadResponse(
        document_id=stored.document_id,
        original_filename=stored.original_filename,
        stored_filename=stored.stored_filename,
        size_bytes=stored.size_bytes,
        upload_status="success",
    )


@router.post(
    "/{document_id}/extract",
    response_model=PDFExtractionResponse,
    status_code=status.HTTP_200_OK,
)
async def extract_document_text(
    document_id: str,
    uploads_dir: Path = Depends(get_uploads_dir),
) -> PDFExtractionResponse:
    """Extract page-by-page text from an already-uploaded, stored PDF."""
    try:
        result = extract_pdf_text(document_id, uploads_dir)
    except InvalidDocumentIdError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except DocumentNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except PDFExtractionError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process the stored document.",
        ) from exc

    return PDFExtractionResponse(
        document_id=result.document_id,
        stored_filename=result.stored_filename,
        total_pages=result.total_pages,
        pages=[ExtractedPageSchema(page_number=p.page_number, text=p.text) for p in result.pages],
    )


@router.post(
    "/{document_id}/chunks",
    response_model=ChunkingResponse,
    status_code=status.HTTP_200_OK,
)
async def chunk_document_text(
    document_id: str,
    uploads_dir: Path = Depends(get_uploads_dir),
) -> ChunkingResponse:
    """Extract and chunk text from an already-uploaded, stored PDF."""
    try:
        extraction_result = extract_pdf_text(document_id, uploads_dir)
    except InvalidDocumentIdError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except DocumentNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except PDFExtractionError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process the stored document.",
        ) from exc

    chunking_result = chunk_document(extraction_result)

    return ChunkingResponse(
        document_id=chunking_result.document_id,
        total_pages=chunking_result.total_pages,
        total_chunks=chunking_result.total_chunks,
        chunks=[
            TextChunkSchema(
                document_id=c.document_id,
                page_number=c.page_number,
                chunk_index=c.chunk_index,
                chunk_index_in_page=c.chunk_index_in_page,
                text=c.text,
                char_count=c.char_count,
            )
            for c in chunking_result.chunks
        ],
    )


def get_embedding_model_instance() -> Any:
    """
    FastAPI dependency resolving the embedding model instance.

    Exposed as a dependency (rather than resolved directly inside the
    route) so tests can override it with a lightweight, deterministic
    fake model instead of loading a real Hugging Face model over the
    network. The real model is loaded lazily and cached by
    get_embedding_model() itself, keyed by the configured model name.

    FastAPI resolves dependencies before entering the route body, so any
    exception raised here would otherwise bypass the route's own
    try/except and surface as an unhandled 500. EmbeddingModelUnavailableError
    is therefore caught and converted to HTTPException(503) right here,
    at the point of failure, rather than relying on the route to catch it.
    """
    settings = get_settings()
    try:
        return get_embedding_model(settings.embedding_model_name)
    except EmbeddingModelUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The embedding model is currently unavailable.",
        ) from exc


@router.post(
    "/{document_id}/embeddings",
    response_model=EmbeddingResponse,
    status_code=status.HTTP_200_OK,
)
async def embed_document_text(
    document_id: str,
    uploads_dir: Path = Depends(get_uploads_dir),
    embedding_model: Any = Depends(get_embedding_model_instance),
) -> EmbeddingResponse:
    """Extract, chunk, and embed text from an already-uploaded, stored PDF."""
    try:
        extraction_result = extract_pdf_text(document_id, uploads_dir)
    except InvalidDocumentIdError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except DocumentNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except PDFExtractionError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process the stored document.",
        ) from exc

    chunking_result = chunk_document(extraction_result)

    try:
        embedding_result = embed_chunking_result(chunking_result, model=embedding_model)
    except EmbeddingModelUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The embedding model is currently unavailable.",
        ) from exc
    except EmbeddingError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate embeddings for the document.",
        ) from exc

    return EmbeddingResponse(
        document_id=embedding_result.document_id,
        model_name=embedding_result.model_name,
        embedding_dimension=embedding_result.embedding_dimension,
        total_chunks=embedding_result.total_chunks,
        embeddings=[
            ChunkEmbeddingSchema(
                document_id=e.document_id,
                page_number=e.page_number,
                chunk_index=e.chunk_index,
                chunk_index_in_page=e.chunk_index_in_page,
                text=e.text,
                embedding=e.embedding,
                model_name=e.model_name,
            )
            for e in embedding_result.embeddings
        ],
    )


def get_vector_stores_dir() -> Path:
    """
    FastAPI dependency resolving the vector store artifact directory.

    Exposed as a dependency (rather than read directly inside the route)
    so tests can override it to point at temporary storage instead of the
    real data/vector_stores/ directory.
    """
    return Path(get_settings().vector_stores_dir)


@router.post(
    "/{document_id}/index",
    response_model=VectorIndexResponse,
    status_code=status.HTTP_200_OK,
)
async def index_document(
    document_id: str,
    uploads_dir: Path = Depends(get_uploads_dir),
    vector_stores_dir: Path = Depends(get_vector_stores_dir),
    embedding_model: Any = Depends(get_embedding_model_instance),
) -> VectorIndexResponse:
    """
    Extract, chunk, embed, and build a FAISS vector index for an
    already-uploaded, stored PDF, persisting the index under
    data/vector_stores/.

    There is no persisted embeddings store to index "already-generated"
    embeddings from (Milestone 2.4 computes embeddings on demand and
    returns them directly, without storing them) — so, consistent with
    every prior stage in this pipeline, this endpoint reruns extraction
    and chunking and embedding itself before indexing. The FAISS index
    and its metadata sidecar are the first artifacts in this pipeline
    that are actually persisted to disk.
    """
    try:
        extraction_result = extract_pdf_text(document_id, uploads_dir)
    except InvalidDocumentIdError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except DocumentNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except PDFExtractionError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process the stored document.",
        ) from exc

    chunking_result = chunk_document(extraction_result)

    try:
        embedding_result = embed_chunking_result(chunking_result, model=embedding_model)
    except EmbeddingModelUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The embedding model is currently unavailable.",
        ) from exc
    except EmbeddingError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate embeddings for the document.",
        ) from exc

    try:
        index, index_result = build_vector_index(embedding_result)
    except EmptyEmbeddingsError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="The document produced no chunks to index.",
        ) from exc
    except InconsistentDimensionError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to build the vector index for the document.",
        ) from exc

    try:
        paths = save_vector_index(index, index_result, vector_stores_dir)
    except VectorStoreError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save the vector index for the document.",
        ) from exc

    return VectorIndexResponse(
        document_id=index_result.document_id,
        index_type=index_result.index_type,
        similarity_metric=index_result.similarity_metric,
        embedding_dimension=index_result.embedding_dimension,
        total_vectors=index_result.total_vectors,
        index_filename=paths.index_filename,
        metadata_filename=paths.metadata_filename,
    )


@router.post(
    "/{document_id}/retrieve",
    response_model=RetrievalResponse,
    status_code=status.HTTP_200_OK,
)
async def retrieve_document_chunks(
    document_id: str,
    request: RetrievalRequest,
    vector_stores_dir: Path = Depends(get_vector_stores_dir),
    embedding_model: Any = Depends(get_embedding_model_instance),
) -> RetrievalResponse:
    """
    Embed a query with the configured embedding model and retrieve the
    top-k most similar chunks from an already-indexed document's
    persisted FAISS index.

    Does not rebuild extraction/chunking/embedding/indexing — this reads
    only the artifacts already persisted by POST /{document_id}/index.
    """
    settings = get_settings()

    try:
        result = retrieve(
            document_id=document_id,
            query=request.query,
            top_k=request.top_k,
            vector_stores_dir=vector_stores_dir,
            model=embedding_model,
            model_name=settings.embedding_model_name,
        )
    except InvalidIndexIdentifierError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except EmptyQueryError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except InvalidTopKError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except VectorIndexNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except VectorStoreCorruptionError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except (EmbeddingModelMismatchError, EmbeddingDimensionMismatchError) as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except EmbeddingGenerationError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate an embedding for the query.",
        ) from exc

    return RetrievalResponse(
        document_id=result.document_id,
        query=result.query,
        total_results=result.total_results,
        results=[
            RetrievedChunkSchema(
                rank=r.rank,
                score=r.score,
                document_id=r.document_id,
                page_number=r.page_number,
                chunk_index=r.chunk_index,
                chunk_index_in_page=r.chunk_index_in_page,
                text=r.text,
                model_name=r.model_name,
            )
            for r in result.results
        ],
    )